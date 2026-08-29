import asyncio
import urllib.parse
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List

import streamlit as st
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv
import subprocess
import sys
import concurrent.futures
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from corpus import (
    UNTRUSTED_DATA_NOTICE,
    SourceEntry,
    has_full_page,
    merge_scraped,
    render_corpus,
    seed_snippet,
    wrap_untrusted,
)
from history_store import build_record, load_records, save_record, search_records

# -- CONFIG & SECRETS -- #
load_dotenv()

@st.cache_resource
def install_playwright():
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as exc:
        st.error(f"Failed to install Playwright's Chromium browser: {exc.stderr.decode(errors='ignore')}")
        st.stop()

install_playwright()

def _load_secret(name: str) -> None:
    """Promote a Streamlit secret into the environment unless it is already set."""
    if os.getenv(name):
        return
    try:
        value = st.secrets[name]
    except Exception:
        # No secrets.toml, or the key is absent — env/.env stays authoritative.
        return
    if value:
        os.environ[name] = str(value)

for _secret_name in ("GOOGLE_API_KEY", "JINA_API_KEY"):
    _load_secret(_secret_name)

MAX_SCRAPED_CHARS = 80_000
MAX_CHARS_PER_PAGE = 15_000

# Default scraping knobs. The UI builds a ScrapeConfig from these and threads it
# explicitly into the run — nothing mutates module state at runtime.
_DEFAULT_SCRAPE_CONCURRENCY = 6
_DEFAULT_PER_DOMAIN_CONCURRENCY = 2
_DEFAULT_URL_TIMEOUT = 30  # seconds
_DEFAULT_RESOURCE_BLOCKING = True
_DEFAULT_MAX_RETRIES = 2

# A page yielding fewer than this many characters counts as a failed scrape
# (bot walls and consent interstitials typically return a very short body).
_MIN_USEFUL_CHARS = 300

# -- LLM CALL RELIABILITY --
_LLM_TIMEOUT = 120          # seconds per LLM request
_LLM_MAX_RETRIES = 3        # client-level transient-error retries (429 / 5xx), with backoff
_LLM_CHAIN_ATTEMPTS = 2     # chain-level retries — covers structured-output parse failures

# -- SCRAPING API FALLBACK (Jina Reader) --
# When the headless browser fails, times out, or is served a block page, the URL
# is re-fetched through Jina's reader API, which renders it server-side from a
# different IP. Works without a key at a low rate limit; JINA_API_KEY raises it.
_DEFAULT_API_FALLBACK_ENABLED = True
_JINA_READER_URL = "https://r.jina.ai/"
_API_TIMEOUT = 45  # seconds — the reader renders the page before responding
_MAX_API_CONCURRENCY = 3
_API_RETRY_DELAY = 2.0  # seconds to back off after a 429
_MAX_API_ATTEMPTS = 2

# -- SEARCH (Jina Search) --
# Unlike the reader endpoint, s.jina.ai rejects unauthenticated requests, so
# JINA_API_KEY is required for the agent to discover any URLs at all.
_JINA_SEARCH_URL = "https://s.jina.ai/"
_SEARCH_RATE_LIMIT_DELAY = 0.5  # seconds between sequential search requests
_SEARCH_RESULT_LIMIT = 5  # results kept per query
_SEARCH_TIMEOUT = 60  # seconds — search runs a live crawl server-side

# -- LOGGING -- #
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_filename = LOG_DIR / f"osint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, "w"),
    ],
)
logger = logging.getLogger("osint_agent")

# -- AGENT STATE & MODELS -- #
class ResearcherState(TypedDict):
    objective: str
    selected_model: str
    search_queries: List[str]
    visited_urls: List[str]
    sources: dict  # url -> SourceEntry; the authoritative corpus
    scraped_data: str  # rendered, char-capped view of `sources` for the LLM nodes
    needs_more_info: bool
    final_report: str
    iteration_count: int
    total_queries_run: int # NEW: Added to track all historical queries
    missing_aspects: List[str]
    api_scraped_count: int # Pages recovered by the scraping API fallback

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 2-3 targeted search queries to find the missing information.")

class Evaluation(BaseModel):
    is_complete: bool = Field(description="True if scraped data fully answers the objective. False if information is missing.")
    reasoning: str = Field(description="Why you made this decision.")
    missing_aspects: List[str] = Field(description="Specific pieces of information still needed. Empty if complete.")


@dataclass(frozen=True)
class ScrapeConfig:
    """Per-run scraping knobs, captured from the UI when a run starts."""
    global_concurrency: int = _DEFAULT_SCRAPE_CONCURRENCY
    per_domain_concurrency: int = _DEFAULT_PER_DOMAIN_CONCURRENCY
    url_timeout: int = _DEFAULT_URL_TIMEOUT
    max_retries: int = _DEFAULT_MAX_RETRIES
    resource_blocking: bool = _DEFAULT_RESOURCE_BLOCKING
    api_fallback_enabled: bool = _DEFAULT_API_FALLBACK_ENABLED


def _make_llm(model: str) -> ChatGoogleGenerativeAI:
    """LLM client with an explicit timeout and transient-error retries."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.2,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )

# -- HELPER FUNCTIONS -- #
def _jina_headers(extra: dict | None = None) -> dict:
    """Shared Jina auth header. The reader works without a key; search does not."""
    headers = {"Accept": "application/json", **(extra or {})}
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _jina_search(http_client: httpx.AsyncClient, query: str) -> list[dict]:
    """Run one web search. Returns a normalised [{url, description}], [] on failure."""
    headers = _jina_headers({"X-Respond-With": "no-content"})

    for attempt in range(_MAX_API_ATTEMPTS):
        try:
            response = await http_client.get(
                _JINA_SEARCH_URL,
                params={"q": query},
                headers=headers,
                timeout=_SEARCH_TIMEOUT,
            )
            if response.status_code == 429:
                logger.warning(
                    "SEARCH RATE LIMITED %r (attempt %d/%d)",
                    query, attempt + 1, _MAX_API_ATTEMPTS
                )
                await asyncio.sleep(_API_RETRY_DELAY)
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except httpx.HTTPError as exc:
            logger.warning("Search failed for %r | %s: %s", query, type(exc).__name__, exc)
            return []
        except ValueError as exc:
            logger.warning("Search returned non-JSON for %r | %s", query, exc)
            return []
    else:
        return []

    results = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        logger.warning(
            "Search returned no results for %r | %s",
            query, payload.get("readableMessage", payload) if isinstance(payload, dict) else payload
        )
        return []

    normalised = []
    for item in results:
        if len(normalised) >= _SEARCH_RESULT_LIMIT:
            break
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        normalised.append({
            "url": url,
            "description": item.get("description") or item.get("content") or "",
        })
    logger.info("Search %r returned %d results", query, len(normalised))
    return normalised


def _normalise(text: str) -> str:
    return " ".join(text.split())[:MAX_CHARS_PER_PAGE]


async def _scrape_via_api(http_client: httpx.AsyncClient, url: str) -> str:
    """Fetch a page through the Jina reader API. Returns "" when it is unavailable."""
    headers = _jina_headers({"Accept": "text/plain", "X-Return-Format": "text"})

    for attempt in range(_MAX_API_ATTEMPTS):
        try:
            response = await http_client.get(
                f"{_JINA_READER_URL}{url}", headers=headers, timeout=_API_TIMEOUT
            )
            if response.status_code == 429:
                logger.warning(
                    "API RATE LIMITED %s (attempt %d/%d)", url, attempt + 1, _MAX_API_ATTEMPTS
                )
                await asyncio.sleep(_API_RETRY_DELAY)
                continue
            response.raise_for_status()
            return _normalise(response.text)
        except httpx.HTTPError as exc:
            logger.warning("API FAILED %s | %s: %s", url, type(exc).__name__, exc)
            break
    return ""

# -- NODES -- #
async def planner_node(state: ResearcherState):
    structured_llm = _make_llm(state["selected_model"]).with_structured_output(SearchQueries)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert OSINT researcher. Your job is to generate HIGHLY SPECIFIC web search queries.
        Rules:
        - Include exact names, model numbers, dates, or identifiers when known
        - Target authoritative sources (official sites, technical docs, reputable news)
        - Never generate a query already covered by the data below
        - Each query must target a DIFFERENT aspect of the objective
        - Prefer queries that would appear on the page you want (e.g. "BMW M4 0-60 mph" not "BMW M4 performance")

        """ + UNTRUSTED_DATA_NOTICE),
        ("user", """Objective: {objective}

        Data gathered so far:
        {scraped_data}

        Still missing (from evaluator): {missing_aspects}

        Generate queries SPECIFICALLY targeting the missing aspects above.""")
    ])

    chain = (prompt | structured_llm).with_retry(stop_after_attempt=_LLM_CHAIN_ATTEMPTS)
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": wrap_untrusted(state.get("scraped_data", "")),
        "missing_aspects": state.get("missing_aspects") or "None",
    })

    return {
        "search_queries": response.queries,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "total_queries_run": state.get("total_queries_run", 0) + len(response.queries) # NEW: Tally up the total
    }

async def search_scraper_node(state: ResearcherState, *, config: ScrapeConfig):
    # `sources` (url -> SourceEntry) is the authoritative corpus. We only ever
    # upgrade an entry (snippet -> full page), never rebuild it with string surgery.
    sources: dict[str, SourceEntry] = dict(state.get("sources", {}))

    # -- WEB SEARCH -- (sequential to stay well inside the rate limit)
    all_results = []
    async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT) as http_client:
        for i, q in enumerate(state["search_queries"]):
            if i > 0:
                await asyncio.sleep(_SEARCH_RATE_LIMIT_DELAY)
            all_results.append(await _jina_search(http_client, q))

    urls_to_scrape: list[str] = []
    for results in all_results:
        for r in results:
            url = r.get("url")
            if not url:
                continue
            if has_full_page(sources, url):
                continue  # already scraped a real page for this URL
            seed_snippet(sources, url, r.get("description", ""))
            if url not in urls_to_scrape:
                urls_to_scrape.append(url)

    snippet_count = sum(1 for e in sources.values() if e["tier"] == "snippet")
    logger.info(
        "Search seeded %d snippet source(s); %d URL(s) queued for scraping",
        snippet_count, len(urls_to_scrape)
    )

    # -- PLAYWRIGHT SCRAPING --
    async def _block_route(route, request):
        if request.resource_type in ("image", "stylesheet", "font", "media"):
            await route.abort()
        else:
            await route.continue_()

    global_sem = asyncio.Semaphore(config.global_concurrency)
    domain_sems: dict[str, asyncio.Semaphore] = {}

    def _domain_sem(domain: str) -> asyncio.Semaphore:
        if domain not in domain_sems:
            domain_sems[domain] = asyncio.Semaphore(config.per_domain_concurrency)
        return domain_sems[domain]

    async def _attempt(u: str, browser) -> str:
        page = await browser.new_page()
        try:
            await Stealth().apply_stealth_async(page)
            if config.resource_blocking:
                await page.route("**/*", _block_route)
            await page.goto(u, wait_until="domcontentloaded", timeout=config.url_timeout * 1000)
            try:
                await page.wait_for_function(
                    "() => document.body.innerText.length > 200",
                    timeout=5000
                )
            except Exception:
                pass
            content = await page.evaluate("() => document.body.innerText")
            title = await page.title()
            result = _normalise(content)
            logger.info("SUCCESS %s | title=%s | chars=%d", u, title, len(result))
            return result
        finally:
            await page.close()

    api_sem = asyncio.Semaphore(_MAX_API_CONCURRENCY)

    async def _crawl(u: str, browser, api_client: httpx.AsyncClient):
        domain = urllib.parse.urlparse(u).netloc
        result = ""
        via = "browser"
        async with global_sem, _domain_sem(domain):
            for attempt in range(config.max_retries + 1):
                try:
                    result = await _attempt(u, browser)
                    break
                except Exception as exc:
                    logger.warning(
                        "FAILED %s (attempt %d/%d) | %s: %s",
                        u, attempt + 1, config.max_retries + 1, type(exc).__name__, exc
                    )

        # The browser gave us nothing usable — retry through the scraping API.
        if config.api_fallback_enabled and len(result) < _MIN_USEFUL_CHARS:
            async with api_sem:
                api_result = await _scrape_via_api(api_client, u)
            if len(api_result) > len(result):
                logger.info("API FALLBACK %s | chars=%d", u, len(api_result))
                result, via = api_result, "api"

        return u, result, via

    crawled: list[tuple[str, str, str]] = []
    if urls_to_scrape:
        async with async_playwright() as p, httpx.AsyncClient(
            timeout=_API_TIMEOUT, follow_redirects=True
        ) as api_client:
            browser = await p.chromium.launch(headless=True)
            try:
                crawled = await asyncio.gather(
                    *[_crawl(url, browser, api_client) for url in urls_to_scrape]
                )
            finally:
                await browser.close()

    # -- MERGE: upgrade snippet entries to full page content where scraping worked --
    api_recovered = 0
    for url, page_content, via in crawled:
        if merge_scraped(sources, url, page_content, via, _MIN_USEFUL_CHARS) and via == "api":
            api_recovered += 1

    scraped_data = render_corpus(sources, MAX_SCRAPED_CHARS)
    full_pages = sum(1 for e in sources.values() if e["tier"] != "snippet")
    logger.info(
        "Corpus: %d rendered chars, %d source(s) (%d full pages, %d recovered via scraping API)",
        len(scraped_data), len(sources), full_pages, api_recovered
    )

    return {
        "sources": sources,
        "scraped_data": scraped_data,
        "visited_urls": list(sources.keys()),
        "api_scraped_count": state.get("api_scraped_count", 0) + api_recovered
    }

async def evaluator_node(state: ResearcherState):
    # Skip LLM call if there's nothing to evaluate yet
    if not state.get("scraped_data", "").strip():
        logger.info("Evaluator skipped — no data yet.")
        return {"needs_more_info": True, "missing_aspects": []}

    structured_llm = _make_llm(state["selected_model"]).with_structured_output(Evaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a quality assurance AI. Check if the source data satisfies the "
         "objective.\n\n" + UNTRUSTED_DATA_NOTICE),
        ("user", "Objective: {objective}\n\nSource Data:\n{scraped_data}")
    ])
    chain = (prompt | structured_llm).with_retry(stop_after_attempt=_LLM_CHAIN_ATTEMPTS)
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": wrap_untrusted(state["scraped_data"])
    })
    return {
    "needs_more_info": not response.is_complete,
    "missing_aspects": response.missing_aspects
}

async def reporter_node(state: ResearcherState):
    scraped_data = state.get("scraped_data", "").strip()
    
    if not scraped_data:
        return {"final_report": "⚠️ No data was collected. Try a different objective or check the logs for scraping errors."}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent analyst. Using only the provided source data, "
            "write a structured Markdown report. Include: an executive summary, "
            "key findings organised by theme, and a source list. "
            "Do not invent information not present in the data.\n\n"
            + UNTRUSTED_DATA_NOTICE
            )),
        ("user", "Objective: {objective}\n\nSource Data:\n{scraped_data}")
    ])
    chain = (prompt | _make_llm(state["selected_model"])).with_retry(
        stop_after_attempt=_LLM_CHAIN_ATTEMPTS
    )

    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": wrap_untrusted(state["scraped_data"])
    })
    
    content = response.content
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = content
    return {"final_report": text}

_MAX_ITERATIONS = 3  # planner/search/evaluator loops before the report is forced

def should_continue(state: ResearcherState):
    if state.get("needs_more_info") and state.get("iteration_count", 0) < _MAX_ITERATIONS:
        return "continue"
    else:
        return "finish"

# -- GRAPH BUILDER -- #
def build_graph(scrape_config: ScrapeConfig):
    workflow = StateGraph(ResearcherState)

    async def _search_scraper(state: ResearcherState):
        return await search_scraper_node(state, config=scrape_config)

    workflow.add_node("planner", planner_node)
    workflow.add_node("search_scraper", _search_scraper)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "search_scraper")
    workflow.add_edge("search_scraper", "evaluator")

    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {"continue": "planner", "finish": "reporter"}
    )
    workflow.add_edge("reporter", END)
    return workflow.compile()

# -- ASYNC RUNNER -- #
async def run_agent_workflow(objective, selected_model, scrape_config, status_container, metric_containers):
    q_metric, u_metric, c_metric = metric_containers

    app = build_graph(scrape_config)
    initial_state = {
        "objective": objective,
        "selected_model": selected_model,
        "search_queries": [],
        "visited_urls": [],
        "sources": {},
        "scraped_data": "",
        "needs_more_info": True,
        "final_report": "",
        "iteration_count": 0,
        "total_queries_run": 0,
        "missing_aspects": [],
        "api_scraped_count": 0
    }
    
    final_report = ""

    # Local trackers for UI
    current_queries = 0
    current_urls = 0
    current_chars = 0
    visited_urls: List[str] = []
    api_scraped_count = 0

    async for output in app.astream(initial_state):
        for node_name, state_update in output.items():
            
            # --- NEW: Update live numbers ---
            if "total_queries_run" in state_update:
                current_queries = state_update["total_queries_run"]
            if "visited_urls" in state_update:
                visited_urls = state_update["visited_urls"]
                current_urls = len(visited_urls)
            if "scraped_data" in state_update:
                current_chars = len(state_update["scraped_data"])
            if "api_scraped_count" in state_update:
                api_scraped_count = state_update["api_scraped_count"]
                
            q_metric.metric("Queries Run", current_queries)
            u_metric.metric("Sites Scraped", current_urls)
            c_metric.metric("Chars Collected", current_chars)

            if node_name == "planner":
                status_container.write(f"🧠 **Planner generated queries:** {', '.join(state_update.get('search_queries', []))}")
            elif node_name == "search_scraper":
                via_api = state_update.get("api_scraped_count", 0)
                api_note = f" ({via_api} recovered via scraping API)" if via_api else ""
                status_container.write(
                    f"🕵️ **Scraping data from:** {len(state_update.get('visited_urls', []))} total sources{api_note}..."
                )
            elif node_name == "evaluator":
                needs_more = state_update.get('needs_more_info')
                if needs_more:
                    status_container.write("⚠️ **Evaluator:** Information incomplete. Looping back for more data.")
                else:
                    status_container.write("✅ **Evaluator:** Data gathering complete! Writing report...")
            elif node_name == "reporter":
                status_container.write("📝 **Reporter:** Report compiled successfully.")
                final_report = state_update.get("final_report", "")

    return {
        "report": final_report,
        "sources": visited_urls,
        "queries_run": current_queries,
        "chars_collected": current_chars,
        "api_scraped_count": api_scraped_count,
    }

# -- STREAMLIT UI -- #
st.set_page_config(page_title="OSINT Agent", page_icon="🕵️‍♂️", layout="centered")

st.title("🕵️‍♂️ Autonomous OSINT Agent")
st.markdown("Enter a research objective. The agent will autonomously plan, search, scrape, and evaluate until it has enough data to write a comprehensive report.")

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("JINA_API_KEY"):
    st.error(
        "Missing API Keys! Please ensure GOOGLE_API_KEY and JINA_API_KEY are set "
        "in your .env or Streamlit Secrets. Get a Jina key at https://jina.ai/api-dashboard/"
    )
    st.stop()
    
# --- Model Selection UI ---
col_model, col_empty = st.columns([1, 2])
with col_model:
    selected_model = st.selectbox(
        "Brain Power:",
        options=[
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview"
        ],
        index=0,
        help="Flash is faster and cheaper. Pro is better at complex reasoning and evaluation."
    )

objective = st.text_input("Research Objective:", placeholder="e.g., Identify the key capabilities of the BMW M4")

st.divider()

# 1. MOVED OUTSIDE: Show the adjusters before the button is pressed
st.subheader("Scrape Tuning (per-run knobs)")
col_a, col_b = st.columns(2)
with col_a:
    ui_global = st.number_input("Global concurrency", min_value=1, max_value=20, value=_DEFAULT_SCRAPE_CONCURRENCY, key="ui_global_concurrency")
    ui_domain = st.number_input("Per-domain concurrency", min_value=1, max_value=20, value=_DEFAULT_PER_DOMAIN_CONCURRENCY, key="ui_domain_concurrency")
with col_b:
    ui_timeout = st.number_input("Per-URL timeout (s)", min_value=5, max_value=120, value=_DEFAULT_URL_TIMEOUT, key="ui_timeout")
    ui_retries = st.number_input("Max retries per URL", min_value=0, max_value=5, value=_DEFAULT_MAX_RETRIES, key="ui_max_retries")
ui_block = st.checkbox("Block heavy resources (images, css, fonts)", value=_DEFAULT_RESOURCE_BLOCKING, key="ui_resource_blocking")
ui_api_fallback = st.checkbox(
    "Scraping API fallback (Jina Reader) when the browser is blocked",
    value=_DEFAULT_API_FALLBACK_ENABLED,
    key="ui_api_fallback",
    help="Re-fetches pages the headless browser could not read through r.jina.ai, "
         "which renders them server-side from a different IP. Those URLs are sent to "
         "Jina. Works without a key at a low rate limit — set JINA_API_KEY to raise it.",
)

def _scrape_config_from_ui() -> ScrapeConfig:
    """Snapshot the tuning widgets into an immutable config for one run."""
    return ScrapeConfig(
        global_concurrency=int(ui_global),
        per_domain_concurrency=int(ui_domain),
        url_timeout=int(ui_timeout),
        max_retries=int(ui_retries),
        resource_blocking=bool(ui_block),
        api_fallback_enabled=bool(ui_api_fallback),
    )

# 2. THE BUTTON BLOCK
if st.button("Start Research", type="primary"):
    if not objective.strip():
        st.warning("Please enter an objective first.")
    else:
        scrape_config = _scrape_config_from_ui()

        # Keep the metric placeholders in here so they appear when the run starts
        col1, col2, col3 = st.columns(3)
        q_metric = col1.empty()
        u_metric = col2.empty()
        c_metric = col3.empty()
        
        q_metric.metric("Queries Run", 0)
        u_metric.metric("Sites Scraped", 0)
        c_metric.metric("Chars Collected", 0)
        
        with st.status("Agent initialized. Starting research loop...", expanded=True) as status:
            try:
                # Grab the current Streamlit context
                ctx = get_script_run_ctx()

                def run_in_thread(objective, model, cfg, status, metrics):
                    add_script_run_ctx(ctx=ctx)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            run_agent_workflow(objective, model, cfg, status, metrics)
                        )
                    finally:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_in_thread, objective, selected_model, scrape_config, status, (q_metric, u_metric, c_metric))
                    run_result = future.result()

                status.update(label="Research Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="An error occurred", state="error")
                st.error(f"Error details: {e}")
                run_result = None

        final_report = run_result.get("report") if run_result else None

        if final_report:
            # Persist the run so it can be re-read later without researching again.
            try:
                saved_path = save_record(build_record(
                    objective,
                    selected_model,
                    final_report,
                    sources=run_result.get("sources", []),
                    queries_run=run_result.get("queries_run", 0),
                    chars_collected=run_result.get("chars_collected", 0),
                    api_scraped_count=run_result.get("api_scraped_count", 0),
                ))
                st.caption(f"💾 Saved to history ({saved_path.name})")
            except OSError as exc:
                st.warning(f"Could not save this run to history: {exc}")

            st.subheader("Final Report")
            st.markdown(final_report)

            with st.expander("View Raw Markdown"):
                st.code(final_report, language="markdown")

# -- HISTORY SEARCH -- #
st.divider()
st.subheader("📚 Past Research")

_history = load_records()
if not _history:
    st.caption("No saved research yet. Completed runs are stored automatically and appear here.")
else:
    history_query = st.text_input(
        "Search past research:",
        placeholder="Filter by objective, report text, or source URL…",
        key="history_search",
    )
    matches = search_records(_history, history_query)
    st.caption(
        f"Showing {len(matches)} of {len(_history)} saved run(s)."
        if history_query.strip()
        else f"{len(_history)} saved run(s)."
    )

    for record in matches:
        with st.expander(f"{record.objective or '(no objective)'}  —  {record.created_display}"):
            meta = [
                f"**Model:** {record.model or 'unknown'}",
                f"**Queries:** {record.queries_run}",
                f"**Sources:** {len(record.sources)}",
                f"**Chars collected:** {record.chars_collected:,}",
            ]
            if record.api_scraped_count:
                meta.append(f"**Recovered via scraping API:** {record.api_scraped_count}")
            st.caption("  ·  ".join(meta))

            report_tab, raw_tab, sources_tab = st.tabs(["Report", "Raw Markdown", "Sources"])
            with report_tab:
                st.markdown(record.report or "_This run produced no report._")
            with raw_tab:
                st.code(record.report, language="markdown")
            with sources_tab:
                if record.sources:
                    st.markdown("\n".join(f"- {src}" for src in record.sources))
                else:
                    st.caption("No sources recorded for this run.")