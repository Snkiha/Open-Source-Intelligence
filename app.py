import asyncio
import time
import urllib.parse
import os
import logging
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
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
import subprocess
import sys
import concurrent.futures
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# -- CONFIG & SECRETS -- #
load_dotenv()

@st.cache_resource
def install_playwright():
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
        capture_output=True
    )

install_playwright()

if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if not os.getenv("TAVILY_API_KEY") and "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

MAX_SCRAPED_CHARS = 80_000
MAX_CHARS_PER_PAGE = 15_000

# Concurrency controls for scraping
_MAX_SCRAPE_CONCURRENCY = 6
_MAX_SCRAPE_PER_DOMAIN_CONCURRENCY = 2
_scrape_semaphore = None
_domain_semaphores = {}

# Runtime scrapers metrics (domain-level latency and success rates)
_DOMAIN_LATENCY_SUM = {}
_DOMAIN_LATENCY_COUNT = {}
_DOMAIN_TOTAL = {}
_DOMAIN_SUCCESS = {}

_URL_TIMEOUT = 30  # seconds (configurable via UI)
_RESOURCE_BLOCKING = True
_MAX_RETRIES = 2

def _get_global_scrape_semaphore():
    global _scrape_semaphore
    if _scrape_semaphore is None:
        _scrape_semaphore = asyncio.Semaphore(_MAX_SCRAPE_CONCURRENCY)
    return _scrape_semaphore

def _get_domain_semaphore(domain: str):
    if domain not in _domain_semaphores:
        _domain_semaphores[domain] = asyncio.Semaphore(_MAX_SCRAPE_PER_DOMAIN_CONCURRENCY)
    return _domain_semaphores[domain]

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
    scraped_data: str
    needs_more_info: bool
    final_report: str
    iteration_count: int
    total_queries_run: int # NEW: Added to track all historical queries
    _SCRAPE_CONCURRENCY = 6
    _SCRAPE_PER_DOMAIN_CONCURRENCY = 2

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 2-3 targeted search queries to find the missing information.")

class Evaluation(BaseModel):
    is_complete: bool = Field(description="True if scraped data fully answers the objective. False if information is missing.")
    reasoning: str = Field(description="Why you made this decision.")

# -- HELPER FUNCTIONS -- #
async def scrape_deep_content(url):
    domain = urllib.parse.urlparse(url).netloc
    dom_sem = _get_domain_semaphore(domain)
    glob_sem = _get_global_scrape_semaphore()
    async with dom_sem:
        async with glob_sem:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                # Block non-essential resources to speed up scraping (configurable)
                if _RESOURCE_BLOCKING:
                    async def _block_route(route, request):
                        rt = request.resource_type
                        if rt in ("image", "stylesheet", "font", "media"):
                            await route.abort()
                        else:
                            await route.continue_()
                    try:
                        await page.route("**/*", _block_route)
                    except Exception:
                        pass
                # Apply stealth to bypass basic bot detection
                await Stealth().apply_stealth_async(page)
                try:
                    logger.info("Scraping: %s", url)
                    await page.goto(url, wait_until="networkidle", timeout=60_000)
                    content = await page.evaluate("() => document.body.innerText")
                    clean = " ".join(content.split())
                    return clean[:MAX_CHARS_PER_PAGE]
                except Exception as exc:
                    logger.warning("Failed to scrape %s: %s", url, exc)
                    return ""
                finally:
                    await browser.close()

# -- NODES -- #
async def planner_node(state: ResearcherState):
    llm = ChatGoogleGenerativeAI(model=state["selected_model"], temperature=0.2)
    structured_llm = llm.with_structured_output(SearchQueries)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert OSINT researcher. Your job is to generate HIGHLY SPECIFIC web search queries.
        Rules:
        - Include exact names, model numbers, dates, or identifiers when known
        - Target authoritative sources (official sites, technical docs, reputable news)
        - Never generate a query already covered by the data below
        - Each query must target a DIFFERENT aspect of the objective
        - Prefer queries that would appear on the page you want (e.g. "BMW M4 0-60 mph" not "BMW M4 performance")
        """),
        ("user", "Objective: {objective}\nData gathered so far: {scraped_data}\n\nWhat should we search for next?")
    ])
    
    chain = prompt | structured_llm
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"] or "None"
    })
    
    return {
        "search_queries": response.queries,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "total_queries_run": state.get("total_queries_run", 0) + len(response.queries) # NEW: Tally up the total
    }
    
async def search_scraper_node(state: ResearcherState):
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    current_data = state.get("scraped_data", "")
    current_urls = state.get("visited_urls", [])

    # Collect URLs across all queries first
    urls_to_scrape = []
    for query in state["search_queries"]:
        results = await client.search(query, max_results=2)
        for r in results.get("results", []):
            url = r["url"]
            if url not in current_urls and url not in urls_to_scrape:
                urls_to_scrape.append(url)

    # Scrape all URLs concurrently with per-domain and global limits (with retries)
    sem_global = _get_global_scrape_semaphore()
    async def _crawl(u: str):
        domain = urllib.parse.urlparse(u).netloc
        dom_sem = _get_domain_semaphore(domain)
        async with dom_sem:
            async with sem_global:
                # Retry loop with simple exponential backoff
                for attempt in range(_MAX_RETRIES + 1):
                    start = time.perf_counter()
                    try:
                        content = await asyncio.wait_for(scrape_deep_content(u), timeout=_URL_TIMEOUT)
                        ok = bool(content)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout scraping %s on attempt %d", u, attempt + 1)
                        content = ""
                        ok = False
                    finally:
                        elapsed = (time.perf_counter() - start) * 1000
                        # Update domain metrics
                        _DOMAIN_LATENCY_SUM[domain] = _DOMAIN_LATENCY_SUM.get(domain, 0) + elapsed
                        _DOMAIN_LATENCY_COUNT[domain] = _DOMAIN_LATENCY_COUNT.get(domain, 0) + 1
                        _DOMAIN_TOTAL[domain] = _DOMAIN_TOTAL.get(domain, 0) + 1
                        _DOMAIN_SUCCESS[domain] = _DOMAIN_SUCCESS.get(domain, 0) + (1 if ok else 0)
                    if content:
                        break
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(2 ** attempt)
                return u, content
    tasks = [_crawl(url) for url in urls_to_scrape]
    pages = await asyncio.gather(*tasks)

    new_urls = []
    for url, page_content in pages:
        if not page_content:
            continue
        if len(current_data) >= MAX_SCRAPED_CHARS:
            logger.warning("MAX_SCRAPED_CHARS reached. Stopping scrape.")
            break
        current_data += f"\n\n-- SOURCE: {url} --\n{page_content}"
        new_urls.append(url)

    return {
        "scraped_data": current_data,
        "visited_urls": [*current_urls, *new_urls]
    }

async def evaluator_node(state: ResearcherState):
    # Skip LLM call if there's nothing to evaluate yet
    if not state.get("scraped_data", "").strip():
        logger.info("Evaluator skipped — no data yet.")
        return {"needs_more_info": True}

    llm = ChatGoogleGenerativeAI(model=state["selected_model"], temperature=0.2)
    structured_llm = llm.with_structured_output(Evaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quality assurance AI. Check if the scraped data satisfies the objective."),
        ("user", "Objective: {objective}\n\nScraped Data:\n{scraped_data}")
    ])
    chain = prompt | structured_llm
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"]
    })
    return {"needs_more_info": not response.is_complete}

async def reporter_node(state: ResearcherState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent analyst. Using only the provided source data, "
            "write a structured Markdown report. Include: an executive summary, "
            "key findings organised by theme, and a source list. "
            "Do not invent information not present in the data."
            )),
        ("user", "Objective: {objective}\n\nSource Data:\n{scraped_data}")
    ])
    llm = ChatGoogleGenerativeAI(model=state["selected_model"], temperature=0.2)
    chain = prompt | llm
    
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"]
    })
    
    return {"final_report": response.content[0]["text"]}

def should_continue(state: ResearcherState):
    if state.get("needs_more_info") and state.get("iteration_count", 0) < 3:
        return "continue"
    else:
        return "finish"

# -- GRAPH BUILDER -- #
def build_graph():
    workflow = StateGraph(ResearcherState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("search_scraper", search_scraper_node)
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
async def run_agent_workflow(objective, selected_model, status_container, metric_containers):
    # Inline live-domain-metrics renderer
    def _render_domain_metrics_inline(status):
        if _DOMAIN_LATENCY_SUM:
            status.write("Domain Metrics:")
            for domain in sorted(_DOMAIN_LATENCY_SUM.keys()):
                total = _DOMAIN_TOTAL.get(domain, 0)
                cnt = _DOMAIN_LATENCY_COUNT.get(domain, 1)
                avg = _DOMAIN_LATENCY_SUM.get(domain, 0) / cnt if cnt else 0
                succ = _DOMAIN_SUCCESS.get(domain, 0)
                rate = succ / total if total > 0 else 0
                status.write(f"- {domain}: avg {avg:.0f} ms, success {rate:.0%} ({succ}/{total})")
    q_metric, u_metric, c_metric = metric_containers
    
    app = build_graph()
    initial_state = {
        "objective": objective,
        "selected_model": selected_model,
        "search_queries": [],
        "visited_urls": [],
        "scraped_data": "",
        "needs_more_info": True,
        "final_report": "",
        "iteration_count": 0,
        "total_queries_run": 0
    }
    
    final_report = ""
    
    # Local trackers for UI
    current_queries = 0
    current_urls = 0
    current_chars = 0
    
    async for output in app.astream(initial_state):
        for node_name, state_update in output.items():
            
            # --- NEW: Update live numbers ---
            if "total_queries_run" in state_update:
                current_queries = state_update["total_queries_run"]
            if "visited_urls" in state_update:
                current_urls = len(state_update["visited_urls"])
            if "scraped_data" in state_update:
                current_chars = len(state_update["scraped_data"])
                
            q_metric.metric("Queries Run", current_queries)
            u_metric.metric("Sites Scraped", current_urls)
            c_metric.metric("Chars Collected", current_chars)
            # ---------------------------------
            # Live domain metrics - show during research
            _render_domain_metrics_inline(status_container)

            if node_name == "planner":
                status_container.write(f"🧠 **Planner generated queries:** {', '.join(state_update.get('search_queries', []))}")
            elif node_name == "search_scraper":
                status_container.write(f"🕵️ **Scraping data from:** {len(state_update.get('visited_urls', []))} total sources...")
            elif node_name == "evaluator":
                needs_more = state_update.get('needs_more_info')
                if needs_more:
                    status_container.write("⚠️ **Evaluator:** Information incomplete. Looping back for more data.")
                else:
                    status_container.write("✅ **Evaluator:** Data gathering complete! Writing report...")
            elif node_name == "reporter":
                status_container.write("📝 **Reporter:** Report compiled successfully.")
                final_report = state_update.get("final_report", "")
                
    return final_report

# -- STREAMLIT UI -- #
st.set_page_config(page_title="OSINT Agent", page_icon="🕵️‍♂️", layout="centered")

st.title("🕵️‍♂️ Autonomous OSINT Agent")
st.markdown("Enter a research objective. The agent will autonomously plan, search, scrape, and evaluate until it has enough data to write a comprehensive report.")

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("Missing API Keys! Please ensure GOOGLE_API_KEY and TAVILY_API_KEY are set in your .env or Streamlit Secrets.")
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
    ui_global = st.number_input("Global concurrency", min_value=1, max_value=20, value=_MAX_SCRAPE_CONCURRENCY, key="ui_global_concurrency")
    ui_domain = st.number_input("Per-domain concurrency", min_value=1, max_value=20, value=_MAX_SCRAPE_PER_DOMAIN_CONCURRENCY, key="ui_domain_concurrency")
with col_b:
    ui_timeout = st.number_input("Per-URL timeout (s)", min_value=5, max_value=120, value=_URL_TIMEOUT, key="ui_timeout")
    ui_retries = st.number_input("Max retries per URL", min_value=0, max_value=5, value=_MAX_RETRIES, key="ui_max_retries")
ui_block = st.checkbox("Block heavy resources (images, css, fonts)", value=_RESOURCE_BLOCKING, key="ui_resource_blocking")

def _refresh_scrape_config_from_ui():
    global _MAX_SCRAPE_CONCURRENCY, _MAX_SCRAPE_PER_DOMAIN_CONCURRENCY, _URL_TIMEOUT, _MAX_RETRIES, _RESOURCE_BLOCKING
    _MAX_SCRAPE_CONCURRENCY = int(ui_global)
    _MAX_SCRAPE_PER_DOMAIN_CONCURRENCY = int(ui_domain)
    _URL_TIMEOUT = int(ui_timeout)
    _MAX_RETRIES = int(ui_retries)
    _RESOURCE_BLOCKING = bool(ui_block)

# 2. THE BUTTON BLOCK
if st.button("Start Research", type="primary"):
    if not objective.strip():
        st.warning("Please enter an objective first.")
    else:
        # Apply the knob settings right as we click start
        _refresh_scrape_config_from_ui()

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

                def run_in_thread(objective, model, status, metrics):
                    add_script_run_ctx(ctx=ctx)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            run_agent_workflow(objective, model, status, metrics)
                        )
                    finally:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_in_thread, objective, selected_model, status, (q_metric, u_metric, c_metric))
                    final_report = future.result()

                status.update(label="Research Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="An error occurred", state="error")
                st.error(f"Error details: {e}")
                final_report = None
        
        if final_report:
            def render_domain_metrics():
                if _DOMAIN_LATENCY_SUM:
                    st.subheader("Domain Metrics")
                    for domain in sorted(_DOMAIN_LATENCY_SUM.keys()):
                        total = _DOMAIN_TOTAL.get(domain, 0)
                        cnt = _DOMAIN_LATENCY_COUNT.get(domain, 1)
                        avg = _DOMAIN_LATENCY_SUM.get(domain, 0) / cnt if cnt else 0
                        succ = _DOMAIN_SUCCESS.get(domain, 0)
                        rate = succ / total if total > 0 else 0
                        st.text(f"{domain}: avg {avg:.0f} ms, success {rate:.0%} ({succ}/{total})")
            render_domain_metrics()
            st.subheader("Final Report")
            st.markdown(final_report)
            
            with st.expander("View Raw Markdown"):
                st.code(final_report, language="markdown")