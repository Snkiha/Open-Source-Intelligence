import asyncio
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List

os.system("playwright install chromium") # Keeps Cloud deployments happy

import streamlit as st
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

# -- CONFIG & SECRETS -- #
load_dotenv() 

if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if not os.getenv("TAVILY_API_KEY") and "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

MAX_SCRAPED_CHARS = 80_000
MAX_CHARS_PER_PAGE = 15_000

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
class ReseacherState(TypedDict):
    objective: str
    search_queries: List[str]
    visited_urls: List[str]
    scraped_data: str
    needs_more_info: bool
    final_report: str
    iteration_count: int
    total_queries_run: int # NEW: Added to track all historical queries

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 2-3 targeted search queries to find the missing information.")

class Evaluation(BaseModel):
    is_complete: bool = Field(description="True if scraped data fully answers the objective. False if information is missing.")
    reasoning: str = Field(description="Why you made this decision.")

# -- HELPER FUNCTIONS -- #
async def scrape_deep_content(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
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
async def planner_node(state: ReseacherState):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    structured_llm = llm.with_structured_output(SearchQueries)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert OSINT researcher. Break down the user's objective into highly targeted web search queries."),
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
    
async def search_scraper_node(state: ReseacherState):
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    
    current_data = state.get("scraped_data", "")
    current_urls = state.get("visited_urls", [])
    new_urls = []
    
    for query in state["search_queries"]:
        results = await client.search(query, max_results=2)
        for r in results.get("results", []):
            url = r["url"]
            if url not in current_urls and url not in new_urls:
                page_content = await scrape_deep_content(url)
                if page_content:
                    current_data += f"\n\n-- SOURCE: {url} --\n{page_content}"
                    new_urls.append(url)
    
    return {
        "scraped_data": current_data,
        "visited_urls": [*current_urls, *new_urls]
    }

async def evaluator_node(state: ReseacherState):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    structured_llm = llm.with_structured_output(Evaluation)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quality assurance AI. Check if the scraped data satisfies the objective. Be strict."),
        ("user", "Objective: {objective}\n\nScraped Data:\n{scraped_data}")
    ])
    
    chain = prompt | structured_llm
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"]
    })
    
    return {"needs_more_info": not response.is_complete}

async def reporter_node(state: ReseacherState):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent analyst. Using only the provided source data, "
            "write a structured Markdown report. Include: an executive summary, "
            "key findings organised by theme, and a source list. "
            "Do not invent information not present in the data."
            )),
        ("user", "Objective: {objective}\n\nSource Data:\n{scraped_data}")
    ])
    chain = prompt | llm
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"]
    })
    
    return {"final_report": response.content[0]["textjff"]}

def should_continue(state: ReseacherState):
    if state.get("needs_more_info") and state.get("iteration_count", 0) < 3:
        return "continue"
    else:
        return "finish"

# -- GRAPH BUILDER -- #
def build_graph():
    workflow = StateGraph(ReseacherState)
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
async def run_agent_workflow(objective, status_container, metric_containers):
    q_metric, u_metric, c_metric = metric_containers
    
    app = build_graph()
    initial_state = {
        "objective": objective,
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

objective = st.text_input("Research Objective:", placeholder="e.g., Identify the key capabilities of the BMW M4")

if st.button("Start Research", type="primary"):
    if not objective.strip():
        st.warning("Please enter an objective first.")
    else:
        st.divider()
        
        # --- NEW: Metric placeholders ---
        col1, col2, col3 = st.columns(3)
        q_metric = col1.empty()
        u_metric = col2.empty()
        c_metric = col3.empty()
        
        # Initialize at zero
        q_metric.metric("Queries Run", 0)
        u_metric.metric("Sites Scraped", 0)
        c_metric.metric("Chars Collected", 0)
        # --------------------------------
        
        with st.status("Agent initialized. Starting research loop...", expanded=True) as status:
            try:
                # Pass the metric containers into the runner so it can update them
                final_report = asyncio.run(run_agent_workflow(
                    objective, 
                    status, 
                    (q_metric, u_metric, c_metric)
                ))
                status.update(label="Research Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="An error occurred", state="error")
                st.error(f"Error details: {e}")
                final_report = None
        
        if final_report:
            st.subheader("Final Report")
            st.markdown(final_report)
            
            with st.expander("View Raw Markdown"):
                st.code(final_report, language="markdown")