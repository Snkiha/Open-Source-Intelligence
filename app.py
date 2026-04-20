import streamlit as st
import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# -- STREAMLIT PAGE CONFIG -- #
st.set_page_config(page_title="Agentic OSINT Researcher", page_icon="🕵️‍♂️", layout="wide")
load_dotenv()

# -- LOGGING SETUP -- #
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_filename = LOG_DIR / f"osint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, "w"),
    ],
)
logger = logging.getLogger("osint_agent")

# -- CONSTANTS -- #
MAX_SCRAPED_CHARS = 80_000
MAX_CHARS_PER_PAGE = 15_000

# -- AGENT STATE -- #
class ResearcherState(TypedDict):
    objective: str
    search_queries: List[str]
    visited_urls: List[str]
    scraped_data: str
    needs_more_info: bool
    final_report: str
    iteration_count: int

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 2-3 targeted search queries to find the missing information.")

class Evaluation(BaseModel):
    is_complete: bool = Field(description="True if scraped data fully answers the objective. False if information is missing.")
    reasoning: str = Field(description="Why you made this decision.")

# -- UTILITY FUNCTIONS -- #
async def scrape_deep_content(url, st_log):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        try:
            logger.info(f"Scraping: {url}")
            st_log.write(f"📄 Scraping: {url}")
            await page.goto(url, wait_until="networkidle", timeout=60_000)
            content = await page.evaluate("() => document.body.innerText")
            clean = " ".join(content.split())
            await browser.close()
            return clean[:MAX_CHARS_PER_PAGE]
        except Exception as exc:
            logger.warning(f"Failed to scrape {url}: {exc}")
            st_log.error(f"❌ Failed to scrape {url}: {exc}")
            await browser.close()
            return ""

# -- AGENT WORKFLOW FACTORY -- #
def create_workflow(google_api_key: str, tavily_api_key: str, max_iterations: int, st_log):
    # Initialize LLM with the provided key
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview", # You can change this to "gemini-2.5-flash" if needed
        temperature=0.2, 
        max_retries=2,
        google_api_key=google_api_key
    )

    async def planner_node(state: ResearcherState):
        st_log.info(f"🧠 **PLANNER:** Analyzing objective (Iteration {state.get('iteration_count', 0) + 1}/{max_iterations})...")
        structured_llm = llm.with_structured_output(SearchQueries)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert OSINT researcher. Break down the user's objective into highly targeted web search queries."),
            ("user", "Objective: {objective}\nData gathered so far: {scraped_data}\n\nWhat should we search for next?")
        ])
        
        chain = prompt | structured_llm
        response = await chain.ainvoke({
            "objective": state["objective"],
            "scraped_data": state.get("scraped_data", "None")
        })
        
        st_log.write(f"🔎 Generated Queries: {', '.join(response.queries)}")
        return {
            "search_queries": response.queries,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
        
    async def search_scraper_node(state: ResearcherState):
        st_log.info("🕸️ **SEARCH & SCRAPE:** Gathering data from the web...")
        client = AsyncTavilyClient(api_key=tavily_api_key)
        
        current_data = state.get("scraped_data", "")
        current_urls = state.get("visited_urls", [])
        new_urls = []
        
        for query in state["search_queries"]:
            st_log.write(f"Tavily Search: `{query}`")
            results = await client.search(query, max_results=2)
            
            for r in results.get("results", []):
                url = r["url"]
                if url not in current_urls and url not in new_urls:
                    page_content = await scrape_deep_content(url, st_log)
                    if page_content:
                        current_data += f"\n\n-- SOURCE: {url} --\n{page_content}"
                        new_urls.append(url)
        
        return {
            "scraped_data": current_data,
            "visited_urls": [*current_urls, *new_urls]
        }

    async def evaluator_node(state: ResearcherState):
        st_log.info("⚖️ **EVALUATOR:** Analyzing gaps in current knowledge...")
        structured_llm = llm.with_structured_output(Evaluation)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a quality assurance AI. Check if the scraped data fully satisfies the objective. Be strict."),
            ("user", "Objective: {objective}\n\nScraped Data:\n{scraped_data}")
        ])
        
        chain = prompt | structured_llm
        response = await chain.ainvoke({
            "objective": state["objective"],
            "scraped_data": state["scraped_data"]
        })
        
        if response.is_complete:
            st_log.success(f"✅ Evaluator says we have enough data! Reasoning: {response.reasoning}")
        else:
            st_log.warning(f"⚠️ Evaluator says data is incomplete. Reasoning: {response.reasoning}")
            
        return {"needs_more_info": not response.is_complete}

    async def reporter_node(state: ResearcherState):
        st_log.info("📝 **REPORTER:** Compiling final dossier...")
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
        st_log.success("🎉 Report generated successfully!")
        return {"final_report": response.content}
        
    def should_continue(state: ResearcherState):
        if state.get("needs_more_info") and state.get("iteration_count", 0) < max_iterations:
            return "continue"
        else:
            return "finish"

    # Build Graph
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
        {
            "continue": "planner",
            "finish": "reporter"
        }
    )
    workflow.add_edge("reporter", END)

    return workflow.compile()

# -- ASYNC RUNNER -- #
async def run_agent_ui(objective, google_key, tavily_key, max_iters, st_log):
    app = create_workflow(google_key, tavily_key, max_iters, st_log)
    
    initial_state = {
        "objective": objective,
        "search_queries": [],
        "visited_urls": [],
        "scraped_data": "",
        "needs_more_info": True,
        "final_report": "",
        "iteration_count": 0
    }
    
    final_state = await app.ainvoke(initial_state)
    return final_state

# -- STREAMLIT UI LAYOUT -- #
def main():
    st.title("🕵️‍♂️ OSINT Researcher Agent")
    st.markdown("A LangGraph-powered autonomous web scraping agent that researches any objective.")

    # Sidebar for Config
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_iters = st.slider("Max Research Iterations", min_value=1, max_value=5, value=3)
        
        st.markdown("---")
        st.markdown("""
        **Requirements Note:**
        If you haven't already, ensure you run:
        ```bash
        playwright install chromium
        ```
        in your terminal before running this app.
        
        **Security Note:**
        API Keys are loaded securely from your `.env` file.
        """)

    # Main Content
    objective = st.text_area("🎯 Research Objective", placeholder="e.g., Identify the key capabilities of the BMW M4...", height=100)
    
    if st.button("🚀 Start Research", use_container_width=True, type="primary"):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not google_api_key or not tavily_api_key:
            st.error("⚠️ Please ensure GOOGLE_API_KEY and TAVILY_API_KEY are set in your `.env` file.")
            return
        if not objective.strip():
            st.error("⚠️ Please enter a research objective.")
            return

        # UI Containers
        progress_container = st.container(border=True)
        
        # Dashboard stats
        stats_placeholder = progress_container.empty()
        render_stats(stats_placeholder, 0, 0, 0)
        st.markdown("---")
        
        progress_container.subheader("🔄 Agent Thought Process")
        st_log = progress_container.empty()
        
        tabs = st.tabs(["📑 Final Report", "🔗 Visited URLs", "🗄️ Raw Scraped Data"])
        
        with st.spinner("Agent is actively researching..."):
            # Run the async graph execution safely in Streamlit
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                final_state = loop.run_until_complete(
                    run_agent_ui(objective, google_api_key, tavily_api_key, max_iters, st_log, stats_placeholder)
                )
                loop.close()
                
                # Render Results
                with tabs[0]:
                    st.markdown(final_state.get("final_report", "No report generated."))
                
                with tabs[1]:
                    st.write("### Sources Extracted")
                    urls = final_state.get("visited_urls", [])
                    if urls:
                        for url in urls:
                            st.markdown(f"- [{url}]({url})")
                    else:
                        st.info("No URLs were visited.")
                        
                with tabs[2]:
                    st.write("### Raw Scraped Text Dump")
                    st.text_area("Data", final_state.get("scraped_data", ""), height=400)
                    
            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")
                logger.exception("Agent execution failed")

if __name__ == "__main__":
    main()