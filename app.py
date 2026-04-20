import streamlit as st
import asyncio
import os
import logging
from datetime import datetime
from typing import TypedDict, List
from pathlib import Path

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Search & Scraping
from tavily import AsyncTavilyClient
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG & UI SETUP ---
st.set_page_config(page_title="Nexus OSINT Agent", layout="wide", page_icon="🎯")

# --- AGENT STATE & SCHEMAS ---
class ResearcherState(TypedDict):
    objective: str
    search_queries: List[str]
    visited_urls: List[str]
    scraped_data: str
    needs_more_info: bool
    final_report: str
    iteration_count: int

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="2-3 targeted search queries.")

class Evaluation(BaseModel):
    is_complete: bool = Field(description="True if data answers objective.")
    reasoning: str = Field(description="Why you made this decision.")

# --- SCRAPER FUNCTION ---
MAX_CHARS_PER_PAGE = 15_000

async def scrape_deep_content(url, log_callback):
    async with async_playwright() as p:
        log_callback(f"Launching headless browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        try:
            log_callback(f"Scraping URL: {url}")
            await page.goto(url, wait_until="networkidle", timeout=60_000)
            content = await page.evaluate("() => document.body.innerText")
            clean = " ".join(content.split())
            log_callback(f"Successfully extracted {len(clean[:MAX_CHARS_PER_PAGE])} chars.")
            return clean[:MAX_CHARS_PER_PAGE]
        except Exception as exc:
            log_callback(f"[WARNING] Failed to scrape {url}: {exc}")
            return ""
        finally:
            await browser.close()

# --- STREAMLIT UI & EXECUTION ---
def main():
    # SIDEBAR CONTROLS
    st.sidebar.title("Web Scraper Agent")
    
    st.sidebar.markdown("### 🎯 Mission")
    objective = st.sidebar.text_area("Target Directive", value="Identify the key capabilities of the BMW M4.")
    
    run_btn = st.sidebar.button("Initialize Run", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Mission Metrics")
    metrics_placeholder = st.sidebar.empty()

    # MAIN PANELS
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💻 Agent Terminal (stdout)")
        terminal_container = st.empty()
    with col2:
        st.markdown("### 📄 Final Dossier")
        report_container = st.empty()

    # Base UI State
    terminal_container.code("System Ready. Awaiting Directive...", language="bash")
    report_container.info("Dossier will appear here upon mission completion.")
    
    metrics_placeholder.markdown("""
    **Queries Run:** 0  
    **Sites Scraped:** 0  
    **Chars Collected:** 0  
    """)

    # --- EXECUTION LOGIC ---
    if run_btn:
        if not google_key or not tavily_key:
            st.error("Please provide both Google and Tavily API Keys in the sidebar.")
            return

        # Initialize the LLM with the provided key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Using stable 2.5-flash to prevent 404s
            temperature=0.2, 
            api_key=google_key
        )
        
        tavily_client = AsyncTavilyClient(api_key=tavily_key)

        # Shared list to store live logs for the Streamlit UI
        live_logs = []
        def append_log(msg):
            time_str = datetime.now().strftime("%H:%M:%S")
            live_logs.append(f"[{time_str}] {msg}")
            terminal_container.code("\n".join(live_logs), language="bash")

        # --- NODE DEFINITIONS (Nested to use append_log) ---
        async def planner_node(state: ResearcherState):
            append_log("-- [NODE: PLANNER] Generating Queries --")
            structured_llm = llm.with_structured_output(SearchQueries)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert OSINT researcher. Break down the user's objective into targeted web search queries."),
                ("user", "Objective: {objective}\nData gathered so far: {scraped_data}\n\nWhat should we search for next?")
            ])
            chain = prompt | structured_llm
            response = await chain.ainvoke({"objective": state["objective"], "scraped_data": state["scraped_data"] or "None"})
            append_log(f"Generated {len(response.queries)} queries.")
            return {"search_queries": response.queries, "iteration_count": state.get("iteration_count", 0) + 1}

        async def search_scraper_node(state: ResearcherState):
            append_log("-- [NODE: SEARCH & SCRAPE] Gathering data --")
            current_data = state.get("scraped_data", "")
            current_urls = state.get("visited_urls", [])
            new_urls = []
            
            for query in state["search_queries"]:
                append_log(f"Searching Tavily for: '{query}'")
                results = await tavily_client.search(query, max_results=2)
                
                for r in results.get("results", []):
                    url = r["url"]
                    if url not in current_urls and url not in new_urls:
                        page_content = await scrape_deep_content(url, append_log)
                        if page_content:
                            current_data += f"\n\n-- SOURCE: {url} --\n{page_content}"
                            new_urls.append(url)
            
            return {"scraped_data": current_data, "visited_urls": [*current_urls, *new_urls]}

        async def evaluator_node(state: ResearcherState):
            append_log("-- [NODE: EVALUATOR] Analyzing Gaps --")
            structured_llm = llm.with_structured_output(Evaluation)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a quality assurance AI. Check if the scraped data satisfies the objective. Be strict."),
                ("user", "Objective: {objective}\n\nScraped Data:\n{scraped_data}")
            ])
            chain = prompt | structured_llm
            response = await chain.ainvoke({"objective": state["objective"], "scraped_data": state["scraped_data"]})
            append_log(f"Evaluator Reasoning: {response.reasoning}")
            return {"needs_more_info": not response.is_complete}

        async def reported_node(state: ResearcherState):
            append_log("-- [NODE: REPORTER] Compiling Final Dossier --")
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an intelligent analyst. Using only the provided source data, write a structured Markdown report. Include: an executive summary, key findings, and a source list."),
                ("user", "Objective: {objective}\n\nSource Data:\n{scraped_data}")
            ])
            chain = prompt | llm
            response = await chain.ainvoke({"objective": state["objective"], "scraped_data": state["scraped_data"]})
            
            # FIXED: Gemini returns a string, no [0]["text"] needed!
            return {"final_report": response.content[0]["text"]}

        def should_continue(state: ResearcherState):
            append_log("-- [ROUTER] Deciding next steps --")
            if state.get("needs_more_info") and state.get("iteration_count", 0) < 3:
                append_log("-> Missing Information. Looping back to planner.")
                return "continue"
            else:
                append_log("-> Objective met or max iterations reached. Routing to Reporter.")
                return "finish"

        # --- GRAPH ASSEMBLY ---
        workflow = StateGraph(ResearcherState)
        workflow.add_node("planner", planner_node)
        workflow.add_node("search_scraper", search_scraper_node)
        workflow.add_node("evaluator", evaluator_node)
        workflow.add_node("reporter", reported_node)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "search_scraper")
        workflow.add_edge("search_scraper", "evaluator")
        workflow.add_conditional_edges("evaluator", should_continue, {"continue": "planner", "finish": "reporter"})
        workflow.add_edge("reporter", END)
        app = workflow.compile()

        # --- ASYNC RUNNER ---
        async def run_agent():
            initial_state = {
                "objective": objective,
                "search_queries": [],
                "visited_urls": [],
                "scraped_data": "",
                "needs_more_info": True,
                "final_report": "",
                "iteration_count": 0
            }
            
            append_log("INITIATING AGENTIC WORKFLOW...")
            
            # Using astream to yield updates dynamically
            final_state = initial_state
            async for event in app.astream(initial_state):
                for node_name, state_update in event.items():
                    # Keep track of the accumulating state
                    for key, val in state_update.items():
                        final_state[key] = val
                    
                    # Update live metrics
                    q_len = len(final_state.get("search_queries", []))
                    s_len = len(final_state.get("visited_urls", []))
                    c_len = len(final_state.get("scraped_data", ""))
                    
                    metrics_placeholder.markdown(f"""
                    **Queries Run:** {q_len}  
                    **Sites Scraped:** {s_len}  
                    **Chars Collected:** {c_len:,}  
                    """)
            
            append_log("--- WORKFLOW COMPLETE ---")
            report_container.markdown(final_state["final_report"])

        # Execute the async graph within Streamlit
        with st.spinner("Agent is actively hunting..."):
            asyncio.run(run_agent())

if __name__ == "__main__":
    main()