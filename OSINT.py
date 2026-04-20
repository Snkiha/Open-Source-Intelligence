import asyncio # Built-in libray to run asyncronous code
from playwright.async_api import async_playwright # A browser automation library that control the Chrome browser programmatically
from playwright_stealth import Stealth # A plugin that modifies the browser's fingerprint to look less like a bot
from typing import TypedDict, List
from langgraph.graph import StateGraph, END # Core library for LangGraph classes. StateGraph is the graph builder

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tavily import AsyncTavilyClient

import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path

load_dotenv()

# -- LOGGING -- #
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"osint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(), # Console output
        logging.FileHandler(log_filename, "w"),
    ],
)
logger = logging.getLogger("osint_agent")

# --CONFIG-- #
MAX_SCRAPED_CHARS = 80_000 # Hard cap on total scraped data
MAX_CHARS_PER_PAGE = 15_000 # Max chars taken from any single page
MAX_ITERATIONS = 3

for var in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
    if not os.getenv(var):
        raise EnvironmentError(f"{var} is not set in .env")

# Initialize the Brain
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2, max_retries=2)

async def scrape_deep_content(url):
    async with async_playwright() as p:
        # Launch headless browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        # Apply stealth to bypass basic bot detection
        await Stealth().apply_stealth_async(page)
        
        try:
            logger.info("Scraping: %s", url)
            await page.goto(url, wait_until="networkidle", timeout=80_000)
            content = await page.evaluate("() => document.body.innerText")
            clean = " ".join(content.split())
            return clean[:MAX_CHARS_PER_PAGE]
        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", url, exc)
            return "" # Empty string if there is error

# Agent State (Agent's Memory)
class ReseacherState(TypedDict):
    objective: str
    search_queries: List[str]
    visited_urls: List[str]
    scraped_data: str
    needs_more_info: bool # New flag for the router
    final_report: str
    iteration_count: int

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 2-3 targeted search queries to find the missing information.")

# Nodes (The "Actors")
async def planner_node(state: ReseacherState):
    print("\n-- [NODE: PLANNER] Generating Queries --")
    
    # Bind the tool to the LLM to force JSON output
    structured_llm = llm.with_structured_output(SearchQueries)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert OSINT researcher. Break down the user's objective into highly targeted web search queries."),
        ("user", "Objective: {objective}\nData gathered so far: {scraped_data}\n\nWhat should we search for next?")
    ])
    
    # Chain it together and run
    chain = prompt | structured_llm
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"] or "None"
    })
    
    # Return the dynamically generated queries to the state
    return {
        "search_queries": response.queries,
        "iteration_count": state.get("iteration_count", 0) + 1
    }
    
async def search_scraper_node(state: ReseacherState):
    print("\n-- [NODE: SEARCH & SCRAPE] Gathering data --")
    # Tavily search and the async Playwright script
    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    current_data = state.get("scraped_data", "")
    current_urls = state.get("visited_urls", [])
    new_urls = []
    
    for query in state["search_queries"]:
        print(f"Searching: {query}")
        results = await client.search(query, max_results=3)
        
        for r in results.get("results", []):
            url = r["url"]
            if url not in current_urls and url not in new_urls:
                page_content = await scrape_deep_content(url)
                current_data += f"\n\n-- SOURCE: {url} --\n{page_content}"
                new_urls.append(url)
    
    return {
        "scraped_data": current_data,
        "visited_urls": [*current_urls, *new_urls]
    }

class Evaluation(BaseModel):
    is_complete: bool = Field(description="True if scraped data fully answers the objective. False if information is missing.")
    reasoning: str = Field(description="Why you made this decision.")

async def evaluator_node(state: ReseacherState):
    print("\n-- [NODE: EVALUATOR] Analyzing Gaps --")
    structured_llm = llm.with_structured_output(Evaluation)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quality assurance AI. Check if the scraped data satisfies the objective. Be strict."),
        ("user", "Objective: {objective}\n\nScraped Data:\n{scraped_data}")
    ])
    
    chain = prompt | structured_llm
    
    # Pass the data to LLM
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"]
    })
    print(f"Evaluator Reasoning: {response.reasoning}")
    
    return {"needs_more_info": not response.is_complete}

async def reported_node(state: ReseacherState):
    print("\n-- [NODE: REPORTER] Compiling Final Dossier --")
    # TODO: Add LLM prompt to format the raw data into a clean Markdown report
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent analyst. Using only the provided source data, "
            "write a structured Markdown report. Include: an executive summary,"
            "key findings organised by theme, and a source list."
            "Do not invent information not present in the data."
            )),
        ("user", "Objective: {objective}\n\nSource Data:\n{scraped_data}")
    ])
    chain = prompt | llm
    response = await chain.ainvoke({
        "objective": state["objective"],
        "scraped_data": state["scraped_data"]
    })
    return {"final_report": response.content[0]["text"]}
    
# Routing Logic (The "Brain")
def should_continue(state: ReseacherState):
    print("\n-- [ROUTER] Deciding next steps --")
    
    if state.get("needs_more_info") and state.get("iteration_count", 0) < 3:
        print("-> Missing Information. Looping back to planner.")
        return "continue"
    else:
        print("-> Objective met. Routing to Reporter.")
        return "finish"

# Build and compile the Graph
workflow = StateGraph(ReseacherState)

# Register Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("search_scraper", search_scraper_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("reporter", reported_node)

# Standard Linear Edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "search_scraper")
workflow.add_edge("search_scraper", "evaluator")

# Conditional edge (The loop)
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "continue": "planner", # Loop back to start
        "finish": "reporter" # Go to final step
    }
)

# End the graph after reporting
workflow.add_edge("reporter", END)

# Compile into an executable application
app = workflow.compile()

async def main():
    initial_state = {
        "objective": "Identify the key capabilities of the Lockheed Martin F-22 Raptor.",
        "search_queries": [],
        "visited_urls": [],
        "scraped_data": "",
        "needs_more_info": True,
        "final_report": "",
        "iteration_count": 0
    }
    
    logger.info("Starting Agentic Loop....")
    logger.info("Objective: %s", initial_state["objective"])
    logger.info("Log file: %s", log_filename)
    
    
    # Invoke Graph
    final_state = await app.ainvoke(initial_state)
    logger.info("---Run Complete---")
    print(final_state["final_report"])

if __name__ == "__main__":
    asyncio.run(main())