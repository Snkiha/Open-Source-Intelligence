import asyncio # Built-in libray to run asyncronous code
from playwright.async_api import async_playwright # A browser automation library that control the Chrome browser programmatically
from playwright_stealth import Stealth # A plugin that modifies the browser's fingerprint to look less like a bot
from typing import TypedDict, List
from langgraph.graph import StateGraph, END # Core library for LangGraph classes. StateGraph is the graph builder

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Initialize the Brain
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

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
            print(f"Scraping: {url}")
            await page.goto(url, wait_until="networkidle", timeout=80000)
            
            # Extract the inner text of the body - this removes tags and keeps content
            content = await page.evaluate("() => document.body.innerText")
            
            # Basic cleaning: remove excessive whitespace
            clean_content = " ". join(content.split())
            
            return clean_content[:10000] # Return first 10k characters to stay within context limits
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"
        finally:
            await browser.close()

# Agent State (Agent's Memory)
class ReseacherState(TypedDict):
    objective: str
    search_queries: List[str]
    visited_url: List[str]
    scraped_data: str
    needs_more_info: bool # New flag for the router
    final_report: str

# Nodes (The "Actors")
def planner_node(state: ReseacherState):
    print("\n-- [NODE: PLANNER] Generating Queries --")
    # TODO: Add LLM prompt to read objective and the generate Google/Tavily queries
    mock_queries = ["defense contractor acquisition 2026", "aerospace drone OSINT."]
    return {"search_queries": mock_queries}


def search_scraper_node(state: ReseacherState):
    print("\n-- [NODE: SEARCH & SCRAPE Gathering data")
    # TODO: Integrate Tavily search and the async Playwright script
    target_url = "https://ai.plainenglish.io/train-test-split-explained-why-data-leakage-happens-so-easily-f2ad54bbfa08"
    
    print(f'Executing scraper on: {target_url}')
    new_data = asyncio.run(scrape_deep_content(target_url))
    
    # Append to existing data
    current_data = state.get("scraped_data", "")
    updated_data = current_data + "\n\n-- NEW SOURCE --" + new_data
    
    # Update the visited urls
    current_urls = state.get("visited_url", [])
    current_urls.append(target_url)
    
    return {"scraped_data": updated_data, "visited_url": current_urls}
        
def evaluator_node(state: ReseacherState):
    print("\n-- [NODE: EVALUATOR] Analyzing Gaps --")
    # TODO: Add LLM prompt to chenck if 'scraped_data' satisfies the 'objective'
    return state

def reported_node(state: ReseacherState):
    print("\n-- [NODE: REPORTER] Compiling Final Dossier --")
    # TODO: Add LLM prompt to format the raw data into a clean Markdown report
    dossier = "# Final OSINT Report\n\nAll objective met."
    return {"final_report": dossier}

# Routing Logic (The "Brain")
def should_continue(state: ReseacherState):
    print("\n-- [ROUTER] Deciding next steps --")
    # TODO: This will eventually read the output of the Evaluator LLM.
    
    decision = "finish"
    
    if decision == "continue":
        print("-> Missing Information. Looping back to planner.")
        return "continue"
    else:
        print("-> Objectives met. Routing to Reporter.")
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
        "objective": "Identify the primary capabilities od a specific defense AI framework.",
        "search_queries": [],
        "visited_urls": [],
        "scraped_data": "",
        "final_report": ""
    }
    
    print("Starting Agentic Loop...")
    # Invoke Graph
    final_state = await app.ainvoke(initial_state)
    print("\n-- FINAL REPORT --")
    print(final_state["final_report"])

if __name__ == "__main__":
    asyncio.run(main())