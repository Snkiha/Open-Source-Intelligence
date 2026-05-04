# Autonomous OSINT Agent

An interactive, AI-powered Open-Source Intelligence (OSINT) research assistant built with LangGraph, Streamlit, and Playwright. 

Provide the agent with a research objective, and it will autonomously break down the task, search the web, scrape deep content, evaluate its findings, and loop back for more information if necessary—culminating in a comprehensive, Markdown-formatted report.

## Key Features

* **Autonomous Agentic Loop:** Uses LangGraph to orchestrate a continuous cycle of planning, searching, evaluating, and reporting.
* **Deep Web Scraping:** Leverages asynchronous Playwright with stealth plugins to bypass basic bot protection and read actual page content, not just SEO metadata.
* **Real-Time UI:** Built on Streamlit, the interface streams the agent's "thought process" and live metrics (Queries Run, Sites Scraped, Characters Collected) directly to the user.
* **Intelligent Evaluation:** Powered by Google's Gemini 3.1, the agent strictly evaluates whether it has enough data to fulfill the user's objective before finalizing the report.
* **Cloud-Ready:** Includes automated Playwright binary installation, making it easily deployable to platforms like Streamlit Community Cloud.

## Architecture (How it Works)

The agent operates on a state graph with four primary nodes:

1. **Planner:** Analyzes the objective and current gathered data to generate highly targeted web search queries.
2. **Search & Scraper:** Uses Tavily to execute searches and Playwright to scrape the textual content from the resulting URLs.
3. **Evaluator:** Acts as Quality Assurance. It reviews the scraped data against the initial objective. If data is missing, it loops back to the Planner. If complete, it moves to the Reporter.
4. **Reporter:** Synthesizes all raw scraped data into a structured executive summary and detailed report.

## Prerequisites

You will need API keys for the following services:
* **Google Gemini API** (For the LLM brain)
* **Tavily API** (For the search engine tool)

## Installation & Setup

**1. Clone the repository (or create your project directory)**
Ensure `app.py` is saved in your working directory.

**2. Install dependencies**
Install the required Python packages:
```bash
pip install streamlit playwright playwright-stealth langgraph langchain-google-genai pydantic tavily-python python-dotenv
