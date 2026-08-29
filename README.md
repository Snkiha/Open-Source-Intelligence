# Autonomous OSINT Agent

An interactive, AI-powered Open-Source Intelligence (OSINT) research assistant built with LangGraph, Streamlit, and Playwright.

Provide the agent with a research objective, and it will autonomously break down the task, search the web, scrape deep content, evaluate its findings, and loop back for more information if necessary—culminating in a comprehensive, Markdown-formatted report.

## Key Features

* **Autonomous Agentic Loop:** Uses LangGraph to orchestrate a continuous cycle of planning, searching, evaluating, and reporting.
* **Deep Web Scraping:** Leverages asynchronous Playwright with stealth plugins to bypass basic bot protection and read actual page content, not just SEO metadata.
* **Scraping API Fallback:** When the headless browser is blocked, times out, or is served a near-empty body, the page is automatically re-fetched through the [Jina Reader API](https://jina.ai/reader/), which renders it server-side from a different IP.
* **Three-Tier Content Strategy:** Every URL degrades gracefully — full browser scrape → scraping API → search snippet — so a run never comes back empty.
* **Real-Time UI:** Built on Streamlit, the interface streams the agent's "thought process" and live metrics (Queries Run, Sites Scraped, Characters Collected) directly to the user.
* **Searchable History:** Every completed run is saved to disk (`history/`, gitignored) and surfaced in a **Past Research** panel. Search previous objectives, report text, or source URLs and re-read a finished report without running the agent again.
* **Intelligent Evaluation:** Powered by Google's Gemini 3.1, the agent strictly evaluates whether it has enough data to fulfill the user's objective before finalizing the report.
* **Cloud-Ready:** Includes automated Playwright binary installation, making it easily deployable to platforms like Streamlit Community Cloud.

## Architecture (How it Works)

The agent operates on a state graph with four primary nodes:

1. **Planner:** Analyzes the objective and current gathered data to generate highly targeted web search queries.
2. **Search & Scraper:** Uses the Jina Search API to execute searches, then scrapes textual content from the resulting URLs (Playwright first, Jina Reader as fallback).
3. **Evaluator:** Acts as Quality Assurance. It reviews the scraped data against the initial objective. If data is missing, it loops back to the Planner. If complete, it moves to the Reporter.
4. **Reporter:** Synthesizes all raw scraped data into a structured executive summary and detailed report.

### How a URL is fetched

Each discovered URL goes through an escalating chain, and the corpus records which tier produced the text:

| Tier | Mechanism | Marker in source data |
|------|-----------|-----------------------|
| 1 | Headless Chromium + stealth, retried up to `Max retries per URL` | `-- SOURCE: <url> --` |
| 2 | Jina Reader API (`https://r.jina.ai/<url>`) | `-- SOURCE (api): <url> --` |
| 3 | Search result snippet (seeded up front) | `-- SOURCE (snippet): <url> --` |

A tier is considered to have failed when it returns fewer than 300 characters, which is the usual signature of a bot wall or consent interstitial.

## Prerequisites

You will need API keys for the following services:

| Key | Required | Purpose |
|-----|----------|---------|
| `GOOGLE_API_KEY` | Yes | Google Gemini — the LLM brain |
| `JINA_API_KEY` | Yes | Jina Search (`s.jina.ai`) for discovery **and** Jina Reader (`r.jina.ai`) for the scrape fallback. Get one at [jina.ai/api-dashboard](https://jina.ai/api-dashboard/). |

> **Note:** the search endpoint rejects unauthenticated requests, so `JINA_API_KEY` is required — without it the agent finds no URLs. The reader endpoint alone would work keyless, but at a much lower rate limit.

> **Privacy note:** search queries go to Jina, and with the scraping API fallback enabled, URLs the browser could not read are sent to Jina's servers for rendering. Turn the fallback off in the UI if that is not acceptable for your use case.

## Installation & Setup

**1. Clone the repository (or create your project directory)**

Ensure `app.py` is saved in your working directory.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your keys**

Create a `.env` file in the project root (it is gitignored):

```
GOOGLE_API_KEY=your_google_key
JINA_API_KEY=your_jina_key
```

On Streamlit Community Cloud, set the same names under **Settings → Secrets** instead.

**4. Run the app**

```bash
streamlit run app.py
```

Playwright's Chromium binary is installed automatically on first launch.

## Scrape Tuning

The UI exposes per-run knobs before you start a research run:

* **Global concurrency** — total pages fetched in parallel.
* **Per-domain concurrency** — parallel requests allowed against a single host.
* **Per-URL timeout (s)** — how long the browser waits for a page.
* **Max retries per URL** — browser retry attempts before falling through to the API.
* **Block heavy resources** — aborts images, CSS, fonts, and media for faster loads.
* **Scraping API fallback** — enables or disables the Jina Reader tier.

## Past Research

Completed runs are written as individual JSON files under `history/` (objective, model, report, source URLs, and run metrics). The **Past Research** panel at the bottom of the app lists them newest-first with a search box that matches on objective, report body, or source URL — every whitespace-separated term must appear. Open a result to read the saved report, its raw Markdown, and its sources.

> Storage is local to the machine. On ephemeral hosts such as Streamlit Community Cloud the history resets on redeploy; for durable history, mount a volume at `history/` or point the store at persistent storage.

Run the history-store unit tests with:

```bash
pytest tests/
```
