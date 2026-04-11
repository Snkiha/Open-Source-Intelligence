import asyncio

from playwright.async_api import async_playwright
from playwright_stealth import Stealth

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
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Extract the inner text of the body - this removes tags and keeps content
            content = await page.evaluate("() => document.body.innerText")
            
            # Basic cleaning: remove excessive whitespace
            clean_content = " ". join(content.split())
            
            return clean_content[:10000] # Return first 10k characters to stay within context limits
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"
        finally:
            await browser.close()

if __name__ == "__main__":
    scraped_text = asyncio.run(scrape_deep_content("https://ai.plainenglish.io/train-test-split-explained-why-data-leakage-happens-so-easily-f2ad54bbfa08"))
    print(scraped_text)