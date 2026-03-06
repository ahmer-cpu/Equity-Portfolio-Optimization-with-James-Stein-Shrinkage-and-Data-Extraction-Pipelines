import asyncio
import sys
import pandas as pd
import time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Fix for Windows subprocess issues with Playwright
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

ticker_symbols = ['AAPL']  # Try with just one stock first

async def fetch_ticker_data(playwright, ticker):
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    url = f"https://finance.yahoo.com/quote/{ticker}/history/?interval=1wk&filter=history&frequency=1d&period1=1680220800&period2=1743392053"
    await page.goto(url)
    await page.wait_for_selector("table")

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    await browser.close()

    table = soup.find("table")
    if not table:
        print(f"[{ticker}] No table found")
        return ticker, None

    rows = table.find_all("tr")
    dates, closes = [], []

    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) == 7:
            try:
                date = pd.to_datetime(cols[0].text.strip())
                adj_close = float(cols[5].text.strip().replace(",", ""))
                dates.append(date)
                closes.append(adj_close)
            except:
                continue

    df = pd.DataFrame({"Date": dates, "Adj Close": closes})

    print(f"[{ticker}] {len(df)} rows")
    return ticker, df

async def run_scraper(tickers):
    results = {}
    start = time.time()

    async with async_playwright() as p:
        tasks = [fetch_ticker_data(p, ticker) for ticker in tickers]
        responses = await asyncio.gather(*tasks)

    for ticker, df in responses:
        if df is not None:
            results[ticker] = df

    print(f"\nTotal time: {time.time() - start:.2f} seconds")
    print(f" Scraped {len(results)} tickers successfully.")
    return results

if __name__ == "__main__":
    results = asyncio.run(run_scraper(ticker_symbols[:1]))  # just AAPL
    combined = pd.concat([df.assign(Ticker=ticker) for ticker, df in results.items()])
    combined.to_csv("aapl_twoyearly_adjclose.csv", index=False)
    print("Saved to aapl_twoyearly_adjclose.csv")
