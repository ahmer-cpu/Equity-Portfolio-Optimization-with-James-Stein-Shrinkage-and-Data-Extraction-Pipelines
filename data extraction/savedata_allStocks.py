import asyncio
import sys
import pandas as pd
import time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Windows compatibility
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load tickers from saved CSV from earlier script (weighted tickers)
ticker_symbols = pd.read_csv("sp500_tickers.csv")['Symbol'].tolist()
print(f"Loaded {len(ticker_symbols)} tickers.")

# Define dynamic time range (Not used in the URL currently)
# Recommendation: Manually get the form of the Yahoo Finance URL and plug it into the fetch function with the 'ticker' variable
period1 = int(pd.Timestamp("2023-04-01").timestamp())  # ~2 years ago
period2 = int(pd.Timestamp("today").timestamp())        # today

MAX_RETRIES = 3

# Scrape adjusted close prices
async def fetch_ticker_data(playwright, ticker):
    for attempt in range(1, MAX_RETRIES + 1):
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            url = (
                f"https://finance.yahoo.com/quote/{ticker}/history/"
                f"?interval=1wk&filter=history&frequency=1d"
                f"&period1=1680220800&period2=1743379200"
            )

            await page.goto(url, timeout=30000)
            await page.wait_for_selector("table", timeout=10000)

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            table = soup.find("table")
            if not table:
                raise ValueError("Table not found")

            rows = table.find_all("tr")
            if len(rows) < 2:
                raise ValueError("No data rows found")

            dates, adj_closes = [], []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) == 7:
                    try:
                        date = pd.to_datetime(cols[0].text.strip())
                        adj_close = float(cols[5].text.strip().replace(",", ""))
                        dates.append(date)
                        adj_closes.append(adj_close)
                    except:
                        continue

            if not dates:
                raise ValueError("Parsed 0 rows")

            df = pd.DataFrame({"Date": dates, "Adj Close": adj_closes})
            df["Ticker"] = ticker
            print(f"[{ticker}] {len(df)} rows (try {attempt})")
            return df

        except Exception as e:
            print(f"[{ticker}] Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"[{ticker}] Failed after {MAX_RETRIES} attempts")
            await asyncio.sleep(2)
        finally:
            await browser.close()

    return None

async def run_scraper(tickers):
    results = []
    failed_tickers = []
    start = time.time()

    async with async_playwright() as p:
        for i, ticker in enumerate(tickers, 1):
            df = await fetch_ticker_data(p, ticker)
            if df is not None:
                results.append(df)
            else:
                failed_tickers.append(ticker)

            print(f"[{i}/{len(tickers)}] Done: {ticker}")
            time.sleep(1)  # polite delay

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    combined.to_csv("sp500_adjusted_close_2y.csv", index=False)

    print(f"\nTotal time: {time.time() - start:.2f} seconds")
    print("Saved to sp500_adjusted_close_2y.csv")

    if failed_tickers:
        pd.Series(failed_tickers).to_csv("failed_tickers.csv", index=False)
        print(f"{len(failed_tickers)} tickers failed and were saved to failed_tickers.csv")

    return combined

if __name__ == "__main__":
    asyncio.run(run_scraper(ticker_symbols))
