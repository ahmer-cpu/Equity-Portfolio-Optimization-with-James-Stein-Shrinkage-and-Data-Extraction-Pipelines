import requests
import pandas as pd
from bs4 import BeautifulSoup

# Set headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

url = "https://www.slickcharts.com/sp500"
response = requests.get(url, headers=headers)

# Parse HTML with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")

# Read the table into a DataFrame
df = pd.read_html(str(table))[0]

# Clean tickers for Yahoo format
df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)

# Save to CSV
df[['Symbol']].to_csv("sp500_tickers.csv", index=False)
print("Ticker list saved to sp500_tickers.csv")

# Optional: assign to a list
ticker_symbols = df['Symbol'].tolist()