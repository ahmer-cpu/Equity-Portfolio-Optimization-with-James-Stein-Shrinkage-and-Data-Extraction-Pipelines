import pandas as pd

# Load the original (successful run)
df_main = pd.read_csv("sp500_adjusted_close_2y.csv", parse_dates=["Date"])

# Load the retry data
df_retry = pd.read_csv("retry_adjusted_close_2y.csv", parse_dates=["Date"])

df_combined = pd.concat([df_main, df_retry], ignore_index=True)


df_combined = df_combined.drop_duplicates(subset=["Date", "Ticker"])

df_combined = df_combined.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
df_combined.to_csv("sp500_adjusted_close_2y_full.csv", index=False)
print("Saved combined data to sp500_adjusted_close_2y_full.csv")
