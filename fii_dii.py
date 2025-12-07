from datetime import datetime, timedelta
import time

def get_fii_dii_range(start_date, end_date):
    base = "https://archives.nseindia.com/content/equities/FIIDII_"
    current = start_date
    all_data = []

    while current <= end_date:
        date_str = current.strftime("%d-%b-%Y").replace(" ", "")  # e.g., 04-Dec-2025
        url = base + date_str + ".csv"
        try:
            df = pd.read_csv(url)
            df['Date'] = current
            all_data.append(df)
            print(f"Fetched: {date_str}")
        except:
            print(f"Not available: {date_str}")
        current += timedelta(days=1)
        time.sleep(0.5)  # Be respectful to server

    return pd.concat(all_data, ignore_index=True) if all_data else None

# Usage
start = datetime(2025, 12,1)
end = datetime(2025,12,4)
data = get_fii_dii_range(start, end)
print(data)