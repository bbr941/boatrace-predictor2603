import asyncio
import datetime
import sys
import os

# パス追加
sys.path.append(os.path.join(os.getcwd(), 'boatrace-v3-predictor'))

from data_fetcher import get_realtime_data
import app_v3 # To access get_active_venues

async def test():
    date_str = "20260316"
    print(f"Testing for date: {date_str}")
    
    # Test venue fetching
    venues = await app_v3.get_active_venues(date_str)
    print(f"Venues found: {[v['name'] for v in venues]}")
    
    if not venues:
        print("No venues found. Target date might be too far in the past or invalid.")
        return

    # Test data fetching for the first venue's 12th race
    v = venues[0]
    print(f"Fetching data for {v['name']} 12R...")
    data = await get_realtime_data(date_str, v['code'], "12")
    
    if data:
        print("Data fetched successfully!")
        print(f"Race Title: {data['race_info']['race_title']}")
        print(f"Exhibition Times: {data['race_info']['before_info']['exhibition_times']}")
        print(f"Odds Sample: {list(data['odds_info'].items())[:3]}")
    else:
        print("Failed to fetch data. Check selectors or URL.")

if __name__ == "__main__":
    asyncio.run(test())
