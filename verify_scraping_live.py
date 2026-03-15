import asyncio
import pandas as pd
from data_fetcher import get_realtime_data
import datetime

async def verify_scraping():
    # 本日の日付
    date = datetime.date.today().strftime("%Y%m%d")
    # 実開催中の場コード (03 江戸川)
    place = "03" 
    race_no = "1"
    
    print(f"--- Verifying Scraping for {date} Place:{place} Race:{race_no} ---")
    
    try:
        data = await get_realtime_data(date, place, race_no)
        
        if data:
            race_info = data['race_info']
            odds_info = data['odds_info']
            
            print("\n[Race Header]")
            print(f"Title: {race_info['race_title']}")
            print(f"Grade: {race_info['race_grade_num']}")
            
            print("\n[Weather]")
            print(f"Weather: {race_info['before_info']['weather_text']}")
            print(f"Wind: {race_info['before_info']['wind_direction_name']} {race_info['before_info']['wind_speed']}m")
            print(f"Temperature: {race_info['before_info']['temperature']}C")
            
            print("\n[Exhibition Times]")
            for boat, t in race_info['before_info']['exhibition_times'].items():
                print(f"Boat {boat}: {t}")
                
            print("\n[Odds Sample (First 5)]")
            for combo, odds in list(odds_info.items())[:5]:
                print(f"{combo}: {odds}")
                
            print(f"\nTotal Odds Count: {len(odds_info)}")
            
        else:
            print("Failed to fetch data. No races at this venue or network error.")
            
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_scraping())
