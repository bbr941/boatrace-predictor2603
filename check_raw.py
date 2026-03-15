import asyncio
import aiohttp
import config
import sys

async def check_raw():
    url = 'https://www.boatrace.jp/owpc/pc/race/index'
    headers = config.HEADERS
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            html = await response.text()
            print("--- HTML Prefix ---")
            print(html[:1000])
            print("\n--- Searching for Venue Keywords ---")
            # 検索用のキーワード（会場名のクラスやテキスト）
            keywords = ["is-jyo", "table1_upperLabel", "戸田", "江戸川", "平和島"]
            for k in keywords:
                count = html.count(k)
                print(f"Keyword '{k}': Found {count} times")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_raw())
