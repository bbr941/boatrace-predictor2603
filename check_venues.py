import asyncio
import aiohttp
from bs4 import BeautifulSoup
import config

async def check_races():
    url = 'https://www.boatrace.jp/owpc/pc/race/index'
    headers = config.HEADERS
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'lxml')
            print("--- Current Venues & Links ---")
            venues = soup.select("div.is-jyo")
            for v in venues:
                name = v.select_one("p.table1_upperLabel")
                link = v.select_one("a")
                if name and link:
                    print(f"Name: {name.text.strip()}, Link: {link['href']}")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_races())
