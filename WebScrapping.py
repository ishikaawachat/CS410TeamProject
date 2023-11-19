import asyncio
import aiofiles
from duckduckgo_search import AsyncDDGS

async def get_results():
    async with AsyncDDGS() as ddgs:
        query = input('Enter Your Query: ')
        results = [r async for r in ddgs.text(query, region='wt-wt', safesearch='off', timelimit='y', max_results=15)]
        return results

async def save_results_to_file(results, output_file):
    async with aiofiles.open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            await file.write(f"{result}\n")

async def main():
    ddgs_results = await get_results()
    output_file = 'search_results.txt'
    await save_results_to_file(ddgs_results, output_file)
    
    print(f"Search results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
