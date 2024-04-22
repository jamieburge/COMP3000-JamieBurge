import asyncio
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

async def main():
    toys = await scanner.find_toys()
    
    for toy in toys:
        if toy.toy_name == 'SM-B882':
            async with SpheroEduAPI(toy) as api:
                await api.spin(360, 1)

if __name__ == "__main__":
    # Create a new event loop for this script
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())