import asyncio, os
os.environ['BLUESKY_USER'] = 'testuser'
os.environ['BLUESKY_PASSWORD'] = 'testpass'
from core.social_media_react_engine import SocialMediaReactEngine

async def main():
    engine = SocialMediaReactEngine()
    result = await engine.run_post_generation(context='test')
    print('Result:', result)

if __name__ == '__main__':
    asyncio.run(main())
