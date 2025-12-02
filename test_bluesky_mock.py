import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Mock environment variables
os.environ['BLUESKY_USER'] = 'testuser'
os.environ['BLUESKY_PASSWORD'] = 'testpass'

# Mock external dependencies BEFORE importing core modules
sys.modules['atproto'] = MagicMock()
sys.modules['atproto_client'] = MagicMock()
sys.modules['atproto_client.models'] = MagicMock()
sys.modules['atproto_client.models.app'] = MagicMock()
sys.modules['atproto_client.models.app.bsky'] = MagicMock()
sys.modules['atproto_client.models.app.bsky.feed'] = MagicMock()
sys.modules['atproto_client.models.app.bsky.feed.post'] = MagicMock()
sys.modules['atproto_client.models.com'] = MagicMock()
sys.modules['atproto_client.models.com.atproto'] = MagicMock()
sys.modules['atproto_client.models.com.atproto.repo'] = MagicMock()
sys.modules['atproto_client.exceptions'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['parsel'] = MagicMock()

# Ensure core is in path
sys.path.append(os.getcwd())

# Mock core.bluesky_api if import fails
try:
    import core.bluesky_api
except ImportError:
    sys.modules['core.bluesky_api'] = MagicMock()

async def main():
    print("Starting Bluesky Mock Test...")
    
    with patch('core.bluesky_api.get_own_posts') as mock_get_posts, \
         patch('core.bluesky_api.get_comments_for_post') as mock_get_comments, \
         patch('core.bluesky_api.reply_to_post') as mock_reply, \
         patch('core.bluesky_api.post_to_bluesky_with_image') as mock_post, \
         patch('core.image_api.generate_image') as mock_gen_image, \
         patch('core.llm_api.run_llm') as mock_llm:

        # Setup mocks
        mock_get_posts.return_value = [
            MagicMock(uri='uri1', cid='cid1', value=MagicMock(text='Test Post 1'))
        ]
        
        # Mock comment structure
        mock_comment = MagicMock()
        mock_comment.post.record.text = "I love this!"
        mock_comment.post.author.handle = "fan_user"
        mock_comment.post.uri = "comment_uri"
        mock_comment.post.cid = "comment_cid"
        mock_comment.replies = [] 
        
        mock_get_comments.return_value = [mock_comment]
        mock_reply.return_value = True
        mock_post.return_value = "Posted!"
        
        mock_image = MagicMock()
        mock_gen_image.return_value = mock_image
        
        def llm_side_effect(*args, **kwargs):
            purpose = kwargs.get('purpose')
            if purpose == 'generate_phrase':
                return {'result': 'LOVE IS CODE'}
            elif purpose == 'social_media_decision':
                return {'result': 'YES'}
            elif purpose == 'social_media_reply':
                return {'result': 'Thank you! 💖'}
            elif purpose == 'social_media_post':
                 return {'result': 'Test Post Content'}
            return {'result': 'Default LLM Response'}
            
        mock_llm.side_effect = llm_side_effect

        # Import tools
        from core.tools_legacy import post_to_bluesky, scan_and_reply_to_bluesky

        # Test post_to_bluesky
        print("\nTesting post_to_bluesky...")
        result = await post_to_bluesky(text="Test Post")
        print(f"Result: {result}")
        
        if mock_post.called:
            print("✅ post_to_bluesky_with_image called.")
        else:
            print("❌ post_to_bluesky_with_image NOT called.")
            
        if mock_gen_image.called:
            print("✅ generate_image called.")
        else:
            print("❌ generate_image NOT called.")
            
        # Test scan_and_reply_to_bluesky
        print("\nTesting scan_and_reply_to_bluesky...")
        result_scan = await scan_and_reply_to_bluesky()
        print(f"Result: {result_scan}")
        
        if mock_get_posts.called:
            print("✅ get_own_posts called.")
        else:
            print("❌ get_own_posts NOT called.")
            
        if mock_get_comments.called:
            print("✅ get_comments_for_post called.")
        else:
            print("❌ get_comments_for_post NOT called.")
            
        if mock_reply.called:
            print("✅ reply_to_post called.")
        else:
            print("❌ reply_to_post NOT called.")

if __name__ == '__main__':
    asyncio.run(main())
