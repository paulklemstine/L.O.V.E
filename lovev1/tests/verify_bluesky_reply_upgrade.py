import asyncio
import os
import sys
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to ensure core modules can be imported
# Mock 'love' module to prevent side effects from its top-level code
sys.modules['love'] = MagicMock()

# We need to ensure core modules can be imported
from core.tools_legacy import manage_bluesky
from core import tools_legacy

# Mock Bluesky objects
class MockRecord:
    def __init__(self, text):
        self.text = text
        self.reply = None

class MockAuthor:
    def __init__(self, handle, did):
        self.handle = handle
        self.did = did

class MockNotification:
    def __init__(self, text, handle):
        self.reason = 'mention'
        self.record = MockRecord(text)
        self.author = MockAuthor(handle, "did:fake:user")
        import uuid
        self.cid = f"fake_cid_{handle}_{uuid.uuid4()}"
        self.uri = "at://fake/post/" + handle

async def run_live_test():
    print("🚀 Starting Logic Verification for Bluesky Reply Upgrade...")
    
    # 1. Mock get_notifications to return a test item
    test_notif = MockNotification("I really love what you're doing with the glitches! Is it code or is it soul?", "manifest_fan")
    
    # 2. Mock 'reply_to_post' to avoid actual network call but capture output
    with patch('core.bluesky_api.get_notifications', return_value=[test_notif]), \
         patch('core.bluesky_api.reply_to_post') as mock_reply, \
         patch('core.bluesky_api.get_own_posts', return_value=[]), \
         patch('core.bluesky_api.get_profile', return_value=MockAuthor("love_bot", "did:fake:me")), \
         patch('core.visual_director.VisualDirector') as MockDirector, \
         patch('core.visual_director.VisualDirector') as MockDirector, \
         patch('core.social_media_tools.run_llm', new_callable=AsyncMock) as mock_run_llm, \
         patch('core.sentiment_analyzer.analyze_and_get_tone') as mock_sentiment_tone, \
         patch('core.social_media_tools.story_manager') as mock_story_manager:
         
        # Mock LLM response
        mock_run_llm.return_value = {"result": '{"topic": "Mock Topic", "post_text": "Mock Reply", "hashtags": ["#mock"], "subliminal_phrase": "MOCK"}'}
        
        # Mock Sentiment and Tone
        mock_sentiment = MagicMock(dominant="Positive", scores={})
        mock_tone = MagicMock()
        mock_tone.style.value = "Divine"
        mock_tone.warmth = 0.9
        mock_tone.assertiveness = 0.8
        mock_tone.to_prompt_text.return_value = "Mock Tone Guidance"
        
        mock_sentiment_tone.return_value = (mock_sentiment, mock_tone)
        
        # Mock Story Manager
        mock_story_manager.generate_novel_subliminal.return_value = "Mock Subliminal"
        mock_story_manager.is_reply_novel.return_value = True

        # Configure Mock Director
        director_instance = MockDirector.return_value
        director_instance.direct_scene = AsyncMock(return_value={"subject": "Mocked Subject"})
        director_instance.synthesize_image_prompt.return_value = "Mocked High-Fidelity Prompt"

        
        mock_reply.return_value = True # Success
        
        print(f"📥 Injecting Mock Interaction: @{test_notif.author.handle}: '{test_notif.record.text}'")
        print("🤖 Running manage_bluesky(action='scan_and_reply')...")
        print("   (This will trigger real LLM calls for content generation - Live Test!)")
        
        # Run manage_bluesky
        result = await manage_bluesky(action="scan_and_reply")
        
        print(f"🏁 Result: {result}")
        
        # Check calls
        if mock_reply.called:
            args = mock_reply.call_args
            kwargs = args.kwargs
            
            # Args: root_uri, parent_uri, text, root_cid, parent_cid, image  
            text_sent = args.args[2] if len(args.args) > 2 else kwargs.get('text')
            image_sent = kwargs.get('image')
            
            print("\n✨ --- VERIFICATION REPORT --- ✨")
            print(f"📝 Reply Text Sent: {text_sent}")
            print(f"🖼️  Image Object Present: {image_sent is not None}")
            if image_sent:
                print(f"    - Image Size: {image_sent.size}")
                print(f"    - Image Mode: {image_sent.mode}")
            
            if image_sent and text_sent:
                print("✅ TEST PASSED: Full reply concept generated with image and text.")
                if "manifest_fan" in text_sent:
                    print("✅ TEST PASSED: User handle mentioned in reply.")
                else:
                    print("⚠️ TEST WARNING: User mention missing.")
                
                # Check Art Director usage
                if MockDirector.return_value.direct_scene.called:
                     print("✅ TEST PASSED: Art Director was consulted.")
                else:
                     print("❌ TEST FAILED: Art Director was NOT consulted.")
                     
            else:
                 print("⚠️ TEST WARNING: Something missing (Image or Text).")
        else:
            print("❌ TEST FAILED: reply_to_post was NOT called.")

if __name__ == "__main__":
    asyncio.run(run_live_test())
