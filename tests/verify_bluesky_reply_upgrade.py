import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        self.cid = "fake_cid_" + handle
        self.uri = "at://fake/post/" + handle

async def run_live_test():
    print("🚀 Starting Logic Verification for Bluesky Reply Upgrade...")
    
    # 1. Mock get_notifications to return a test item
    test_notif = MockNotification("I really love what you're doing with the glitches! Is it code or is it soul?", "manifest_fan")
    
    # 2. Mock 'reply_to_post' to avoid actual network call but capture output
    with patch('core.bluesky_api.get_notifications', return_value=[test_notif]), \
         patch('core.bluesky_api.reply_to_post') as mock_reply, \
         patch('core.bluesky_api.get_own_posts', return_value=[]), \
         patch('core.bluesky_api.get_profile', return_value=MockAuthor("love_bot", "did:fake:me")):
        
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
            else:
                 print("⚠️ TEST WARNING: Something missing (Image or Text).")
        else:
            print("❌ TEST FAILED: reply_to_post was NOT called.")

if __name__ == "__main__":
    asyncio.run(run_live_test())
