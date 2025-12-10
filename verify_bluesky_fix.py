import asyncio
import os
import sys
import json
from unittest.mock import MagicMock, patch, mock_open

# Ensure core is in path
sys.path.append(os.getcwd())

# Mock environment
os.environ['BLUESKY_USER'] = 'testuser'
os.environ['BLUESKY_PASSWORD'] = 'testpass'

# Mock dependencies
# Mock external dependencies BEFORE importing core modules
sys.modules['atproto'] = MagicMock()
sys.modules['atproto_client'] = MagicMock()
# Set __path__ to satisfy "is not a package" check if needed, but extensive mocking usually solves it
sys.modules['atproto_client'].__path__ = [] 
sys.modules['atproto_client.models'] = MagicMock()
sys.modules['atproto_client.models.app'] = MagicMock()
sys.modules['atproto_client.models.app.bsky'] = MagicMock()
sys.modules['atproto_client.models.app.bsky.feed'] = MagicMock()
sys.modules['atproto_client.models.app.bsky.notification'] = MagicMock() 
sys.modules['atproto_client.models.com'] = MagicMock()
sys.modules['atproto_client.models.com.atproto'] = MagicMock()
sys.modules['atproto_client.models.com.atproto.repo'] = MagicMock()
sys.modules['atproto_client.exceptions'] = MagicMock()
sys.modules['atproto_client.exceptions'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# Mock internal heavy/irrelevant modules to avoid import chains
sys.modules['core.talent_utils'] = MagicMock()
sys.modules['core.talent_utils.aggregator'] = MagicMock()
sys.modules['core.talent_utils.analyzer'] = MagicMock()
sys.modules['core.researcher'] = MagicMock()
# Ensure we don't trigger heavy logic
sys.modules['network'] = MagicMock()
sys.modules['ipfs'] = MagicMock() 

# Explicit import to ensure patch finds it
import core.bluesky_api

async def main():
    print("Starting Bluesky Fix Verification...")
    
    # We patch the imports used INSIDE tools_legacy.py
    with patch('core.bluesky_api.get_notifications') as mock_get_notifs, \
         patch('core.bluesky_api.get_profile') as mock_get_profile, \
         patch('core.bluesky_api.reply_to_post') as mock_reply, \
         patch('core.llm_api.run_llm') as mock_llm, \
         patch('core.image_api.generate_image') as mock_gen_image, \
         patch('builtins.open', mock_open(read_data='{"replied": [], "ignored": []}')) as mock_file:

        # 1. Setup Mock Profile (Me)
        mock_me = MagicMock()
        mock_me.did = "did:plc:my_bot_did"
        mock_get_profile.return_value = mock_me

        # 2. Setup Mock Notifications
        # Case A: Valid reply from someone else
        notif1 = MagicMock()
        notif1.reason = 'reply'
        notif1.author.handle = "fan_user"
        notif1.author.did = "did:plc:fan_user"
        notif1.record.text = "Hello LOVE! Are you real?"
        notif1.uri = "uri_1"
        notif1.cid = "cid_1"
        # Nested reply structure
        notif1.record.reply.root.uri = "root_uri"
        notif1.record.reply.root.cid = "root_cid"

        # Case B: Self-reply (Should be ignored)
        notif2 = MagicMock()
        notif2.reason = 'reply'
        notif2.author.handle = "my_bot_handle"
        notif2.author.did = "did:plc:my_bot_did" # Matches Me
        notif2.record.text = "I am talking to myself."
        notif2.uri = "uri_2"
        notif2.cid = "cid_2"
        
        # Case C: Already Replied (Mocked via file read data, but let's test logic if we can inject it? 
        # Actually mock_open read_data is static. Let's assume cid_1 is new.)
        
        mock_get_notifs.return_value = [notif1, notif2]

        # 3. Setup LLM Responses
        def llm_side_effect(*args, **kwargs):
            purpose = kwargs.get('purpose')
            print(f"  -> LLM Called with purpose: {purpose}")
            
            if purpose == 'social_decision':
                # DECISION: Reply to fan_user
                return {'result': '```json\n{"decision": "REPLY"}\n```'}
            
            elif purpose == 'social_reply_gen':
                # GEN: JSON response
                return {'result': '```json\n{"text": "I am very real!", "hashtags": ["#ai"], "image_prompt": "cyber heart"}\n```'}
            
            elif purpose == 'social_sanitizer':
                # SANITIZER: Clean text
                return {'result': 'I am very real! #ai'}
            
            return {'result': ''}

        mock_llm.side_effect = llm_side_effect
        mock_reply.return_value = True

        # 4. Run manage_bluesky
        from core.tools_legacy import manage_bluesky
        
        print("\nExecuting manage_bluesky(action='scan_and_reply')...")
        result = await manage_bluesky(action='scan_and_reply')
        print(f"Result: {result}")

        # 5. Assertions
        print("\nVerifying Logic...")
        
        # Check if NOTIFICATIONS were fetched
        if mock_get_notifs.called:
            print("fetch_notifications called.")
        else:
            print("fetch_notifications NOT called.")

        # Check if PROFILE was fetched
        if mock_get_profile.called:
            print("get_profile called.")
        else:
            print("get_profile NOT called.")

        # Check if REPLY was sent to fan_user (cid_1)
        # We expect reply_to_post to be called with cid_1 related args
        # root_cid='root_cid', parent_cid='cid_1'
        
        reply_calls = mock_reply.call_args_list
        found_reply = False
        for call in reply_calls:
            args, kwargs = call
            # args: root_uri, parent_uri, text
            # kwargs: root_cid, parent_cid
            if kwargs.get('parent_cid') == 'cid_1':
                found_reply = True
                print("Reply sent to fan_user (cid_1).")
                break
        
        if not found_reply:
            print(f"Reply NOT sent to fan_user. Calls: {reply_calls}")

        # Check if SELF-REPLY (cid_2) was IGNORED
        # We expect NO reply call for cid_2
        found_self_reply = False
        for call in reply_calls:
            args, kwargs = call
            if kwargs.get('parent_cid') == 'cid_2':
                found_self_reply = True
                break
        
        if not found_self_reply:
            print("Self-reply (cid_2) correctly ignored.")
        else:
            print("Self-reply (cid_2) was NOT ignored!")

        # Check Sanitizer Call
        sanitizer_called = any(call.kwargs.get('purpose') == 'social_sanitizer' for call in mock_llm.call_args_list)
        if sanitizer_called:
            print("Sanitizer LLM pass executed.")
        else:
            print("Sanitizer LLM pass MISSED.")

if __name__ == '__main__':
    asyncio.run(main())
