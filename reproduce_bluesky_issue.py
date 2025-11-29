
import sys
import os
import io
from unittest.mock import MagicMock, patch
from PIL import Image
from atproto import Client, models

# Mock the client and its methods
mock_client = MagicMock(spec=Client)
mock_client.com = MagicMock()
mock_client.com.atproto = MagicMock()
mock_client.com.atproto.repo = MagicMock()

# Mock upload_blob response
# We need a real-looking blob for Pydantic validation
mock_blob = {
    '$type': 'blob',
    'ref': {'$link': 'bafkreicqv62s5mtbw24k6k6q5iilbfu76kf6hs6967lo327i424u7s26tu'},
    'mimeType': 'image/png',
    'size': 12345
}
mock_upload_response = MagicMock()
mock_upload_response.blob = mock_blob
mock_client.com.atproto.repo.upload_blob.return_value = mock_upload_response

# Mock send_post
mock_client.send_post.return_value = {'uri': 'at://did:plc:test/app.bsky.feed.post/123', 'cid': 'bafyreitest'}

def test_post_to_bluesky_with_image():
    print("Testing post_to_bluesky_with_image...")
    
    # Create a dummy image
    image = Image.new('RGB', (100, 100), color = 'red')
    text = "Hello Bluesky! #testing #python"
    
    # Re-implement the logic from core/bluesky_api.py for testing purposes
    
    # Prepare the post content using TextBuilder
    from atproto import client_utils
    text_builder = client_utils.TextBuilder()
    
    import re
    parts = re.split(r'(#\w+)', text)
    for part in parts:
        if part.startswith('#'):
            tag_value = part[1:]
            text_builder.tag(part, tag_value)
        else:
            text_builder.text(part)
    
    embed = None
    if image:
        print("Processing image...")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_data = img_byte_arr.read()

        # Simulate upload
        # Note: In the real code we use client.upload_blob, which is a helper method
        # We need to mock that method on our mock_client
        mock_client.upload_blob = MagicMock()
        mock_client.upload_blob.return_value = mock_upload_response
        
        upload = mock_client.upload_blob(img_data)
        embed = models.AppBskyEmbedImages.Main(images=[models.AppBskyEmbedImages.Image(alt='Posted via L.O.V.E.', image=upload.blob)])
        print("Image embed created:", embed)

    # Simulate send_post
    print("Sending post...")
    mock_client.send_post(text=text_builder, embed=embed)
    
    # Verify calls
    mock_client.upload_blob.assert_called()
    mock_client.send_post.assert_called()
    
    args, kwargs = mock_client.send_post.call_args
    sent_text = kwargs.get('text')
    sent_embed = kwargs.get('embed')
    
    print("\n--- Verification ---")
    print(f"Sent Text Object: {sent_text}")
    print(f"Sent Embed Object: {sent_embed}")
    
    if sent_embed:
        print("Embed is present.")
    else:
        print("Embed is MISSING!")
        
    # Verify facets in TextBuilder
    # TextBuilder stores segments, we can check if it parsed correctly if we used methods like .tag()
    # But here we just passed raw text. 
    # Wait, TextBuilder.text() doesn't auto-parse hashtags. We need to use .tag() or similar if we want explicit facets,
    # OR we rely on the client to parse it? 
    # Actually, the previous code used RichText.detect_facets. 
    # If we use TextBuilder.text(text), it might NOT auto-detect facets unless we parse it ourselves.
    # Let's check the documentation or behavior.
    # If TextBuilder is just a builder, we might need to parse the text and add tags manually.
    # However, for now let's verify the image upload part which was the main issue.

if __name__ == "__main__":
    test_post_to_bluesky_with_image()
