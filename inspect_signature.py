
import inspect
from atproto import Client

try:
    client = Client()
    # verify where create_record is
    print(f"Type of client.com.atproto.repo: {type(client.com.atproto.repo)}")
    method = client.com.atproto.repo.create_record
    print(f"Signature of create_record: {inspect.signature(method)}")
except Exception as e:
    print(f"Error inspecting: {e}")
