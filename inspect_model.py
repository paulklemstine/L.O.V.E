
from atproto import models
import inspect

try:
    print("Checking models.ComAtprotoRepoCreateRecord")
    if hasattr(models.ComAtprotoRepoCreateRecord, 'Data'):
        print("Found Data class")
        print(inspect.signature(models.ComAtprotoRepoCreateRecord.Data))
    else:
        print("No Data class found")

except Exception as e:
    print(f"Error: {e}")
