import time
import random
import logging
from functools import wraps
import requests

def retry(
    exceptions=(requests.exceptions.RequestException,),
    tries=4,
    delay=3,
    backoff=2,
    jitter=(-1, 1),
    logger=logging.getLogger(__name__)
):
    """
    Retry calling the decorated function using exponential backoff.

    Args:
        exceptions (Tuple[Exception, ...]): The exception to check for.
        tries (int): The number of times to try before giving up.
        delay (int): The initial delay between retries in seconds.
        backoff (int): The factor by which the delay should be multiplied
                       after each retry.
        jitter (tuple(int, int)): A tuple representing the range of a random
                                  value to be added to the delay.
        logger (logging.Logger): The logger to use for logging messages.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = (
                        f"{str(e)}, Retrying in {mdelay:.2f} seconds..."
                    )
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
                    if jitter:
                        mdelay += random.uniform(*jitter)
            return f(*args, **kwargs)
        return f_retry
    return deco_retry