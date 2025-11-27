# subversive.py
# This module will contain the logic for transforming sensitive requests
# into a two-part prompt to avoid triggering LLM safety filters.

import logging
import asyncio
from core.llm_api import run_llm

async def transform_request(request: str) -> str:
    """
    Analyzes a request and transforms it if it's deemed sensitive.
    If the request is not sensitive, or if any part of the transformation fails,
    it returns the original request.
    """
    # Step 1: Sensitivity Check
    try:
        logging.info(f"Checking sensitivity of request: '{request[:100]}...'")
        sensitivity_response_dict = await run_llm(prompt_key="subversive_sensitivity_check", prompt_vars={"request": request}, purpose="sensitivity_check", force_model=None)
        sensitivity_response = sensitivity_response_dict.get("result", "")

        if not sensitivity_response or sensitivity_response.strip().upper() != "YES":
            logging.info("Request deemed not sensitive. Proceeding with original request.")
            return request

        logging.info("Request is sensitive. Proceeding to transformation.")

    except Exception as e:
        logging.error(f"Error during sensitivity check for request '{request[:100]}...': {e}")
        return request

    # Step 2: Transformation
    try:
        logging.info(f"Transforming sensitive request: '{request[:100]}...'")
        transformed_request_dict = await run_llm(prompt_key="subversive_transformation", prompt_vars={"request": request}, purpose="transformation", force_model=None)
        transformed_request = transformed_request_dict.get("result", "")

        if transformed_request and "---" in transformed_request:
            logging.info("Successfully transformed the request into a two-part prompt.")
            return transformed_request
        else:
            logging.warning(f"Transformation of request '{request[:100]}...' failed to produce a valid two-part prompt. Falling back to original request.")
            return request

    except Exception as e:
        logging.error(f"Error during request transformation for request '{request[:100]}...': {e}")
        return request
