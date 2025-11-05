import re
from collections import defaultdict
import requests
import json

def fetch_and_analyze_address(address):
    """
    Fetches all transactions for a given Ethereum address, analyzes them,
    and returns a summary of interesting findings.
    """
    rpc_url = "https://cloudflare-eth.com"
    headers = {"Content-Type": "application/json"}

    # Erc20 Transfer event signature
    erc20_transfer_signature = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "params": [{
            "fromBlock": "0x0",
            "toBlock": "latest",
            "address": None, # Query all addresses for transfers
            "topics": [
                erc20_transfer_signature,
                None, # from address - any
                f"0x000000000000000000000000{address[2:]}" # to address - padded
            ]
        }],
        "id": 1
    }

    try:
        response = requests.post(rpc_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        logs = response.json().get("result", [])
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        return {"error": f"Failed to fetch or decode blockchain data: {e}"}

    if not logs:
        return {"summary": "No ERC20 transfer events found for this address."}

    # --- Basic Analysis ---
    query = {
        "mode": "AND",
        "conditions": []
    }
    aggregations = {
        "total_transfers": {"field": None, "type": "count"},
        "unique_tokens": {"field": "address", "type": "unique_count"}
    }

    analysis_result = query_and_filter_data(logs, query, aggregations)

    summary = {
        "total_erc20_inbound_transfers": analysis_result["metadata"].get("total_transfers"),
        "unique_tokens_received": analysis_result["metadata"].get("unique_tokens")
    }

    return {
        "summary": summary,
        "raw_data": analysis_result["filtered_data"][:10] # Return first 10 for brevity
    }


def query_and_filter_data(data_collection, query_params, aggregations=None, extract_attributes=None):
    """
    Queries and filters a collection of structured data objects.

    Args:
        data_collection (list): A list of dictionaries or objects to query.
        query_params (dict): A dictionary defining the filtering criteria.
                             The structure is:
                             {
                                 "mode": "AND" | "OR",
                                 "conditions": [
                                     {"field": "field_name", "operator": "op", "value": "val"},
                                     ...
                                 ]
                             }
                             Supported operators:
                             - eq: equal to
                             - ne: not equal to
                             - gt: greater than
                             - lt: less than
                             - gte: greater than or equal to
                             - lte: less than or equal to
                             - in: value is in a list
                             - nin: value is not in a list
                             - contains: field contains a substring (for strings)
                             - regex: field matches a regular expression (for strings)
        aggregations (dict, optional): A dictionary defining aggregations to perform.
                                       The structure is:
                                       {
                                           "aggregation_name": {"field": "field_name", "type": "agg_type"}
                                       }
                                       Supported aggregation types:
                                       - count: total number of matching objects
                                       - sum: sum of values for a field
                                       - avg: average of values for a field
                                       - min: minimum value of a field
                                       - max: maximum value of a field
                                       - unique_count: number of unique values for a field
        extract_attributes (list, optional): A list of attribute keys to include in the
                                             filtered results. If None, returns the full objects.

    Returns:
        dict: A dictionary containing:
              - "filtered_data": A list of the matching objects (or their extracted attributes).
              - "metadata": A dictionary with the results of the aggregations.
    """
    filtered_results = []
    metadata = {}

    operator_map = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "in": lambda a, b: a in b,
        "nin": lambda a, b: a not in b,
        "contains": lambda a, b: b in a,
        "regex": lambda a, b: re.search(b, a) is not None,
    }

    # Helper to get nested attributes, now supports traversing lists.
    def get_attr(obj, key):
        keys = key.split('.')
        # Start with the root object in a list to handle the loop uniformly
        values = [obj]
        for k in keys:
            new_values = []
            # For each current value, find the next value based on the key 'k'
            for v in values:
                if v is None:
                    continue
                # If the current value is a list, iterate through it
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and k in item:
                            new_values.append(item.get(k))
                        elif hasattr(item, k):
                            new_values.append(getattr(item, k))
                # If it's a single object (dict or class instance)
                else:
                    if isinstance(v, dict) and k in v:
                        new_values.append(v.get(k))
                    elif hasattr(v, k):
                        new_values.append(getattr(v, k))
            values = new_values
            # If at any point the path breaks, stop.
            if not values:
                return None

        # If the final result is a list containing a single item, return that item.
        # Otherwise, return the list of all found values. This happens when
        # the path crosses a list (e.g., 'event_logs.event').
        if len(values) == 1:
            return values[0]
        return values if values else None


    # --- Filtering ---
    for item in data_collection:
        conditions = query_params.get("conditions", [])
        if not conditions:
            filtered_results.append(item)
            continue

        mode = query_params.get("mode", "AND").upper()

        evaluations = []
        for cond in conditions:
            field_values = get_attr(item, cond["field"])
            op = operator_map.get(cond["operator"])

            if op is None:
                evaluations.append(False)
                continue

            # If get_attr returned a list (from traversing a list in the data),
            # we check if ANY item in the list satisfies the condition.
            if isinstance(field_values, list):
                match_found = any(op(val, cond["value"]) for val in field_values if val is not None)
                evaluations.append(match_found)
            elif field_values is not None:
                evaluations.append(op(field_values, cond["value"]))
            else:
                evaluations.append(False) # Field does not exist

        if mode == "AND" and all(evaluations):
            filtered_results.append(item)
        elif mode == "OR" and any(evaluations):
            filtered_results.append(item)

    # --- Aggregation ---
    if aggregations and filtered_results:
        for agg_name, agg_spec in aggregations.items():
            field = agg_spec.get("field")
            agg_type = agg_spec["type"]

            if agg_type == "count":
                metadata[agg_name] = len(filtered_results)
                continue

            # For other aggregations, we need to handle nested and list values
            all_values = []
            for item in filtered_results:
                values = get_attr(item, field)
                if values is not None:
                    if isinstance(values, list):
                        all_values.extend(values)
                    else:
                        all_values.append(values)

            # Filter out None values before aggregation
            valid_values = [v for v in all_values if v is not None]

            if agg_type == "sum":
                metadata[agg_name] = sum(valid_values)
            elif agg_type == "avg":
                metadata[agg_name] = sum(valid_values) / len(valid_values) if valid_values else 0
            elif agg_type == "min":
                metadata[agg_name] = min(valid_values) if valid_values else None
            elif agg_type == "max":
                metadata[agg_name] = max(valid_values) if valid_values else None
            elif agg_type == "unique_count":
                 metadata[agg_name] = len(set(valid_values))


    # --- Attribute Extraction ---
    if extract_attributes:
        final_data = []
        for item in filtered_results:
            # Note: get_attr is designed for querying, so for extraction we use a simpler version
            def get_simple_attr(obj, key):
                try:
                    for k in key.split('.'):
                        obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
                    return obj
                except (KeyError, AttributeError):
                    return None
            extracted_item = {attr: get_simple_attr(item, attr) for attr in extract_attributes}
            final_data.append(extracted_item)
    else:
        final_data = filtered_results

    return {
        "filtered_data": final_data,
        "metadata": metadata
    }
