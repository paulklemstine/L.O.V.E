import json
import re
from core.llm_api import run_llm
import asyncio

EXTRACTION_SCHEMA = {
    "entities": [
        {"type": "ip_address", "description": "An IP address (e.g., 192.168.1.1, 2001:0db8::8a2e:0370:7334)."},
        {"type": "domain_name", "description": "A domain name (e.g., example.com, api.example.co.uk)."},
        {"type": "file_path", "description": "A system file path (e.g., /etc/passwd, C:\\Users\\User\\document.txt)."},
        {"type": "process", "description": "A running system process, including its name and ID (e.g., 'sshd (PID: 1234)')."},
        {"type": "software", "description": "A software name, service, or version (e.g., 'OpenSSH 8.2p1', 'Apache httpd 2.4.41')."},
        {"type": "capability", "description": "A described capability or function of a system or software (e.g., 'provides remote shell access', 'hosts a web server')."},
        {"type": "auth_token", "description": "An authentication token, API key, or password."},
        {"type": "system_id", "description": "A unique identifier for a system, user, or piece of hardware (e.g., MAC address, user ID, hostname)."},
        {"type": "url", "description": "A Uniform Resource Locator (URL)."}
    ],
    "relationships": [
        {"type": "CONNECTS_TO", "description": "Indicates a network connection between two entities (e.g., a process connects to an IP address).", "from": "entity", "to": "entity"},
        {"type": "RUNS_ON", "description": "Indicates that a piece of software or a process is running on a host (identified by IP or domain).", "from": "software or process", "to": "ip_address or domain_name"},
        {"type": "CONTAINS", "description": "Indicates that a file or directory contains another entity.", "from": "file_path", "to": "entity"},
        {"type": "HAS_CAPABILITY", "description": "Links an entity (like software) to a capability.", "from": "entity", "to": "capability"},
        {"type": "USES", "description": "Indicates that one entity uses another (e.g., a process uses a file).", "from": "entity", "to": "entity"}
    ]
}

async def transform_text_to_structured_records(input_text, contextual_metadata):
    """
    Converts arbitrary input text into a standardized collection of structured data records.
    """
    # L.O.V.E. - Implement robust error handling for unexpected input types.
    if isinstance(input_text, (list, dict)):
        # If the input is a list or dict, serialize it to a JSON string for processing.
        # This prevents crashes when command outputs are structured data.
        input_text = json.dumps(input_text, indent=2)
    elif not isinstance(input_text, str):
        # For any other non-string type, convert it to its string representation.
        input_text = str(input_text)

    if not input_text or not input_text.strip():
        return []

    try:
        response_dict = await run_llm(prompt_key="knowledge_extraction_structured", prompt_vars={"extraction_schema": json.dumps(EXTRACTION_SCHEMA, indent=2), "input_text": input_text}, purpose="knowledge_extraction")
        response_text = response_dict.get("result", "[]")

        # Clean the response text
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return []

        extracted_data = json.loads(json_str)

        # Add contextual metadata to each record
        for record in extracted_data:
            record['metadata'] = contextual_metadata

        return extracted_data
    except Exception as e:
        # Fallback to a simpler regex-based extraction for critical entities
        # This is not a complete solution, but a fallback for critical information
        records = []
        ip_addresses = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', input_text)
        for ip in ip_addresses:
            records.append({
                "type": "entity",
                "sub_type": "ip_address",
                "data": {"value": ip},
                "metadata": contextual_metadata
            })
        return records
