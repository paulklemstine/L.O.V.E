import requests
from bs4 import BeautifulSoup
import re
import json
from jsonpath_ng import jsonpath, parse

def structured_content_acquisition(configuration):
    """
    Acquires and transforms content from various sources based on a configuration.

    Args:
        configuration: A dictionary detailing data sources and parsing strategies.

    Returns:
        A list of structured data objects.
    """
    results = []
    for source in configuration.get('sources', []):
        try:
            if source['type'] == 'url':
                response = requests.get(source['location'])
                response.raise_for_status()
                content = response.text

                for item_selector in source.get('item_selectors', []):
                    soup = BeautifulSoup(content, 'html.parser')
                    for item in soup.select(item_selector['selector']):
                        entity = {}
                        for field, field_selector in item_selector.get('fields', {}).items():
                            if field_selector['type'] == 'css':
                                element = item.select_one(field_selector['value'])
                                if element:
                                    entity[field] = element.text.strip()
                            elif field_selector['type'] == 'regex':
                                match = re.search(field_selector['value'], str(item))
                                if match:
                                    entity[field] = match.group(1)
                        if entity:
                            entity['node_type'] = item_selector.get('node_type', 'generic')
                            results.append(entity)
            elif source['type'] == 'api':
                response = requests.get(source['location'])
                response.raise_for_status()
                data = response.json()

                jsonpath_expression = parse(source.get('json_path'))
                for match in jsonpath_expression.find(data):
                    item = match.value
                    item['node_type'] = source.get('node_type', 'generic')
                    results.append(item)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {source.get('location')}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return results
