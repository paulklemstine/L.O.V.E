import time
import uuid
from core.structured_data_extractor import structured_content_acquisition
from core.graph_manager import GraphDataManager

def main():
    """
    Orchestrates the structured data extraction and integration process.
    """
    # Initialize the GraphDataManager
    knowledge_base = GraphDataManager()
    try:
        knowledge_base.load_graph('knowledge_base.graphml')
        print("Knowledge base loaded successfully.")
    except FileNotFoundError:
        print("No existing knowledge base found. A new one will be created.")

    # Define the configuration for data extraction
    configuration = {
        'sources': [
            {
                'type': 'url',
                'location': 'https://news.ycombinator.com/',
                'item_selectors': [
                    {
                        'selector': '.athing',
                        'fields': {
                            'title': {'type': 'css', 'value': '.titleline > a'},
                            'url': {'type': 'css', 'value': '.titleline > a[href]'}
                        },
                        'node_type': 'opportunity'
                    }
                ]
            },
            {
                'type': 'url',
                'location': 'https://nvd.nist.gov/vuln/search/results?form_type=Basic&results_type=overview&query=&search_type=all&isCpeNameSearch=false',
                'item_selectors': [
                    {
                        'selector': 'tr[data-testid^="vuln-row-"]',
                        'fields': {
                            'summary': {'type': 'css', 'value': 'p[data-testid^="vuln-summary-"]'},
                            'cve_id': {'type': 'css', 'value': 'a[data-testid^="vuln-detail-link-"]'}
                        },
                        'node_type': 'risk'
                    }
                ]
            }
        ]
    }

    # Run the data acquisition
    extracted_data = structured_content_acquisition(configuration)

    # Integrate the data into the GraphDataManager
    for item in extracted_data:
        node_id = str(uuid.uuid4())
        node_type = item.pop('node_type', 'generic')
        knowledge_base.add_node(node_id, node_type, attributes=item)

    # Save the updated knowledge base
    knowledge_base.save_graph('knowledge_base.graphml')
    print(f"Knowledge base updated with {len(extracted_data)} new items and saved.")

if __name__ == "__main__":
    while True:
        main()
        print("Orchestration complete. Waiting for the next run...")
        # Schedule to run every 24 hours
        time.sleep(60 * 60 * 24)
