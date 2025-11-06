from bs4 import BeautifulSoup

class KnowledgeExtractor:
    """
    Parses raw data from various sources and enriches the knowledge base.
    """

    def __init__(self, knowledge_base):
        """
        Initializes the KnowledgeExtractor.

        Args:
            knowledge_base: An instance of GraphDataManager.
        """
        self.knowledge_base = knowledge_base

    def parse_probe_data(self, host_ip, ports_data):
        """
        Parses the results of a network probe (nmap scan) and updates the knowledge base.

        Args:
            host_ip: The IP address of the scanned host.
            ports_data: A dictionary of open ports and their associated services.
        """
        if not ports_data:
            return

        for port, data in ports_data.items():
            service = data.get('service', 'unknown')
            version = data.get('version', 'unknown')
            software_id = f"{service}_{version}"

            # Add a node for the software if it doesn't exist
            if not self.knowledge_base.get_node(software_id):
                self.knowledge_base.add_node(software_id, 'software', attributes={'service': service, 'version': version})

            # Create a relationship between the host and the software
            self.knowledge_base.add_edge(host_ip, software_id, 'runs_software', attributes={'port': port})

    def parse_web_content(self, url, html_content):
        """
        Parses the HTML content of a web page to extract the title and links.

        Args:
            url: The URL of the web page.
            html_content: The HTML content of the page.
        """
        if not html_content:
            return

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract and store the title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            self.knowledge_base.add_node(url, 'webrequest', attributes={'title': title})

        # Extract and store links
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Basic validation to filter out relative links for now
            if href.startswith('http'):
                self.knowledge_base.add_node(href, 'webrequest', attributes={})
                self.knowledge_base.add_edge(url, href, 'links_to')
