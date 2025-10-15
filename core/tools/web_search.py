class WebSearchTool:
    def __init__(self):
        self.name = "web_search"
        self.description = "Searches the web for a given query."

    def run(self, query):
        """Simulates running a web search."""
        print(f"WebSearchTool: Searching for '{query}'...")
        return f"Simulated search results for '{query}'."