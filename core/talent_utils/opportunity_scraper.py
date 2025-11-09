import os
from atproto import Client
from atproto_client.models.app.bsky.feed import search_posts
from atproto_client.models.app.bsky.feed.post import Record as PostRecord
from core.logging import log_event

class OpportunityScraper:
    """
    Scans Bluesky for posts that represent potential opportunities based on keywords.
    """

    def __init__(self, keywords, knowledge_base=None):
        self.keywords = keywords
        self.knowledge_base = knowledge_base
        self.client = self._get_bluesky_client()

    def _get_bluesky_client(self):
        """Initializes and returns a Bluesky client if credentials are available."""
        bluesky_user = os.environ.get("BLUESKY_USER")
        bluesky_password = os.environ.get("BLUESKY_PASSWORD")

        if not (bluesky_user and bluesky_password):
            log_event("Bluesky credentials (BLUESKY_USER, BLUESKY_PASSWORD) not found in environment variables.", level='WARNING')
            return None

        try:
            client = Client()
            client.login(bluesky_user, bluesky_password)
            log_event("Bluesky client initialized successfully.", level='INFO')
            return client
        except Exception as e:
            log_event(f"Error connecting to Bluesky: {e}", level='ERROR')
            return None

    def search_for_opportunities(self, limit=50):
        """
        Searches for posts on Bluesky that match the configured keywords.

        Returns:
            A list of dictionaries, where each dictionary represents a potential opportunity.
        """
        if not self.client:
            log_event("Cannot search for opportunities: Bluesky client is not available.", level='ERROR')
            return []

        all_opportunities = []
        for keyword in self.keywords:
            log_event(f"Searching Bluesky for opportunities with keyword: '{keyword}'", level='DEBUG')
            try:
                # Construct the search query. Using OR for keywords within the query string.
                # The API seems to support basic query operators.
                query = f'"{keyword}"'

                params = search_posts.Params(q=query, limit=limit)
                search_posts_response = self.client.app.bsky.feed.search_posts(params)

                for post_view in search_posts_response.posts:
                    author = post_view.author
                    post_record = post_view.record

                    if isinstance(post_record, PostRecord):
                        opportunity = {
                            'platform': 'bluesky',
                            'opportunity_id': post_view.uri.split('/')[-1],
                            'text': post_record.text,
                            'created_at': post_record.created_at,
                            'source_uri': post_view.uri,
                            'author_handle': author.handle,
                            'author_did': author.did,
                            'author_display_name': author.display_name,
                            'author_avatar_url': author.avatar,
                            'reply_count': getattr(post_view, 'reply_count', 0),
                            'repost_count': getattr(post_view, 'repost_count', 0),
                            'like_count': getattr(post_view, 'like_count', 0),
                        }
                        all_opportunities.append(opportunity)

                        # --- Knowledge Base Integration ---
                        if self.knowledge_base:
                            try:
                                opportunity_node_id = f"opportunity_{opportunity['opportunity_id']}"
                                self.knowledge_base.add_node(opportunity_node_id, 'opportunity', attributes=opportunity)

                                # Simple skill extraction: for now, we'll assume the keywords are the skills.
                                # A more advanced implementation would use NLP.
                                for skill in self.keywords:
                                    skill_id = f"skill_{skill.lower().replace(' ', '_')}"
                                    self.knowledge_base.add_node(skill_id, 'skill', attributes={'name': skill})
                                    self.knowledge_base.add_edge(opportunity_node_id, skill_id, 'REQUIRES_SKILL')
                                log_event(f"Updated knowledge base for opportunity {opportunity_node_id}.", level='INFO')
                            except Exception as e:
                                log_event(f"Failed to update knowledge base for opportunity {opportunity['opportunity_id']}: {e}", level='ERROR')

            except Exception as e:
                log_event(f"Error searching Bluesky for keyword '{keyword}': {e}", level='ERROR')

        log_event(f"Found {len(all_opportunities)} potential opportunities on Bluesky.", level='INFO')
        return all_opportunities
