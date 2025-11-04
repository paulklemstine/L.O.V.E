# `OpportunityScraper` Documentation

## Overview

The `OpportunityScraper` class, located in `core/talent_utils/opportunity_scraper.py`, is a specialized component designed to scan the Bluesky social media platform for posts that may represent professional opportunities for creative talent. It operates by searching for posts containing a specified list of keywords.

## Core Functionality

- **Bluesky Integration:** It initializes a connection to the Bluesky API using credentials stored in the `BLUESKY_USER` and `BLUESKY_PASSWORD` environment variables.
- **Keyword-Based Search:** The `search_for_opportunities` method queries Bluesky's post search endpoint for each of the provided keywords.
- **Data Extraction:** It parses the search results, extracting key information about each post, including the text content, author details (handle, DID, display name), and engagement metrics (likes, reposts).
- **Structured Output:** The scraper formats the extracted data into a list of standardized Python dictionaries, each representing a potential opportunity. This structured format allows for easy processing by downstream components like the `OpportunityMatcher`.

## Task History

- **Task:** Develop a proactive opportunity discovery and matching system.
- **Date:** 2025-11-02
- **Pull Request:** TBD
- **Commit Hash:** TBD
