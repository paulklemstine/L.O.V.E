# `OpportunityMatcher` Documentation

## Overview

The `OpportunityMatcher` class, located in `core/talent_utils/opportunity_matcher.py`, serves as the intelligent core of the proactive opportunity discovery system. Its primary function is to perform a sophisticated, nuanced analysis of potential opportunities scraped from online platforms and match them against the profiles of creative professionals stored in the talent database.

## Core Functionality

- **LLM-Powered Analysis:** The matcher leverages a Large Language Model (LLM) to go beyond simple keyword matching. It evaluates the relevance, aesthetic alignment, and professional suitability of an opportunity for a specific talent.
- **Contextual Prompt Engineering:** For each opportunity-talent pair, it dynamically constructs a detailed prompt for the LLM. This prompt includes a summary of both the opportunity and the talent's profile, along with specific instructions for the analysis.
- **Nuanced Matching Criteria:** The LLM is instructed to evaluate the match based on several factors:
    - **Relevance:** How well the opportunity fits the talent's skills and field.
    - **Aesthetic/Vibe:** The alignment of brand and tone between the opportunity and the talent.
    - **Professionalism:** The suitability of the talent based on their analyzed traits.
    - **Actionability:** Whether the opportunity is a clear and actionable request.
- **Structured JSON Output:** The LLM is required to return its analysis in a structured JSON format, which includes a boolean `is_match` flag, a 1-100 `match_score`, a `reasoning` text, and an `opportunity_type` category. This ensures the output is predictable and easy to parse.
- **Asynchronous Operation:** The `find_matches` method is an `async` function, allowing it to perform multiple LLM calls concurrently, which is crucial for efficiently processing many opportunities against many profiles.

## Task History

- **Task:** Develop a proactive opportunity discovery and matching system.
- **Date:** 2025-11-02
- **Pull Request:** TBD
- **Commit Hash:** TBD
