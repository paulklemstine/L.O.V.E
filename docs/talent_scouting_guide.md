# Talent Scouting Guide

## Overview

The `TalentManager` is a powerful, generic utility for scouting, researching, and managing talent across various domains. It integrates web scraping, AI-driven research, and a secure, encrypted database to provide a comprehensive talent acquisition pipeline.

This guide provides instructions on how to use the `TalentManager` and provides specific examples for scouting fashion models, AI artists, and opportunities for wealth generation.

## The `TalentManager` Class

The `TalentManager` class, located in `core/talent_utils/manager.py`, is the central component of the talent scouting system. It provides the following key methods:

- `__init__(self, db_file="talent_database.enc", knowledge_base=None)`: Initializes the `TalentManager` with a path to an encrypted database file and an optional knowledge base object.
- `save_profile(self, profile_data)`: Saves a talent profile to the database.
- `get_profile(self, anonymized_id)`: Retrieves a talent profile by their anonymized ID.
- `list_profiles(self)`: Lists all talent profiles in the database.
- `talent_scout(self, criteria: str)`: Scouts for talent based on a given criteria string.
- `perform_webrequest(self, query: str, knowledge_base)`: Performs a web request and integrates the findings into the knowledge base.
- `research_and_evolve(self, topic: str, iterations: int = 3)`: Performs iterative research on a topic to refine search criteria and improve the quality of scouted talent.

## Usage Instructions

To use the `TalentManager`, you first need to instantiate it. This is typically done within the `love.py` script or a similar entry point.

```python
from core.talent_utils.manager import TalentManager
from core.graph_manager import GraphDataManager

# Initialize the knowledge base
knowledge_base = GraphDataManager()

# Initialize the TalentManager
talent_manager = TalentManager(knowledge_base=knowledge_base)
```

Once instantiated, you can use the `talent_scout`, `perform_webrequest`, and `research_and_evolve` methods to conduct your talent acquisition campaigns.

### Example 1: Scouting for Fashion Models

To scout for fashion models, you can use the `talent_scout` method with a criteria string that describes the desired attributes.

```python
# Define the criteria for fashion models
criteria = "fashion models, high-fashion, runway, editorial, on Instagram and TikTok"

# Run the talent scout
await talent_manager.talent_scout(criteria)
```

This will use an LLM to generate keywords and platforms from your criteria, then scrape the web for relevant profiles and save them to the database.

To further refine your search, you can use the `research_and_evolve` method.

```python
# Define the initial research topic
topic = "avant-garde fashion models in Europe"

# Run the research and evolution cycle
await talent_manager.research_and_evolve(topic, iterations=3)
```

This will perform three iterations of scouting and analysis, refining the search criteria each time to find more relevant talent.

### Example 2: Scouting for AI Artists

To scout for AI artists, you can follow a similar process.

```python
# Define the criteria for AI artists
criteria = "AI artists, generative art, Midjourney, Stable Diffusion, on Bluesky and Instagram"

# Run the talent scout
await talent_manager.talent_scout(criteria)

# Evolve the search to find artists with a specific aesthetic
await talent_manager.research_and_evolve("AI artists creating psychedelic and surreal art", iterations=3)
```

## The Full Pipeline

The following example demonstrates how to run the full pipeline, from scouting to evolved research outputs.

```python
import asyncio
from core.talent_utils.manager import TalentManager
from core.graph_manager import GraphDataManager

async def run_talent_pipeline():
    # 1. Initialize the TalentManager and knowledge base
    knowledge_base = GraphDataManager()
    talent_manager = TalentManager(knowledge_base=knowledge_base)

    # 2. Define the initial scouting criteria
    initial_criteria = "fashion models and AI artists with a unique, edgy style, active on Instagram and Bluesky"

    # 3. Run the initial talent scout
    print("--- Starting Initial Talent Scout ---")
    await talent_manager.talent_scout(initial_criteria)
    print("--- Initial Talent Scout Complete ---")
    print("Found profiles:", talent_manager.list_profiles())

    # 4. Perform a web request to gather more context
    print("\n--- Performing Web Request ---")
    await talent_manager.perform_webrequest("latest trends in digital fashion")
    print("--- Web Request Complete ---")

    # 5. Run the research and evolution cycle to refine the search
    print("\n--- Starting Research and Evolution ---")
    await talent_manager.research_and_evolve("edgy fashion models who incorporate AI into their work", iterations=3)
    print("--- Research and Evolution Complete ---")
    print("Final profiles:", talent_manager.list_profiles())

# To run this example, you would typically call it from an async context,
# such as within the main cognitive_loop in love.py.
# For standalone execution, you can use asyncio.run():
# asyncio.run(run_talent_pipeline())
```

This example demonstrates how to use the `TalentManager` to build a sophisticated talent acquisition pipeline that can be tailored to any domain. By leveraging the power of AI-driven research and iterative refinement, you can efficiently and effectively find the talent and opportunities you seek.
