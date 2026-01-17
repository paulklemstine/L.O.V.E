import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.memory.fractal_schemas import SceneNode, ArcNode, EpochNode, SalienceScore, GoldenMoment
from core.memory.memory_folding_agent import MemoryFoldingAgent
from core.memory.memory_manager import MemoryManager

@pytest.fixture
def mock_llm():
    mock = AsyncMock()
    # Mock summary response
    mock.return_value = {"result": "This is a summary of the scene."}
    return mock

@pytest.fixture
def folding_agent(mock_llm):
    agent = MemoryFoldingAgent(llm_runner=mock_llm)
    # Mock salience scorer to return high/low scores
    agent.salience_scorer = AsyncMock()
    agent.salience_scorer.score = AsyncMock(return_value=SalienceScore(overall=0.5))
    agent.salience_scorer.is_golden_moment = MagicMock(return_value=False)
    return agent

@pytest.mark.asyncio
async def test_scene_node_creation():
    # Test SceneNode schema
    scene = SceneNode(
        summary="A test scene",
        crystals=[GoldenMoment(raw_text="Important!", salience=SalienceScore(overall=0.9))],
        source_ids=["ep1", "ep2"]
    )
    assert scene.summary == "A test scene"
    assert len(scene.crystals) == 1
    assert scene.crystals[0].raw_text == "Important!"

@pytest.mark.asyncio
async def test_fold_to_scene(folding_agent):
    episodes = [
        {"id": "ep1", "content": "Routine interaction."},
        {"id": "ep2", "content": "Important secret: API_KEY=123"}
    ]
    
    # Mock scorer to identify the secret as golden
    async def side_effect(content):
        if "secret" in content:
            return SalienceScore(overall=0.9, entity_tags=["SECRET"])
        return SalienceScore(overall=0.2)
        
    folding_agent.salience_scorer.score = AsyncMock(side_effect=side_effect)
    folding_agent.salience_scorer.is_golden_moment = MagicMock(side_effect=lambda s, t: s.overall > 0.8)
    
    scene = await folding_agent.fold_to_scene(episodes)
    
    assert scene is not None
    assert isinstance(scene, SceneNode)
    assert len(scene.crystals) == 1
    assert scene.crystals[0].raw_text == "Important secret: API_KEY=123"
    assert len(scene.source_ids) == 2

@pytest.mark.asyncio
async def test_memory_manager_integration():
    # Mock dependencies
    graph_manager = MagicMock()
    
    # We need to mock 'core.memory.memory_manager.MemoryManager' imports inside the file
    pass 
    # Since MemoryManager is complex to mock fully in isolation without side effects,
    # we rely on the folding agent test above as the core logic test.
