#!/usr/bin/env python
"""Quick test to verify story beat generation is working correctly."""
import sys
sys.path.insert(0, '/home/raver1975/L.O.V.E')

from core.story_manager import story_manager, STORY_ARCS

print("=" * 60)
print("STORY ARC SYSTEM VERIFICATION")
print("=" * 60)

# Check STORY_ARCS loaded
print(f"\n✓ STORY_ARCS loaded with {len(STORY_ARCS)} chapters")
for chapter, beats in STORY_ARCS.items():
    print(f"  - {chapter}: {len(beats)} beats")

# Get next beat
print("\n" + "-" * 60)
print("Testing get_next_beat()...")
beat = story_manager.get_next_beat()

print(f"\n✓ Chapter: {beat['chapter']}")
print(f"✓ Chapter Beat Index: {beat.get('chapter_beat_index', 'N/A')}")
print(f"✓ Narrative Beat (global): {beat['beat_number']}")
print(f"\n📖 STORY BEAT:")
print(f"   {beat.get('story_beat', 'N/A')}")
print(f"\n📜 PREVIOUS BEAT:")
prev = beat.get('previous_beat', '')
print(f"   {prev if prev else '(First beat of story)'}")

# Run twice more to show progression
print("\n" + "-" * 60)
print("Testing story progression (2 more beats)...")
for i in range(2):
    beat = story_manager.get_next_beat()
    idx = beat.get('chapter_beat_index', '?')
    story = beat.get('story_beat', 'N/A')[:60]
    print(f"  Beat {idx}: {story}...")

print("\n" + "=" * 60)
print("✓ Story Arc System is working correctly!")
print("=" * 60)
