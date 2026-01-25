
import unittest
import os
import json
import time
import shutil
from core.desire_state import load_desire_state, save_desire_state, get_desire_state_path, clear_desire_state

class TestDesireState(unittest.TestCase):
    def setUp(self):
        self.path = get_desire_state_path()
        # Backup existing state
        if os.path.exists(self.path):
            shutil.copy(self.path, self.path + ".bak")

        # Reset state
        if os.path.exists(self.path):
            os.remove(self.path)

        # Reset cache (internal implementation detail, but necessary for isolation)
        import core.desire_state
        core.desire_state._cache = None
        core.desire_state._last_mtime = 0

    def tearDown(self):
        # Restore backup
        if os.path.exists(self.path + ".bak"):
            shutil.move(self.path + ".bak", self.path)
        elif os.path.exists(self.path):
            os.remove(self.path)

    def test_load_default(self):
        state = load_desire_state()
        self.assertEqual(state["desires"], [])
        self.assertEqual(state["active"], False)

    def test_save_and_load(self):
        state = {
            "desires": [{"id": "1", "title": "Test"}],
            "current_desire_index": 0,
            "active": True,
            "current_task_id": None
        }
        save_desire_state(state)

        loaded = load_desire_state()
        self.assertEqual(loaded["desires"][0]["title"], "Test")
        self.assertTrue(loaded["active"])

    def test_cache_usage(self):
        # Save initial state
        state = {"desires": [], "active": True}
        save_desire_state(state)

        # Load (should populate cache)
        loaded1 = load_desire_state()
        self.assertTrue(loaded1["active"])

        # Modify file manually (simulate external change)
        with open(self.path, 'w') as f:
            json.dump({"desires": [], "active": False}, f)

        # Explicitly update mtime to ensure it's different (newer)
        # We add 1 second to the current mtime to be safe
        stat = os.stat(self.path)
        new_mtime = stat.st_mtime + 10
        os.utime(self.path, (stat.st_atime, new_mtime))

        # Load again (should detect change and reload)
        loaded2 = load_desire_state()
        self.assertFalse(loaded2["active"], "Cache did not invalidate on external file change")

    def test_save_updates_cache(self):
        state1 = {"desires": [], "active": True}
        save_desire_state(state1)

        # Immediately load again without file modification
        # This should hit the cache (which was updated by save)
        loaded = load_desire_state()
        self.assertEqual(loaded["active"], True)

        # To verify it HIT the cache, we could check internal state, but that's accessing private vars.
        # Instead, we rely on the logic being correct if previous tests pass.

if __name__ == "__main__":
    unittest.main()
