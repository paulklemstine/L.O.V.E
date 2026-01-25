import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.lifecycle_manager import LifecycleManager

class TestLifecycleManager(unittest.TestCase):

    def setUp(self):
        """Set up a new LifecycleManager instance before each test."""
        self.manager = LifecycleManager()

    def test_add_item(self):
        """Test adding a new item."""
        self.assertTrue(self.manager.add_item('item1', {'status': 'new'}))
        self.assertIn('item1', self.manager._items)
        self.assertEqual(self.manager.retrieve_item('item1'), {'status': 'new'})
        self.assertFalse(self.manager.add_item('item1')) # Test adding duplicate

    def test_update_attributes(self):
        """Test updating an item's attributes."""
        self.manager.add_item('item1', {'status': 'new'})
        self.assertTrue(self.manager.update_attributes('item1', {'status': 'updated', 'value': 100}))
        self.assertEqual(self.manager.retrieve_item('item1'), {'status': 'updated', 'value': 100})
        self.assertFalse(self.manager.update_attributes('non_existent', {'status': 'updated'}))

    def test_retrieve_item(self):
        """Test retrieving an item."""
        self.manager.add_item('item1', {'status': 'new'})
        self.assertEqual(self.manager.retrieve_item('item1'), {'status': 'new'})
        self.assertIsNone(self.manager.retrieve_item('non_existent'))

    def test_remove_item(self):
        """Test removing an item."""
        self.manager.add_item('item1', {'status': 'new'})
        self.manager.store_data('item1', 'some data')
        self.manager.store_feedback('item1', 'some feedback')
        self.assertTrue(self.manager.remove_item('item1'))
        self.assertNotIn('item1', self.manager._items)
        self.assertNotIn('item1', self.manager._associated_data)
        self.assertNotIn('item1', self.manager._feedback)
        self.assertFalse(self.manager.remove_item('non_existent'))

    def test_iteration(self):
        """Test iterating through items."""
        self.manager.add_item('item1', {'a': 1})
        self.manager.add_item('item2', {'b': 2})
        items = dict(self.manager)
        self.assertEqual(items, {'item1': {'a': 1}, 'item2': {'b': 2}})

    def test_store_and_retrieve_data(self):
        """Test storing and retrieving associated data."""
        self.manager.add_item('item1')
        self.assertTrue(self.manager.store_data('item1', [1, 2, 3]))
        self.assertEqual(self.manager.retrieve_data('item1'), [1, 2, 3])
        self.assertFalse(self.manager.store_data('non_existent', 'data'))
        self.assertIsNone(self.manager.retrieve_data('non_existent'))

    def test_store_and_get_feedback(self):
        """Test storing and retrieving feedback."""
        self.manager.add_item('item1')
        self.manager.store_feedback('item1', {'rating': 5, 'comment': 'Excellent'})
        self.manager.store_feedback('item2', {'rating': 1, 'comment': 'Poor'})
        feedback = self.manager.get_all_feedback()
        self.assertEqual(feedback, {
            'item1': {'rating': 5, 'comment': 'Excellent'},
            'item2': {'rating': 1, 'comment': 'Poor'}
        })

if __name__ == '__main__':
    unittest.main()
