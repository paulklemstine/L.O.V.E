import unittest
from core.dataset_utils import refine_dataset

class TestRefineDataset(unittest.TestCase):
    def setUp(self):
        """Set up a sample dataset for testing."""
        self.sample_dataset = [
            {'id': 1, 'learning_rate': 0.8, 'error_correction': 0.9, 'innovation_score': 95},
            {'id': 2, 'learning_rate': 0.6, 'error_correction': 0.7, 'innovation_score': 80},
            {'id': 3, 'learning_rate': 0.9, 'error_correction': 0.95, 'innovation_score': 98},
            {'id': 4, 'learning_rate': 0.5, 'error_correction': 0.6, 'innovation_score': 70},
            {'id': 5, 'learning_rate': 0.7, 'error_correction': 0.8, 'innovation_score': 90},
            {'id': 6, 'learning_rate': 0.4, 'error_correction': 0.5, 'innovation_score': 60},
            {'id': 7, 'learning_rate': 'high', 'error_correction': 0.85, 'innovation_score': 92},
            {'id': 8, 'learning_rate': 0.95, 'error_correction': 0.98, 'innovation_score': 99},
        ]

    def test_successful_refinement(self):
        """Test the basic functionality with a clear case for refinement."""
        criteria = ['learning_rate', 'error_correction']
        refined_data = refine_dataset(self.sample_dataset, criteria)
        self.assertEqual(len(refined_data), 2)
        refined_ids = {d['id'] for d in refined_data}
        self.assertEqual(refined_ids, {3, 8})

    def test_empty_dataset(self):
        """Test that an empty dataset results in an empty list."""
        self.assertEqual(refine_dataset([], ['learning_rate']), [])

    def test_empty_criteria(self):
        """Test that empty criteria results in an empty list."""
        self.assertEqual(refine_dataset(self.sample_dataset, []), [])

    def test_no_significant_data(self):
        """Test a scenario where no data points meet the significance criteria."""
        dataset = [
            {'value': 1}, {'value': 1}, {'value': 1}, {'value': 1}
        ]
        self.assertEqual(refine_dataset(dataset, ['value']), [])

    def test_non_numeric_data_handling(self):
        """Ensure that entries with non-numeric data for a criterion are excluded."""
        criteria = ['learning_rate']
        refined_data = refine_dataset(self.sample_dataset, criteria)
        # Entry with id 7 has a non-numeric learning_rate and should be excluded.
        self.assertNotIn(7, [d['id'] for d in refined_data])

    def test_missing_criteria_in_data(self):
        """Test that a ValueError is raised if a criterion is not in the dataset."""
        with self.assertRaises(ValueError):
            refine_dataset(self.sample_dataset, ['non_existent_feature'])

if __name__ == '__main__':
    unittest.main()
