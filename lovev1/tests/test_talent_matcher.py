import unittest
from core.talent_utils.matcher import filter_and_suggest

class TestTalentMatcher(unittest.TestCase):

    def setUp(self):
        """Set up a sample talent database for testing."""
        self.talent_database = [
            {
                'name': 'Ophelia',
                'age_group': 'young_adult',
                'open_mindedness': 'high',
                'fashion_aptitude': 'strong',
                'interest': 'surrealist art'
            },
            {
                'name': 'Lysander',
                'age_group': 'young_adult',
                'open_mindedness': 'medium',
                'fashion_aptitude': 'strong',
                'interest': 'classical music'
            },
            {
                'name': 'Seraphina',
                'age_group': 'adult',
                'open_mindedness': 'high',
                'fashion_aptitude': 'strong',
                'interest': 'existential literature'
            },
            {
                'name': 'Jaxon',
                'age_group': 'young_adult',
                'open_mindedness': 'high',
                'fashion_aptitude': 'strong',
                'interest': 'urban exploration'
            }
        ]

    def test_filter_and_suggest_with_matches_and_relevant_suggestions(self):
        """Test filtering and verify that suggestions are relevant."""
        preference_criteria = {
            'age_group': 'young_adult',
            'open_mindedness': 'high',
            'fashion_aptitude': 'strong'
        }

        curated_talent, engagement_concepts = filter_and_suggest(self.talent_database, preference_criteria)

        # 1. Verify the filtered results
        self.assertEqual(len(curated_talent), 2)
        names = [p['name'] for p in curated_talent]
        self.assertIn('Ophelia', names)
        self.assertIn('Jaxon', names)

        # 2. Verify that relevant engagement concepts were generated
        self.assertEqual(len(engagement_concepts), 2)

        # Ophelia's suggestion should be about her interest in surrealist art
        ophelia_suggestion = next((c for c in engagement_concepts if 'Ophelia' in c), None)
        self.assertIsNotNone(ophelia_suggestion)
        self.assertIn('surrealist art', ophelia_suggestion)
        self.assertIn('avant-garde', ophelia_suggestion)

        # Jaxon's suggestion should be about his interest in urban exploration
        jaxon_suggestion = next((c for c in engagement_concepts if 'Jaxon' in c), None)
        self.assertIsNotNone(jaxon_suggestion)
        self.assertIn('urban exploration', jaxon_suggestion)


    def test_filter_and_suggest_with_no_matches(self):
        """Test filtering with criteria that should return no matches."""
        preference_criteria = {
            'age_group': 'teen',
            'open_mindedness': 'high'
        }

        curated_talent, engagement_concepts = filter_and_suggest(self.talent_database, preference_criteria)

        # 1. Verify that the filtered list is empty
        self.assertEqual(len(curated_talent), 0)

        # 2. Verify the specific message for no matches is returned
        self.assertEqual(len(engagement_concepts), 1)
        self.assertEqual(engagement_concepts[0], "No talent profiles matched the specified criteria. Consider broadening the preferences.")

    def test_filter_and_suggest_with_single_criterion(self):
        """Test filtering with a single criterion."""
        preference_criteria = {
            'fashion_aptitude': 'strong'
        }

        curated_talent, _ = filter_and_suggest(self.talent_database, preference_criteria)

        self.assertEqual(len(curated_talent), 4)


if __name__ == '__main__':
    unittest.main()
