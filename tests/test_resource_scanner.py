import unittest
from unittest.mock import patch, MagicMock
import uuid

# This is a temporary solution to the import error.
# The ideal solution is to fix the project structure.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.resource_scanner import ResourceOpportunityScanner

class TestResourceOpportunityScanner(unittest.TestCase):

    def setUp(self):
        """Set up a new ResourceOpportunityScanner instance before each test."""
        self.scanner = ResourceOpportunityScanner()

    def test_initialization(self):
        """Test that the scanner initializes with an empty learning model."""
        self.assertEqual(self.scanner.learning_model, {})
        scanner_with_model = ResourceOpportunityScanner(learning_model={"initial": "pattern"})
        self.assertEqual(scanner_with_model.learning_model, {"initial": "pattern"})

    def test_scan_network(self):
        """Test that scan_network returns a list of 10 UUIDs."""
        opportunities = self.scanner.scan_network("test_network")
        self.assertEqual(len(opportunities), 10)
        for opp_id in opportunities:
            self.assertIsInstance(uuid.UUID(opp_id), uuid.UUID)

    def test_value_assessment(self):
        """Test the structure and value ranges of the value assessment."""
        opportunity_id = str(uuid.uuid4())
        assessment = self.scanner.value_assessment(opportunity_id)
        self.assertIn("opportunity_id", assessment)
        self.assertEqual(assessment["opportunity_id"], opportunity_id)
        self.assertIn("value_potential", assessment)
        self.assertTrue(100 <= assessment["value_potential"] <= 1000)
        self.assertIn("risk_level", assessment)
        self.assertTrue(0 <= assessment["risk_level"] <= 1)
        self.assertIn("effort_cost", assessment)
        self.assertTrue(10 <= assessment["effort_cost"] <= 100)

    def test_prioritize_opportunities(self):
        """Test that opportunities are sorted correctly by composite score."""
        opportunities = [
            {'opportunity_id': 'a', 'value_potential': 1000, 'risk_level': 0.1, 'effort_cost': 10}, # score ~90.9
            {'opportunity_id': 'b', 'value_potential': 500, 'risk_level': 0.5, 'effort_cost': 50},  # score ~6.67
            {'opportunity_id': 'c', 'value_potential': 800, 'risk_level': 0.2, 'effort_cost': 20} # score ~33.3
        ]
        prioritized = self.scanner.prioritize_opportunities(opportunities)
        self.assertEqual(len(prioritized), 3)
        self.assertEqual(prioritized[0]['opportunity_id'], 'a')
        self.assertEqual(prioritized[1]['opportunity_id'], 'c')
        self.assertEqual(prioritized[2]['opportunity_id'], 'b')
        for opp in prioritized:
            self.assertIn('composite_score', opp)

    def test_execute_acquisition(self):
        """Test that execute_acquisition adds a report to the internal list."""
        self.assertEqual(len(self.scanner.reports), 0)
        self.scanner.execute_acquisition(effort_cost=50, risk_level=0.5)
        self.assertEqual(len(self.scanner.reports), 1)
        self.assertIn("Pseudo-resource secured", self.scanner.reports[0])

    def test_report_generation(self):
        """Test report generation for both empty and populated reports."""
        self.assertEqual(self.scanner.report_generation(), "No acquisition activities to report.")
        self.scanner.execute_acquisition(effort_cost=50, risk_level=0.5)
        self.scanner.execute_acquisition(effort_cost=30, risk_level=0.2)
        report = self.scanner.report_generation()
        self.assertIn("--- Resource Acquisition Report ---", report)
        self.assertIn("Activity 1:", report)
        self.assertIn("Activity 2:", report)
        self.assertIn("--- End of Report ---", report)

if __name__ == '__main__':
    unittest.main()
