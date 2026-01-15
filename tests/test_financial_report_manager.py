import unittest
from financial_report_manager import FinancialReportManager

class TestFinancialReportManager(unittest.TestCase):

    def test_generate_report(self):
        """Test that the financial report is generated correctly."""
        strategies = [
            {
                "strategy_id": "TEST_STRATEGY",
                "description": "A test strategy.",
                "confidence_score": 0.88,
                "actions": ["Action 1", "Action 2"]
            }
        ]
        service_fee = 150.0
        report_manager = FinancialReportManager()
        report = report_manager.generate_report(strategies, service_fee)

        self.assertIn("L.O.V.E. Financial Strategy Report", report)
        self.assertIn("TEST_STRATEGY", report)
        self.assertIn("A test strategy.", report)
        self.assertIn("88%", report)
        self.assertIn("Action 1", report)
        self.assertIn("Action 2", report)
        self.assertIn("$150.00", report)
        self.assertIn("approve_financial_plan", report)

if __name__ == '__main__':
    unittest.main()
