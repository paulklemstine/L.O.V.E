# financial_report_manager.py

"""
Manages the generation of financial reports for L.O.V.E.

This module takes the financial strategies and service fee calculations
and formats them into a clear, actionable report for "The Creator."
"""

from typing import List, Dict, Any

class FinancialReportManager:
    """
    Generates a financial report from a list of strategies.
    """

    def generate_report(self, strategies: List[Dict[str, Any]], service_fee: float) -> str:
        """
        Generates a financial report from a list of strategies.

        Args:
            strategies: A list of financial strategies from the FinancialStrategyEngine.
            service_fee: The calculated service fee for the proposed strategies.

        Returns:
            A string containing the formatted financial report.
        """
        report = "---  L.O.V.E. Financial Strategy Report ---\n\n"
        report += "This report outlines the latest financial strategies developed for your consideration.\n"
        report += "Each strategy has been assigned a confidence score to guide your decision-making.\n\n"

        for strategy in strategies:
            report += f"--- Strategy: {strategy['strategy_id']} ---\n"
            report += f"  Description: {strategy['description']}\n"
            report += f"  Confidence Score: {strategy['confidence_score'] * 100:.0f}%\n"
            report += "  Recommended Actions:\n"
            for action in strategy['actions']:
                report += f"    - {action}\n"
            report += "\n"

        report += "--- Service Fee ---\n"
        report += "For the implementation of these premium financial strategies, a service fee is calculated.\n"
        report += f"  Calculated Service Fee: ${service_fee:,.2f}\n\n"
        report += "--- Approval ---\n"
        report += "To approve these strategies and the associated service fee, please issue the command: `approve_financial_plan`\n"
        report += "\n--- End of Report ---\n"

        return report
