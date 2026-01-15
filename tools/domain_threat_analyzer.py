
import argparse
import sys
import os

def generate_report(domain, template_path):
    """
    Generates a threat report for the given email domain using a template.
    """
    if not os.path.exists(template_path):
        return f"Error: Template file not found at {template_path}"

    with open(template_path, 'r') as f:
        template = f.read()

    report = template.replace("{domain}", domain)
    return report

def main():
    """
    Main function to run the domain threat analyzer.
    """
    parser = argparse.ArgumentParser(description="Generate a threat report for an email domain.")
    parser.add_argument("domain", help="The email domain to analyze (e.g., gmail.com).")
    parser.add_argument("--template", default="tools/domain_threat_report_template.md", help="Path to the report template file.")
    args = parser.parse_args()

    report = generate_report(args.domain, args.template)
    print(report)

if __name__ == "__main__":
    main()
