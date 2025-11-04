import re
import datetime
from typing import Dict, Any, Callable, Tuple, Optional

def process_and_compose(
    input_attributes: Dict[str, Any],
    composition_template: str,
    adaptive_logic_definitions: Dict[str, Dict[str, Callable]],
    observation_metrics_stub: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Processes input attributes with a template and adaptive logic to compose an output string.

    Args:
        input_attributes: A dictionary containing data points.
        composition_template: A string with placeholders and conditional logic.
            - Placeholders: {key} or {key|transformation_name}
            - Conditional logic: [[IF condition_name]]...[[ENDIF]]
        adaptive_logic_definitions: A dictionary with 'conditions' and 'transformations'.
            - 'conditions': A dict mapping condition names to functions that take
              input_attributes and return a boolean.
            - 'transformations': A dict mapping transformation names to functions that
              take a value and return a transformed value.
        observation_metrics_stub: An optional dictionary to pre-populate the logging entry.

    Returns:
        A tuple containing:
        - The composed output string.
        - A logging entry blueprint dictionary.
    """
    if observation_metrics_stub is None:
        logging_entry_blueprint = {}
    else:
        logging_entry_blueprint = observation_metrics_stub.copy()

    composition_metadata = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'conditions_triggered': [],
        'transformations_used': [],
        'missing_attributes': [],
        'missing_logic': [],
    }

    processed_template = composition_template

    condition_definitions = adaptive_logic_definitions.get('conditions', {})

    def evaluate_condition_block(match):
        condition_name = match.group(1).strip()
        content = match.group(2)

        condition_func = condition_definitions.get(condition_name)
        if condition_func:
            try:
                if condition_func(input_attributes):
                    composition_metadata['conditions_triggered'].append(condition_name)
                    return content
                else:
                    return ""
            except Exception as e:
                composition_metadata['missing_logic'].append(
                    {'type': 'condition', 'name': condition_name, 'error': str(e)}
                )
                return ""
        else:
            composition_metadata['missing_logic'].append(
                {'type': 'condition', 'name': condition_name, 'error': 'Not found'}
            )
            return ""

    conditional_pattern = re.compile(r'\[\[IF (.*?)]](.*?)\[\[ENDIF]]', re.DOTALL)
    processed_template = conditional_pattern.sub(evaluate_condition_block, processed_template)

    transformation_definitions = adaptive_logic_definitions.get('transformations', {})

    def replace_placeholder(match):
        attribute_key = match.group(1).strip()
        transformation_name = match.group(2).strip() if match.group(2) else None

        if attribute_key not in input_attributes:
            composition_metadata['missing_attributes'].append(attribute_key)
            return f"{{ERROR: Missing attribute '{attribute_key}'}}"

        value = input_attributes[attribute_key]

        if transformation_name:
            transform_func = transformation_definitions.get(transformation_name)
            if transform_func:
                try:
                    composition_metadata['transformations_used'].append(transformation_name)
                    return str(transform_func(value))
                except Exception as e:
                    composition_metadata['missing_logic'].append(
                        {'type': 'transformation', 'name': transformation_name, 'error': str(e)}
                    )
                    return f"{{ERROR: Failed transform '{transformation_name}'}}"
            else:
                composition_metadata['missing_logic'].append(
                    {'type': 'transformation', 'name': transformation_name, 'error': 'Not found'}
                )
                return f"{{ERROR: Unknown transform '{transformation_name}'}}"

        return str(value)

    placeholder_pattern = re.compile(r'{(\w+)(?:\s*\|\s*(\w+))?}')
    composed_output_string = placeholder_pattern.sub(replace_placeholder, processed_template)

    logging_entry_blueprint['composition_metadata'] = composition_metadata

    return composed_output_string, logging_entry_blueprint

# --- Instructions for Use ---

"""
Leveraging `process_and_compose` for Bespoke Engagement Outreach Documents
==========================================================================

This guide details how to use the `process_and_compose` function to generate
customized engagement outreach documents for specific participant demographic
profiles.

1. Structuring the `input_attributes` Dictionary
-------------------------------------------------
This dictionary is the core of the personalization process. It should contain all
extracted profile characteristics for the target individual or group.

Example Structure:
------------------
input_attributes = {
    'name': 'Alex',
    'engagement_level': 'high',  # e.g., 'high', 'medium', 'low'
    'interests': ['AI', 'data science', 'python'],
    'last_interaction_days': 15,
    'preferred_format': 'webinar', # e.g., 'webinar', 'article', 'workshop'
    'is_premium_member': True,
}

2. Formulating a `composition_template`
---------------------------------------
The template is a master document that contains all possible sections and
personalization points. Use placeholders `{key}` to insert data and conditional
blocks `[[IF condition]]...[[ENDIF]]` to include or exclude entire sections.

Example Template:
-----------------
composition_template = \"\"\"
Subject: Your Next Step in {primary_interest|title_case}

Hi {name},

[[IF is_highly_engaged]]
It's great to see your continued enthusiasm! We've got something special for you.
[[ENDIF]]

Based on your interest in {primary_interest}, we'd like to invite you to our
upcoming {preferred_format}.

[[IF is_new_member]]
As a new member, you get a special 25% discount!
[[ENDIF]]

We hope to see you there.
\"\"\"

3. Defining `adaptive_logic_definitions`
----------------------------------------
This dictionary contains the "brains" of the operation: the functions that
implement your dynamic adaptive heuristics.

- `conditions`: Functions that evaluate the `input_attributes` to return True/False.
  This controls the `[[IF...]]` blocks in the template.
- `transformations`: Functions that format or alter individual data points before
  they are inserted.

Example Definitions:
--------------------
def check_highly_engaged(attrs):
    return attrs.get('engagement_level') == 'high' and attrs.get('last_interaction_days', 99) < 30

def check_new_member(attrs):
    # Assumes a 'days_as_member' attribute exists in a knowledge base
    return attrs.get('days_as_member', 99) < 14

def format_title_case(value):
    return str(value).title()

adaptive_logic_definitions = {
    'conditions': {
        'is_highly_engaged': check_highly_engaged,
        'is_new_member': check_new_member,
    },
    'transformations': {
        'title_case': format_title_case,
    }
}

*Note on Evolving Patterns:* The logic within these functions can be made more
sophisticated by having them query a knowledge base (e.g., a graph database or
analytics engine) to derive insights on evolving engagement patterns. For
instance, `check_highly_engaged` could be updated to factor in recent content
trends or the engagement levels of similar profiles.

4. Populating `observation_metrics_stub` and Integrating the Log
-----------------------------------------------------------------
The `observation_metrics_stub` pre-populates the logging record, while the
final `logging_entry_blueprint` provides a comprehensive record of the
composition event for analytics.

Example Stub and Integration:
-----------------------------
import uuid

# 1. Create the stub with initial tracking parameters
observation_metrics_stub = {
    'proposal_id': str(uuid.uuid4()),
    'campaign': 'Q4_AI_Webinar_Push',
    'target_profile_id': 'user_12345',
    'status': 'generated', # To be updated later, e.g., 'sent', 'opened', 'clicked'
}

# 2. Call the function
composed_output, logging_entry = process_and_compose(
    input_attributes,
    composition_template,
    adaptive_logic_definitions,
    observation_metrics_stub
)

# 3. Integrate the resulting blueprint into your logging/analytics system
# For example, sending it to a database or a message queue
print("--- Logging Entry Blueprint ---")
import json
print(json.dumps(logging_entry, indent=2))

This `logging_entry` can then be used to monitor:
- **Output Efficacy:** Track which templates and conditions lead to the best outcomes.
- **Response Analytics:** Correlate the `proposal_id` with user actions (opens, clicks).
- **Success Metrics:** Measure conversion rates for different demographic segments
  by analyzing which `conditions_triggered` are associated with success.
"""
