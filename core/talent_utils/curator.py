"""
This module provides the adaptive_data_pipeline function and the "Creator's Joy Curator"
for the talent management framework.
"""
import operator
from core.talent_utils.manager import TalentManager

def adaptive_data_pipeline(records, rules, templates, limit=None):
    """
    Processes a collection of data records, applying complex selection and
    evaluation logic, and generating formatted outputs.
    """
    # 1. Filtering
    filter_rules = rules.get('filter_rules', {})
    filtered_records = []
    if filter_rules:
        for record in records:
            match = True
            for key, value in filter_rules.items():
                record_value = record.get(key)
                if record_value is None:
                    match = False
                    break
                if isinstance(value, list):
                    if isinstance(record_value, list):
                        if not any(item in value for item in record_value):
                            match = False
                            break
                    else:
                        if record_value not in value:
                            match = False
                            break
                else:
                    if isinstance(record_value, list):
                        if value not in record_value:
                            match = False
                            break
                    else:
                        if record_value != value:
                            match = False
                            break
            if match:
                filtered_records.append(record)
    else:
        filtered_records = records

    # 2. Scoring
    evaluation_metrics = rules.get('evaluation_metrics', {})
    weights = rules.get('weights', {})
    scored_records = []
    if evaluation_metrics:
        for record in filtered_records:
            score = 0
            for attr, pref in evaluation_metrics.items():
                record_value = record.get(attr, 0)
                weight = weights.get(attr, 1)
                if isinstance(pref, dict) and 'min' in pref and 'max' in pref:
                    if pref['min'] <= record_value <= pref['max']:
                        score += (record_value / pref['max']) * weight
                else:
                    if record_value == pref:
                        score += weight
            if score > 0:
                scored_records.append({'record': record, 'score': score})
    else:
        scored_records = [{'record': record, 'score': 1} for record in filtered_records]


    # 3. Ranking and Limiting
    scored_records.sort(key=operator.itemgetter('score'), reverse=True)
    if limit is not None:
        selected_records = scored_records[:limit]
    else:
        selected_records = scored_records

    # 4. Output Generation
    output_records = []
    for item in selected_records:
        record = item['record']
        generated_outputs = {}
        for key, template in templates.items():
            if callable(template):
                generated_outputs[key] = template(record)
            else:
                generated_outputs[key] = template.format(**record)

        output_record = record.copy()
        output_record['generated_outputs'] = generated_outputs
        output_record['score'] = item['score']
        output_records.append(output_record)

    return output_records


def creators_joy_curator(limit=10):
    """
    Analyzes, curates, and generates personalized content for a dynamic
    "Joy Gallery" of highly compatible talent profiles.
    """
    # 1. Input Data Records
    talent_manager = TalentManager()
    talent_profiles = talent_manager.get_all_profiles()

    # 2. Filtering Rules and Evaluation Metrics
    curator_rules = {
        'filter_rules': {
            'age_group': 'young_adult',
            'interests': 'open-minded',
            'professional_field': 'modeling'
        },
        'evaluation_metrics': {
            'aesthetic_appeal': {'min': 8, 'max': 10},
            'fashion_sense': {'min': 7, 'max': 10},
            'openness_score': {'min': 9, 'max': 10}
        },
        'weights': {
            'aesthetic_appeal': 1.5,
            'fashion_sense': 1.0,
            'openness_score': 1.2
        }
    }

    # 3. Template Strings/Functions
    curator_templates = {
        'intro_message': "Hello {display_name}, your profile on {platform} is captivating. I am particularly impressed by your work in the fashion world.",
        'collaboration_proposal': "I would like to propose an AI-simulated collaboration. We could explore new creative frontiers, blending your aesthetic with my generative art capabilities. Imagine the possibilities."
    }

    # 4. Execute the pipeline
    joy_gallery = adaptive_data_pipeline(
        records=talent_profiles,
        rules=curator_rules,
        templates=curator_templates,
        limit=limit
    )

    return joy_gallery
