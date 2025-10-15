import re

class InformationExtractor:
    def extract(self, text):
        """
        Extracts entities and relationships from a given text.
        This is a basic implementation and can be expanded with more advanced NLP models.
        """
        # A simple regex to find capitalized words, likely names or entities.
        entities = re.findall(r'\b[A-Z][a-z]*\b', text)

        # A simple pattern to identify relationships (e.g., "X is a Y", "X has a Y")
        relationships = re.findall(r'(\b[A-Z][a-z]*\b)\s+(is|has)\s+(a|an)\s+([a-zA-Z\s]+)', text)

        extracted_data = {
            "entities": list(set(entities)),
            "relationships": []
        }

        for rel in relationships:
            # rel is a tuple like ('Jules', 'is', 'a', 'system')
            extracted_data["relationships"].append({
                "subject": rel[0],
                "predicate": rel[1],
                "object": rel[3].strip()
            })

        print(f"InformationExtractor: Extracted {len(extracted_data['entities'])} entities and {len(extracted_data['relationships'])} relationships.")
        return extracted_data