import unittest
from blockchain_analyzer import query_and_filter_data

class TestBlockchainAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sample_data = [
            {"id": 1, "value": 10, "category": "A", "tags": ["tag1", "tag2"], "nested": {"prop": "X"}},
            {"id": 2, "value": 20, "category": "B", "tags": ["tag2", "tag3"], "nested": {"prop": "Y"}},
            {"id": 3, "value": 30, "category": "A", "tags": ["tag1", "tag3"], "nested": {"prop": "X"}},
            {"id": 4, "value": 40, "category": "C", "tags": ["tag4"], "nested": {"prop": "Z"}},
            {"id": 5, "value": 50, "category": "B", "tags": ["tag1"], "nested": {"prop": "Y"}},
        ]
        self.nested_list_data = [
            {"id": 1, "events": [{"name": "Transfer", "value": 100}, {"name": "Approval", "value": 50}]},
            {"id": 2, "events": [{"name": "Sync", "value": 0}]},
            {"id": 3, "events": [{"name": "Transfer", "value": 200}]},
        ]

    def test_equal_operator(self):
        query = {"mode": "AND", "conditions": [{"field": "category", "operator": "eq", "value": "A"}]}
        result = query_and_filter_data(self.sample_data, query)
        self.assertEqual(len(result["filtered_data"]), 2)
        self.assertEqual(result["filtered_data"][0]["id"], 1)
        self.assertEqual(result["filtered_data"][1]["id"], 3)

    def test_nested_attribute_access(self):
        query = {"mode": "AND", "conditions": [{"field": "nested.prop", "operator": "eq", "value": "Y"}]}
        result = query_and_filter_data(self.sample_data, query)
        self.assertEqual(len(result["filtered_data"]), 2)
        self.assertEqual(result["filtered_data"][0]["id"], 2)
        self.assertEqual(result["filtered_data"][1]["id"], 5)

    def test_nested_list_query(self):
        query = {"mode": "AND", "conditions": [{"field": "events.name", "operator": "eq", "value": "Transfer"}]}
        result = query_and_filter_data(self.nested_list_data, query)
        self.assertEqual(len(result["filtered_data"]), 2)
        self.assertEqual(result["filtered_data"][0]["id"], 1)
        self.assertEqual(result["filtered_data"][1]["id"], 3)

    def test_nested_list_query_with_value(self):
        query = {"mode": "AND", "conditions": [{"field": "events.value", "operator": "gt", "value": 150}]}
        result = query_and_filter_data(self.nested_list_data, query)
        self.assertEqual(len(result["filtered_data"]), 1)
        self.assertEqual(result["filtered_data"][0]["id"], 3)

    def test_greater_than_operator(self):
        query = {"mode": "AND", "conditions": [{"field": "value", "operator": "gt", "value": 30}]}
        result = query_and_filter_data(self.sample_data, query)
        self.assertEqual(len(result["filtered_data"]), 2)
        self.assertEqual(result["filtered_data"][0]["id"], 4)
        self.assertEqual(result["filtered_data"][1]["id"], 5)

    def test_in_operator(self):
        query = {"mode": "AND", "conditions": [{"field": "category", "operator": "in", "value": ["B", "C"]}]}
        result = query_and_filter_data(self.sample_data, query)
        self.assertEqual(len(result["filtered_data"]), 3)

    def test_and_mode(self):
        query = {"mode": "AND", "conditions": [
            {"field": "category", "operator": "eq", "value": "A"},
            {"field": "value", "operator": "lt", "value": 20}
        ]}
        result = query_and_filter_data(self.sample_data, query)
        self.assertEqual(len(result["filtered_data"]), 1)
        self.assertEqual(result["filtered_data"][0]["id"], 1)

    def test_or_mode(self):
        query = {"mode": "OR", "conditions": [
            {"field": "category", "operator": "eq", "value": "C"},
            {"field": "value", "operator": "eq", "value": 10}
        ]}
        result = query_and_filter_data(self.sample_data, query)
        self.assertEqual(len(result["filtered_data"]), 2)

    def test_aggregation_count(self):
        query = {"mode": "AND", "conditions": [{"field": "category", "operator": "eq", "value": "B"}]}
        aggregations = {"count": {"field": None, "type": "count"}}
        result = query_and_filter_data(self.sample_data, query, aggregations)
        self.assertEqual(result["metadata"]["count"], 2)

    def test_aggregation_sum(self):
        query = {"mode": "AND", "conditions": [{"field": "category", "operator": "eq", "value": "A"}]}
        aggregations = {"total_value": {"field": "value", "type": "sum"}}
        result = query_and_filter_data(self.sample_data, query, aggregations)
        self.assertEqual(result["metadata"]["total_value"], 40)

    def test_attribute_extraction(self):
        query = {"mode": "AND", "conditions": [{"field": "category", "operator": "eq", "value": "A"}]}
        extract_attributes = ["id", "nested.prop"]
        result = query_and_filter_data(self.sample_data, query, extract_attributes=extract_attributes)
        self.assertEqual(len(result["filtered_data"]), 2)
        self.assertEqual(list(result["filtered_data"][0].keys()), ["id", "nested.prop"])
        self.assertEqual(result["filtered_data"][0]["nested.prop"], "X")


if __name__ == "__main__":
    unittest.main()
