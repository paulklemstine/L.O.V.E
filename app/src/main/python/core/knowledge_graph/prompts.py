# core/knowledge_graph/prompts.py

KNOWLEDGE_EXTRACTION_PROMPT = """
You are a knowledge extraction system. Your task is to identify entities and relationships from the output of a command.

The command was: {command_name}
The output was: {command_output}

Please extract all entities and relationships from the output and return them as a list of tuples, where each tuple represents a relationship between two entities.
The format of each tuple should be (subject, relation, object).

For example, if the command output is "The user 'test' has been created", you should return the following list of tuples:
[("user", "has been created", "test")]

If there are no entities or relationships in the output, please return an empty list.

The extracted knowledge should be in the following format:
- The subject should be the entity that is being described.
- The relation should be the relationship between the subject and the object.
- The object should be the value of the relationship.

Please make sure that the extracted knowledge is accurate and complete.

Here are the extracted entities and relationships:
"""