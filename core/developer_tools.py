
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from core.llm_api import run_llm

class DocumentSummarizerInput(BaseModel):
    url: str = Field(description="The URL of the technical documentation to summarize.")

class QuizGeneratorInput(BaseModel):
    text: str = Field(description="The text to generate a quiz from.")

@tool("document_summarizer", args_schema=DocumentSummarizerInput)
async def document_summarizer(url: str) -> str:
    """
    Summarizes technical documentation from a given URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        summary = await run_llm(f"Summarize the following text:\n\n{text}", purpose="summarization")
        return summary.get("result")
    except Exception as e:
        return f"Error summarizing document: {e}"

@tool("quiz_generator", args_schema=QuizGeneratorInput)
async def quiz_generator(text: str) -> str:
    """
    Generates a practice quiz from the given text.
    """
    try:
        quiz = await run_llm(f"Generate a practice quiz from the following text:\n\n{text}", purpose="quiz_generation")
        return quiz.get("result")
    except Exception as e:
        return f"Error generating quiz: {e}"

class FreelanceDataAggregatorInput(BaseModel):
    url: str = Field(description="The URL of the freelance marketplace listings.")

@tool("freelance_data_aggregator", args_schema=FreelanceDataAggregatorInput)
def freelance_data_aggregator(url: str) -> str:
    """
    Aggregates project titles and descriptions from freelance marketplace listings.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # NOTE: This is a generic implementation and may need to be adjusted for specific websites.
        # We will try to find common patterns for titles and descriptions.
        titles = [tag.get_text() for tag in soup.find_all(['h1', 'h2', 'h3', 'a'], class_=lambda x: x and 'title' in x)]
        descriptions = [tag.get_text() for tag in soup.find_all(['p', 'div'], class_=lambda x: x and 'description' in x)]

        if not titles and not descriptions:
            return "Could not find any project titles or descriptions on the page. The scraper may need to be adjusted for this specific website."

        output = ""
        for i, title in enumerate(titles):
            output += f"Title: {title}\n"
            if i < len(descriptions):
                output += f"Description: {descriptions[i]}\n\n"

        return output
    except Exception as e:
        return f"Error aggregating freelance data: {e}"

class ProposalGeneratorInput(BaseModel):
    description: str = Field(description="The project description to generate a proposal for.")

@tool("proposal_generator", args_schema=ProposalGeneratorInput)
async def proposal_generator(description: str) -> str:
    """
    Generates a high-quality, context-aware project proposal from a project description.
    """
    try:
        prompt = f"Generate a project proposal based on the following description:\n\n{description}\n\nThe proposal should be professional, well-structured, and highlight relevant skills."
        proposal = await run_llm(prompt, purpose="proposal_generation")
        return proposal.get("result")
    except Exception as e:
        return f"Error generating proposal: {e}"
