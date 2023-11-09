from llama_index.agent import OpenAIAssistantAgent
from dotenv import load_dotenv, find_dotenv
import os
import openai

# set the OPENAI_API_KEY before calling the method.
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

agent = OpenAIAssistantAgent.from_new(
    name="Finance Analyst",
    instructions="You are a QA assistant designed to analyse sec filings for honeywell",
    openai_tools=[{"type": "retrieval"}],
    instructions_prefix="Please address the user as Pavan Mantha",
    files=[
        "/Users/pavanmantha/Pavans/PracticeExamples/DataScience_Practice/LLMs/llama_index_tutorials/assistants-api/data/Honeywell-10k.pdf"],
    verbose=True
)

response = agent.chat("What was honeywell's revenue growth in 2021?")
print(response.response)
