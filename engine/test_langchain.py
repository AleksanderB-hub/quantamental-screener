import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()  # loads ANTHROPIC_API_KEY from your .env

# llm = ChatAnthropic(model="claude-sonnet-4-6")
# response = llm.invoke("Say hello in one sentence.")
# print(response.content)

search = DuckDuckGoSearchResults(num_results=5)
results = search.invoke("Goldman Sachs recent news march 2026")
print(results)


