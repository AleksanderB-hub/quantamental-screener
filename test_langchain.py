import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()  # loads ANTHROPIC_API_KEY from your .env

# llm = ChatAnthropic(model="claude-sonnet-4-6")
# response = llm.invoke("Say hello in one sentence.")
# print(response.content)

search = DuckDuckGoSearchResults(max_results=5)
results = search.invoke("Goldman Sachs recent news march 2026")
print(results)


Alright wiht that updates in place, I would liek for you to provde a detiled overview of the Stage 3 script with a focus on RAG and Summary components. 



I need to understand how they work in principle, how and why we configured them in this way, and the important concepts that are necessary to understand to deploy them. 