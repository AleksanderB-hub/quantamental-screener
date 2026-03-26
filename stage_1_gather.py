"""
Stage 1: Data Gathering
Searches the web for recent news, ESG signals, and management 
changes for each candidate stock. Saves raw text corpus as JSON.
This uses claude API for retrieval, the base model used is sonnet 4.6, you can adjust accordingly if needed.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import config as cfg

load_dotenv()

def search_stock(ticker: str, company_name: str) -> dict:
    """
    Runs 4 targeted searches for a single stock.
    Returns a dict with all search results organised by category.
    """
    search_tool = DuckDuckGoSearchResults(max_results=cfg.MAX_RESULTS_PER_SEARCH)

    # Define search queries per category
    queries = {
        "recent_news": f"{company_name} {ticker} stock news 2025",
        "earnings_outlook": f"{company_name} {ticker} earnings analyst forecast",
        "esg": f"{company_name} ESG controversy environmental social governance",
        "management": f"{company_name} CEO executive management changes insider trading",
    }
    
    all_results = {
        "ticker": ticker,
        "company_name": company_name,
        "search_date": datetime.now().isoformat(),
        "categories": {}
    }
    
    for category, query in queries.items():
        print(f"  Searching [{category}]: {query}")
        try:
            raw_results = search_tool.invoke(query)
            all_results["categories"][category] = {
                "query": query,
                "raw_results": raw_results,
            }
        except Exception as e:
            print(f"  WARNING: Search failed for {category}: {e}")
            all_results["categories"][category] = {
                "query": query,
                "raw_results": "",
                "error": str(e)
            }
        
        time.sleep(cfg.DELAY_BETWEEN_SEARCHES)
    
    return all_results

def summarise_with_claude(search_results: dict) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2000, temperature=0)
    
    ticker = search_results["ticker"]
    company = search_results["company_name"]
    
    search_context = ""
    for category, data in search_results["categories"].items():
        search_context += f"\n--- {category.upper()} ---\n"
        search_context += data.get("raw_results", "No results found.")
        search_context += "\n"
    
    messages = [
        SystemMessage(content="""You are a financial research assistant. 
        Given web search results about a stock, extract and organise the key findings.

        Respond ONLY with a valid JSON object. No markdown backticks, no explanation, 
        no text before or after the JSON. Start your response with { and end with }.

        Required keys:
        - news_summary: 2-3 sentence summary of recent news
        - sentiment: POSITIVE, NEUTRAL, or NEGATIVE
        - earnings_outlook: 1-2 sentences on analyst expectations
        - esg_flags: any ESG concerns found, or NONE
        - management_signals: leadership changes or insider trading, or NONE
        - key_risks: 1-2 key risks identified
        - confidence: HIGH, MEDIUM, or LOW"""),
        HumanMessage(content=f"Research results for {company} ({ticker}):\n{search_context}"),
    ]
    
    response = llm.invoke(messages)
    
    # Clean and parse the response
    raw = response.content.strip()
    # Strip markdown backticks if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    try:
        signals = json.loads(raw)
    except json.JSONDecodeError:
        signals = {"raw_response": response.content, "parse_error": True}
    
    return signals

def process_single_stock(ticker: str, company_name: str) -> dict:
    """
    Full Stage 1 pipeline for one stock:
    1. Search the web
    2. Send results to Claude for structured extraction
    3. Save everything to JSON
    """
    print(f"\n{'='*60}")
    print(f"Processing: {company_name} ({ticker})")
    print(f"{'='*60}")
    
    # Step 1: Search
    search_results = search_stock(ticker, company_name)
    
    # Step 2: Claude extracts structured signals
    print(f"  Sending to Claude for analysis...")
    signals = summarise_with_claude(search_results)
    
    # Step 3: Combine everything
    output = {
        "ticker": ticker,
        "company_name": company_name,
        "processed_date": datetime.now().isoformat(),
        "search_results": search_results["categories"],
        "signals": signals,
    }
    
    # Step 4: Save to JSON
    output_path = cfg.STAGE1_DIR / f"{ticker}_research.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"  Saved to: {output_path}")
    print(f"  Sentiment: {signals.get('sentiment', 'UNKNOWN')}")
    print(f"  Confidence: {signals.get('confidence', 'UNKNOWN')}")
    
    return output

if __name__ == "__main__":
    # Test with one stock from your candidates
    result = process_single_stock("EQT", "EQT Corporation")
    
    print(f"\n{'='*60}")
    print("EXTRACTED SIGNALS:")
    print(f"{'='*60}")
    print(json.dumps(result["signals"], indent=2))