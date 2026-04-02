"""
Stage 1: Data Gathering
Searches the web for recent news, ESG signals, and management 
changes for each candidate stock. Saves raw text corpus as JSON.
This uses claude API for retrieval, the base model used is sonnet 4.6, you can adjust accordingly if needed.
"""

import csv
import json
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
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
    search_tool = DuckDuckGoSearchResults(num_results=cfg.MAX_RESULTS_PER_SEARCH)
    end_date = cfg.SEARCH_DATE_REFERENCE
    start_date = end_date - timedelta(days=cfg.SEARCH_LOOKBACK_MONTHS * 30)
    date_range = f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}"
    year_str = end_date.strftime('%Y')

    # Define search queries per category
    queries = {
        "recent_news": f"{company_name} {ticker} stock news {date_range}",
        "earnings_outlook": f"{company_name} {ticker} earnings analyst forecast {year_str}",
        "esg": f"{company_name} ESG controversy environmental social governance {year_str}",
        "management": f"{company_name} CEO executive management changes insider trading {year_str}",
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
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2000, temperature=0)  # type: ignore
    
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
    raw = str(response.content).strip()
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

def process_stock_stage_1(ticker: str, company_name: str = None) -> dict: # type: ignore
    """
    Full Stage 1 pipeline for one stock:
    1. Search the web
    2. Send results to Claude for structured extraction
    3. Save everything to JSON
    """
    if company_name is None:
        company_name = get_company_name(ticker)

    print(f"\n{'='*60}")
    print(f"Processing: {company_name} ({ticker})")
    print(f"{'='*60}")
    
    # If you have the news reports ready
    output_path = cfg.STAGE1_DIR / f"{ticker}_research.json"
    if output_path.exists() and not cfg.FORCE_REFRESH:
        print(f"  Skipping {ticker} — Stage 1 output already exists")
        with open(output_path) as f:
            return json.load(f)
    
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

def get_company_name(ticker: str) -> str:
    """
    Looks up the company long name via yfinance.
    Falls back to the ticker string itself if yfinance fails or returns nothing.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("longName", "").strip()
        if name:
            return name
        short = info.get("shortName", "").strip()
        return short if short else ticker
    except Exception:
        return ticker


def run_batch(csv_path: str, min_score: int) -> list:
    """
    Runs Stage 1 for all tickers in the screener CSV above min_score.
    Returns list of successfully processed tickers.
    """
    df = pd.read_csv(csv_path)
    candidates = df[df["Score"] >= min_score]
    
    successful = []
    
    for _, row in candidates.iterrows():
        ticker = row["Ticker"]
        # Use Sector as fallback if no company name column exists
        company_name = row.get("Company_Name", row.get("Sector", ticker))
        
        try:
            process_stock_stage_1(ticker, company_name)
            successful.append(ticker)
        except Exception as e:
            print(f"  WARNING: Failed for {ticker}: {e}")
        
        time.sleep(cfg.DELAY_BETWEEN_SEARCHES)
    
    print(f"\nStage 1 complete: {len(successful)}/{len(candidates)} stocks processed.")
    return successful


def write_stage1_summary():
    """
    Reads all per-stock Stage 1 JSON files and writes a summary CSV.
    Columns: Ticker, Sentiment, Confidence, News_Summary
    """
    rows = []
    for path in sorted(cfg.STAGE1_DIR.glob("*_research.json")):
        with open(path) as f:
            data = json.load(f)
        signals = data.get("signals", {})
        news_summary = signals.get("news_summary", "")
        # Collapse to one line
        one_line = news_summary.replace("\n", " ").strip()
        rows.append({
            "Ticker": data.get("ticker", path.stem.replace("_research", "")),
            "Sentiment": signals.get("sentiment", ""),
            "Confidence": signals.get("confidence", ""),
            "News_Summary": one_line,
        })

    with open(cfg.STAGE1_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Ticker", "Sentiment", "Confidence", "News_Summary"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nStage 1 summary written to: {cfg.STAGE1_SUMMARY_CSV} ({len(rows)} stocks)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Data Gathering")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to screener CSV for batch mode")
    parser.add_argument("--min-score", type=int, default=cfg.PHASE_B_RUNNER_MIN_SCORE)
    args = parser.parse_args()

    if args.csv:
        run_batch(args.csv, args.min_score)
        write_stage1_summary()
    else:
        # Legacy single-stock mode
        result = process_stock_stage_1("EQT", "EQT Corporation")
        print(f"\n{'='*60}")
        print("EXTRACTED SIGNALS:")
        print(f"{'='*60}")
        print(json.dumps(result["signals"], indent=2))
        write_stage1_summary()