"""
Stage 2: Content Processing (Local LLM)
Takes Stage 1 raw search results and classifies each snippet
using the local Qwen model via vLLM.
"""
import csv
import json
from pathlib import Path
import httpx
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
import sys, os
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# Connect to local vLLM (same interface as OpenAI thanks to LangChain)

if cfg.LOCAL_MODEL:
    local_llm = ChatOpenAI(
        base_url=cfg.LOCAL_MODEL_URL,
        model=cfg.LOCAL_MODEL_NAME,
        api_key="not-needed",
        temperature=0,
        max_tokens=300,  # type: ignore
    )
else:
    local_llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001", # type: ignore
        max_tokens=300, # type: ignore
        temperature=0,
    )
    

def classify_snippet(snippet: str, ticker: str) -> dict:
    """
    Sends a single search result snippet to the local Qwen model
    for classification.
    """
    messages = [
        SystemMessage(content=f"""You are a financial text classifier analysing news about {ticker}.
        Given a search result snippet, respond ONLY with a valid JSON object. No other text.

        Classification guidelines:
        - Insider selling, regulatory investigations, and lawsuits are NEGATIVE
        - Routine corporate actions (dividends, buybacks, stock splits) are NEUTRAL
        - Analyst upgrades, earnings beats, and revenue growth are POSITIVE
        - If the snippet is not clearly about {ticker}, relevance is LOW

        Required JSON keys:
        - sentiment: POSITIVE, NEUTRAL, or NEGATIVE
        - category: one of EARNINGS, REGULATORY, MANAGEMENT, ESG, PRODUCT, MACRO, OTHER
        - relevance: HIGH, MEDIUM, or LOW
        - summary: one sentence summarizing the key point"""),
        
        HumanMessage(content=f"Classify this snippet:\n\n{snippet}"),
    ]
    
    try:
        response = local_llm.invoke(messages)
        raw = str(response.content).strip()
        
        # Clean markdown backticks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        return {
            "sentiment": "UNKNOWN",
            "category": "OTHER",
            "relevance": "LOW",
            "summary": "Failed to classify",
            "error": str(e)
        }


def process_stock_stage_2(ticker: str) -> dict | None:
    """
    Loads Stage 1 output for a stock, classifies each search result
    snippet through the local LLM, and saves structured signals.
    """
    # If you have the processed reports ready
    output_path = cfg.STAGE2_DIR / f"{ticker}_processed.json"
    if output_path.exists() and not cfg.FORCE_REFRESH:
        print(f"  Skipping {ticker} — Stage 2 output already exists")
        with open(output_path) as f:
            return json.load(f)
        
    # Load Stage 1 output
    stage1_path = cfg.STAGE1_DIR / f"{ticker}_research.json"
    if not stage1_path.exists():
        print(f"  ERROR: No Stage 1 data for {ticker}")
        return None
    
    with open(stage1_path) as f:
        stage1_data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Stage 2 Processing: {ticker}")
    print(f"{'='*60}")
    
    # Process each category's search results
    all_signals = []
    
    for category, data in stage1_data["search_results"].items():
        raw_results = data.get("raw_results", "")
        
        if not raw_results or raw_results == "":
            continue
        
        # DuckDuckGo returns results as a string of snippets
        # Split into individual results by looking for common patterns
        snippets = split_search_results(raw_results)
        
        print(f"\n  [{category.upper()}] — {len(snippets)} snippets to classify")
        
        for i, snippet in enumerate(snippets):
            if len(snippet.strip()) < 30:  # Skip very short/empty snippets
                continue
                
            print(f"    Classifying snippet {i+1}/{len(snippets)}...", end=" ")
            classification = classify_snippet(snippet, ticker)
            
            signal = {
                "source_category": category,
                "original_snippet": snippet,
                "classification": classification,
            }
            all_signals.append(signal)
            
            sentiment = classification.get("sentiment", "?")
            relevance = classification.get("relevance", "?")
            print(f"Sentiment={sentiment}, Relevance={relevance}")
    
    # Build Stage 2 output
    output = {
        "ticker": ticker,
        "company_name": stage1_data.get("company_name", ticker),
        "stage1_claude_signals": stage1_data.get("signals", {}),
        "stage2_local_signals": all_signals,
        "signal_summary": summarise_signals(all_signals),
    }
    
    # Save
    output_path = cfg.STAGE2_DIR / f"{ticker}_processed.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Saved to: {output_path}")
    print(f"  Total signals: {len(all_signals)}")
    print(f"  Signal summary: {output['signal_summary']}")
    
    return output


def split_search_results(raw_results: str) -> list:
    """
    Splits DuckDuckGo's raw result string into individual snippets.
    DuckDuckGo returns results as: [snippet: ..., title: ..., link: ...], ...
    """
    snippets = []
    
    # DuckDuckGoSearchResults returns a string representation of a list
    # Try to extract individual result blocks
    if "snippet:" in raw_results:
        # Split on the pattern that separates results
        parts = raw_results.split("snippet: ")
        for part in parts[1:]:  # skip the first empty part
            # Extract just the snippet text (before the next field)
            snippet_text = part.split(", title:")[0] if ", title:" in part else part
            snippet_text = snippet_text.strip().strip(",").strip()
            if snippet_text:
                snippets.append(snippet_text)
    else:
        # Fallback: treat the whole thing as one snippet
        snippets = [raw_results]
    
    return snippets


def summarise_signals(signals: list) -> dict:
    """
    Aggregates the individual classifications into a summary.
    No LLM needed — just counting.
    """
    if not signals:
        return {"total": 0}
    
    sentiments = [s["classification"].get("sentiment", "UNKNOWN") for s in signals]
    relevances = [s["classification"].get("relevance", "LOW") for s in signals]
    categories = [s["classification"].get("category", "OTHER") for s in signals]
    
    return {
        "total_signals": len(signals),
        "sentiment_breakdown": {
            "positive": sentiments.count("POSITIVE"),
            "neutral": sentiments.count("NEUTRAL"),
            "negative": sentiments.count("NEGATIVE"),
        },
        "high_relevance_count": relevances.count("HIGH"),
        "category_breakdown": {cat: categories.count(cat) for cat in set(categories)},
    }


def run_batch(tickers: list) -> list:
    """
    Runs process_stock_stage_2() for each ticker in the list.
    Skips tickers whose Stage 2 JSON already exists.
    Catches vLLM connection errors and logs them without stopping the batch.
    Returns list of tickers that were successfully processed or already existed.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2 BATCH — {len(tickers)} tickers")
    print(f"{'='*60}")

    successful = []
    for ticker in tickers:
        output_path = cfg.STAGE2_DIR / f"{ticker}_processed.json"
        if output_path.exists():
            print(f"  [SKIP] {ticker} — already processed")
            successful.append(ticker)
            continue

        print(f"  [RUN ] {ticker}")
        try:
            result = process_stock_stage_2(ticker)
            if result is not None:
                successful.append(ticker)
            else:
                print(f"  [ERROR] {ticker} — process_stock_stage_2() returned None")
        except (ConnectionError, httpx.ConnectError) as e:
            print(f"  [ERROR] {ticker} — vLLM not reachable: {e}")
            print(f"           Stage 2 skipped for {ticker}. Re-run when vLLM is available.")
        except Exception as e:
            print(f"  [ERROR] {ticker} — {e}")

    return successful


def write_stage2_summary():
    """
    Reads all per-stock Stage 2 JSON files and writes a summary CSV.
    Columns: Ticker, Positive_Count, Neutral_Count, Negative_Count,
             High_Relevance_Count, Top_Category
    """
    rows = []
    for path in sorted(cfg.STAGE2_DIR.glob("*_processed.json")):
        with open(path) as f:
            data = json.load(f)
        summary = data.get("signal_summary", {})
        breakdown = summary.get("sentiment_breakdown", {})
        category_breakdown = summary.get("category_breakdown", {})
        top_category = max(category_breakdown, key=category_breakdown.get) if category_breakdown else ""
        rows.append({
            "Ticker": data.get("ticker", path.stem.replace("_processed", "")),
            "Positive_Count": breakdown.get("positive", 0),
            "Neutral_Count": breakdown.get("neutral", 0),
            "Negative_Count": breakdown.get("negative", 0),
            "High_Relevance_Count": summary.get("high_relevance_count", 0),
            "Top_Category": top_category,
        })

    with open(cfg.STAGE2_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Ticker", "Positive_Count", "Neutral_Count", "Negative_Count",
            "High_Relevance_Count", "Top_Category",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nStage 2 summary written to: {cfg.STAGE2_SUMMARY_CSV} ({len(rows)} stocks)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Content Processing")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Space-separated list of tickers for batch mode")
    args = parser.parse_args()

    if args.tickers:
        run_batch(args.tickers)
        write_stage2_summary()
    else:
        # Legacy single-stock mode
        result = process_stock_stage_2("EQT")
        if result:
            print(f"\n{'='*60}")
            print("SIGNAL SUMMARY:")
            print(f"{'='*60}")
            print(json.dumps(result["signal_summary"], indent=2))
        write_stage2_summary()