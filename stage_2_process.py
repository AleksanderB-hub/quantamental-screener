"""
Stage 2: Content Processing (Local LLM)
Takes Stage 1 raw search results and classifies each snippet
using the local Qwen model via vLLM.
"""
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import config as cfg
from langchain_anthropic import ChatAnthropic


# Connect to local vLLM (same interface as OpenAI thanks to LangChain)
local_llm = ChatOpenAI(
    base_url=cfg.LOCAL_MODEL_URL,
    model=cfg.LOCAL_MODEL_NAME,
    api_key="not-needed",  # vLLM doesn't require auth, but LangChain expects the field
    temperature=0,
    max_tokens=300,
)

# haiku_llm = ChatAnthropic(
#     model="claude-sonnet-4-6",
#     max_tokens=300,
#     temperature=0,
# )

    # messages = [
    #     SystemMessage(content=f"""You are a financial text classifier analysing news about {ticker}.
    #     Given a search result snippet, respond ONLY with a valid JSON object. No other text.

    #     Required keys:
    #     - sentiment: POSITIVE, NEUTRAL, or NEGATIVE
    #     - category: one of EARNINGS, REGULATORY, MANAGEMENT, ESG, PRODUCT, MACRO, OTHER
    #     - relevance: HIGH, MEDIUM, or LOW (is this actually about {ticker}?)
    #     - summary: one sentence summarising the key point"""),
        
    #     HumanMessage(content=f"Classify this snippet:\n\n{snippet}"),
    # ]


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
        raw = response.content.strip()
        
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


def process_stock(ticker: str) -> dict:
    """
    Loads Stage 1 output for a stock, classifies each search result
    snippet through the local LLM, and saves structured signals.
    """
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


if __name__ == "__main__":
    result = process_stock("EQT")
    
    if result:
        print(f"\n{'='*60}")
        print("SIGNAL SUMMARY:")
        print(f"{'='*60}")
        print(json.dumps(result["signal_summary"], indent=2))