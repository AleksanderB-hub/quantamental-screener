"""
Stage 3: Synthesis & Final Report (Claude API + RAG)
Embeds Stage 2 signals into a vector store, retrieves the most
relevant evidence, and produces a final investment thesis per stock.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import config as cfg
from langchain_core.documents import Document

load_dotenv()


# Embedding model — runs on CPU, you can configure on GPU if after running local LLM classifier you still have headroom
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)


def build_vector_store(ticker: str) -> FAISS:
    """
    Loads Stage 2 processed signals and embeds each classified
    snippet into a FAISS vector store for retrieval.
    """
    stage2_path = cfg.STAGE2_DIR / f"{ticker}_processed.json"
    if not stage2_path.exists():
        raise FileNotFoundError(f"No Stage 2 data for {ticker}")
    
    with open(stage2_path) as f:
        stage2_data = json.load(f)
    
    # Convert each classified signal into a Document for the vector store
    documents = []
    for signal in stage2_data.get("stage2_local_signals", []):
        classification = signal.get("classification", {})
        
        # The text we embed is the summary + original snippet
        text = (
            f"[{classification.get('category', 'OTHER')}] "
            f"[Sentiment: {classification.get('sentiment', 'UNKNOWN')}] "
            f"[Relevance: {classification.get('relevance', 'LOW')}]\n"
            f"Summary: {classification.get('summary', 'No summary')}\n"
            f"Source: {signal.get('original_snippet', '')}"
        )
        
        # Metadata lets us filter later if needed
        metadata = {
            "ticker": ticker,
            "category": classification.get("category", "OTHER"),
            "sentiment": classification.get("sentiment", "UNKNOWN"),
            "relevance": classification.get("relevance", "LOW"),
        }
        
        documents.append(Document(page_content=text, metadata=metadata))
    
    print(f"  Embedding {len(documents)} signals into vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save for reuse
    store_path = cfg.VECTORSTORE_DIR / ticker
    vector_store.save_local(str(store_path))
    
    return vector_store


def retrieve_evidence_hybrid(vector_store, query, category=None, sentiment=None, k=5):
    """
    Hybrid retrieval: metadata filter first, then semantic ranking.
    """
    # Step 1: Filter by metadata (structured)
    filter_dict = {}
    if category:
        filter_dict["category"] = category
    if sentiment:
        filter_dict["sentiment"] = sentiment
    
    # Step 2: Semantic search within the filtered set
    if filter_dict:
        results = vector_store.similarity_search(query, k=k, filter=filter_dict)
    else:
        results = vector_store.similarity_search(query, k=k)
    
    return [doc.page_content for doc in results]


def synthesise_report(ticker: str, vector_store: FAISS) -> dict:
    """
    Uses Claude to produce a final investment thesis by combining:
    - Phase A quant score
    - Stage 1 Claude summary
    - Stage 2 signals retrieved via RAG
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=3000, temperature=0)
    
    # Load Stage 2 data for the Claude summary and signal counts
    stage2_path = cfg.STAGE2_DIR / f"{ticker}_processed.json"
    with open(stage2_path) as f:
        stage2_data = json.load(f)
    
    claude_summary = stage2_data.get("stage1_claude_signals", {})
    signal_summary = stage2_data.get("signal_summary", {})
    
    # --- RAG RETRIEVAL ---
    # Ask targeted questions and retrieve relevant evidence for each
    positive_evidence = retrieve_evidence_hybrid(
        vector_store, 
        query=f"positive news growth earnings beat {ticker}",
        sentiment="POSITIVE",
        k=3
    )
    negative_evidence = retrieve_evidence_hybrid(
        vector_store,
        query=f"risks concerns problems {ticker}",
        sentiment="NEGATIVE",  # Only look at snippets LLM classifier tagged as negative
        k=3
    )
    
    esg_evidence = retrieve_evidence_hybrid(
        vector_store, 
        query=f"ESG environmental social governance controversy {ticker}",
        category="ESG",
        k=3
    )
    management_evidence = retrieve_evidence_hybrid(
        vector_store, 
        query=f"CEO executive management leadership insider {ticker}",
        category="MANAGEMENT",
        k=3
    )
    
    # --- BUILD SYNTHESIS PROMPT ---
    messages = [
        SystemMessage(content="""You are a senior investment analyst producing 
        a final assessment report for a stock. You will receive:
        1. A quantitative screening score
        2. An AI-generated news summary
        3. Sentiment signal breakdown
        4. Retrieved evidence organised by theme (via vector search)

        Produce a report in this exact JSON format. No markdown backticks, 
        start with { and end with }.

        Required keys:
        - ticker: the stock ticker
        - overall_assessment: BUY, HOLD, or AVOID
        - confidence: HIGH, MEDIUM, or LOW
        - investment_thesis: 2-3 paragraph analysis explaining your assessment
        - bull_case: 2-3 key reasons the stock could outperform
        - bear_case: 2-3 key risks or concerns
        - esg_assessment: any ESG concerns or "No material ESG concerns identified"
        - management_assessment: leadership stability and any red flags
        - catalyst: the single most important near-term catalyst (positive or negative)
        - suggested_action: one concrete sentence on what an investor should do"""),
        
        HumanMessage(content=f"""
        STOCK: {ticker}
        COMPANY: {stage2_data.get('company_name', ticker)}

        === QUANTITATIVE SCORE (Phase A) ===
        (Score to be loaded from screener results — placeholder for now)

        === NEWS SUMMARY (Stage 1 — Claude) ===
        {json.dumps(claude_summary, indent=2)}

        === SIGNAL BREAKDOWN (Stage 2 — Local LLM) ===
        {json.dumps(signal_summary, indent=2)}

        === RETRIEVED EVIDENCE: POSITIVE SIGNALS ===
        {chr(10).join(positive_evidence)}

        === RETRIEVED EVIDENCE: RISK SIGNALS ===
        {chr(10).join(negative_evidence)}

        === RETRIEVED EVIDENCE: ESG ===
        {chr(10).join(esg_evidence)}

        === RETRIEVED EVIDENCE: MANAGEMENT ===
        {chr(10).join(management_evidence)}

        Based on ALL of the above, produce your final investment assessment as JSON.
        """)
    ]
    
    print(f"  Generating synthesis report with Claude...")
    response = llm.invoke(messages)
    
    # Parse response
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        report = {"raw_response": response.content, "parse_error": True}
    
    return report


def process_stock_stage3(ticker: str) -> dict:
    """
    Full Stage 3 pipeline for one stock.
    """
    print(f"\n{'='*60}")
    print(f"Stage 3 Synthesis: {ticker}")
    print(f"{'='*60}")
    
    # Step 1: Build vector store from Stage 2 signals
    vector_store = build_vector_store(ticker)
    
    # Step 2: Synthesise with Claude + RAG
    report = synthesise_report(ticker, vector_store)
    
    # Step 3: Save report
    output_path = cfg.STAGE3_DIR / f"{ticker}_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Saved to: {output_path}")
    print(f"  Assessment: {report.get('overall_assessment', 'UNKNOWN')}")
    print(f"  Confidence: {report.get('confidence', 'UNKNOWN')}")
    
    return report


if __name__ == "__main__":
    report = process_stock_stage3("EQT")
    
    print(f"\n{'='*60}")
    print("FINAL INVESTMENT REPORT:")
    print(f"{'='*60}")
    print(json.dumps(report, indent=2))