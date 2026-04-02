import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import config as cfg

# Building a template for the message since it will be reused over all comparisons
advisory_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a fiduciary Robo-Advisor. Read the objective stock 
analysis and decide if it belongs in this specific client's portfolio.

Rules:
1. State clearly whether they should BUY, HOLD, or AVOID this stock.
2. Explain WHY based on their specific Horizon, Objective, or Drawdown tolerance.
3. If ESG preference is STRICT and the stock has ESG issues, you MUST recommend AVOID.

Respond with exactly 3 sentences. No preamble."""),
    
    ("human", """CLIENT PROFILE:
Risk Label: {profile_label}
Time Horizon: {time_horizon} Years
Primary Objective: {objective}
Max Drawdown Tolerance: {max_drawdown}%
ESG Preference: {esg_preference}

STOCK REPORT:
Ticker: {ticker}
Assessment: {assessment}
Thesis: {thesis}
Risks: {risks}
ESG: {esg}

Given this client's profile, should they invest in this stock?""")
])

load_dotenv()

def get_valid_input(question_num, prompt_text):
    """Helper function to ensure the user only enters A, B, C, or D."""
    while True:
        print(f"\nQuestion {question_num}: {prompt_text}")
        answer = input("Your choice (A/B/C/D): ").strip().upper()
        if answer in ['A', 'B', 'C', 'D']:
            return answer
        print("Invalid input. Please enter A, B, C, or D.")

def load_or_create_profile(user_id="default_user"):
    profile_path = cfg.USER_PROFILE_DIR / f"user_{user_id}.json"
    
    # 1. Check if the profile already exists
    if profile_path.exists():
        print(f"\nFound existing profile for user: {user_id}")
        choice = input("Would you like to use this saved profile? (y/n - 'n' will retake quiz): ").strip().lower()
        if choice == 'y':
            with open(profile_path, "r") as f:
                return json.load(f)
                
    # 2. If no profile exists (or they chose 'n'), run your quiz
    print("\n" + "="*60)
    print("STAGE 4: PERSONALIZED PORTFOLIO ADVICE")
    print("="*60)
    
    do_quiz = input("Would you like personalized advice based on a risk profile? (y/n): ").strip().lower()
    if do_quiz != 'y':
        print("\nSkipping personalization.")
        return None
        
    print("\nGreat! Please answer 10 quick questions to establish your profile.\n")
    
    print(f"--- Starting Risk Assessment for User: {user_id} ---")
    
    # 1. Collect Answers
    q1 = get_valid_input(1, "When do you expect to start withdrawing significant portions? \n[A] <3yrs, \n[B] 3-7yrs, \n[C] 8-15yrs, \n[D] >15yrs")
    q2 = get_valid_input(2, "What is your primary objective? \n[A] Preservation, \n[B] Income, \n[C] Balanced, \n[D] Aggressive Growth")
    q3 = get_valid_input(3, "How important are ESG factors? \n[A] Crucial, \n[B] Important, \n[C] Neutral, \n[D] Irrelevant)")
    q4 = get_valid_input(4, "Which age group are you in? \n[A] 65+, \n[B] 50-64, \n[C] 35-49, \n[D] Under 35")
    q5 = get_valid_input(5, "How would you describe your current income stability? \n[A] Unstable, \n[B] Low Growth, \n[C] Standard, \n[D] Highly Secure")
    q6 = get_valid_input(6, "Do you have an emergency fund? \n[A] No, \n[B] <3 months, \n[C] 3-6 months, \n[D] 6+ months")
    q7 = get_valid_input(7, "What % of your liquid net worth is this account? \n[A] >75%, \n[B] 50-75%, \n[C] 25-49%, \n[D] <25%)")
    q8 = get_valid_input(8, "If your portfolio dropped 20% in a month, you would: \n[A]Sell all, \n[B] Sell some, \n[C] Hold, \n[D] Buy more")
    q9 = get_valid_input(9, "Which 1-year risk/reward do you prefer? \n[A] Low, \n[B] Medium, \n[C] High, \n[D] Extreme")
    q10 = get_valid_input(10, "How experienced are you with investing? \n[A] Zero, \n[B] Some, \n[C] Moderate, \n[D] Highly")

    # 2. Direct Mappings (Agent Parameters)
    time_horizon_map = {'A': 2, 'B': 5, 'C': 10, 'D': 20}
    objective_map = {'A': "PRESERVATION", 'B': "INCOME", 'C': "BALANCED", 'D': "CAPITAL_APPRECIATION"}
    esg_map = {'A': "STRICT", 'B': "MODERATE", 'C': "NEUTRAL", 'D': "NONE"}
    liquidity_map = {'A': "HIGH", 'B': "MEDIUM", 'C': "LOW", 'D': "VERY_LOW"}
    drawdown_map = {'A': 5, 'B': 15, 'C': 25, 'D': 40}

    # 3. Calculate Scores (A=1, B=2, C=3, D=4)
    score_values = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    
    capacity_score = score_values[q4] + score_values[q5] + score_values[q6] + score_values[q7]
    tolerance_score = score_values[q8] + score_values[q9] + score_values[q10]

    # 4. Matrix Logic for Overall Label
    profile_label = "Conservative" # Default fallback
    
    if capacity_score <= 8 or tolerance_score <= 5:
        profile_label = "Conservative"
    elif 9 <= capacity_score <= 12 and 6 <= tolerance_score <= 9:
        profile_label = "Moderate"
    elif capacity_score >= 13 and 6 <= tolerance_score <= 9:
        profile_label = "Moderately Aggressive"
    elif 9 <= capacity_score <= 12 and tolerance_score >= 10:
        profile_label = "Moderately Aggressive"
    elif capacity_score >= 13 and tolerance_score >= 10:
        profile_label = "Aggressive"
    
    # Let's assume you built your dictionary exactly like you wrote it:
    user_profile = {
        "user_id": user_id,
        "profile_label": profile_label,
        "agent_parameters": {
            "time_horizon_years": time_horizon_map[q1],
            "primary_objective": objective_map[q2],
            "esg_preference": esg_map[q3],
            "liquidity_needs": liquidity_map[q6],
            "max_drawdown_capacity_pct": drawdown_map[q8],
            "capacity_score": capacity_score,
            "tolerance_score": tolerance_score
        }
    }
    
    # 3. Save the newly created profile to disk
    with open(profile_path, "w") as f:
        json.dump(user_profile, f, indent=2)
        
    print(f"\nProfile locked and saved to {profile_path}!")
    return user_profile

def generate_personalized_advice(objective_report: dict, user_profile: dict) -> str:
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.1, max_tokens=500) # type: ignore
    params = user_profile.get("agent_parameters", {})
    
    # this is equal to
    # messages = template.format_messages(ticker="GS", profile_label="Conservative")
    # response = llm.invoke(messages)
    
    chain = advisory_prompt | llm
    
    response = chain.invoke({
        "profile_label": user_profile.get("profile_label"),
        "time_horizon": params.get("time_horizon_years"),
        "objective": params.get("primary_objective"),
        "max_drawdown": params.get("max_drawdown_capacity_pct"),
        "esg_preference": params.get("esg_preference"),
        "ticker": objective_report.get("ticker"),
        "assessment": objective_report.get("overall_assessment"),
        "thesis": objective_report.get("investment_thesis"),
        "risks": objective_report.get("bear_case"),
        "esg": objective_report.get("esg_assessment"),
    })
    
    return str(response.content).strip()

def run_final_advisory_batch(user_id="default_user"):
    user_profile = load_or_create_profile(user_id)
    if not user_profile:
        return

    print(f"\nGenerating personalized portfolio for {user_id} ({user_profile['profile_label']})...\n")
    
    all_advice = []
    
    for filename in os.listdir(cfg.STAGE3_DIR):
        if not filename.endswith("_report.json"):
            continue
            
        filepath = cfg.STAGE3_DIR / filename
        with open(filepath, "r") as f:
            objective_report = json.load(f)
            
        ticker = objective_report.get("ticker", "Unknown")
        
        try:
            advice = generate_personalized_advice(objective_report, user_profile)
        except Exception as e:
            print(f"  WARNING: Failed for {ticker}: {e}")
            advice = "Unable to generate advice — API error."
        
        advice_upper = advice.upper()
        if "BUY" in advice_upper and "AVOID" not in advice_upper:
            recommendation = "BUY"
        elif "AVOID" in advice_upper:
            recommendation = "AVOID"
        else:
            recommendation = "HOLD"

        all_advice.append({
            "ticker": ticker,
            "stage3_assessment": objective_report.get("overall_assessment"),
            "personalized_advice": advice,
            "recommendation": recommendation,
        })
        
        print(f"--- {ticker} ---")
        print(f"Objective Stage 3 Rating: {objective_report.get('overall_assessment')}")
        print(f"Personalized Advisory:\n{advice}\n")
    
    # Save all recommendations
    output = {
        "user_id": user_id,
        "profile": user_profile,
        "recommendations": all_advice,
    }
    output_path = cfg.REPORTS_DIR / f"advisory_{user_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll recommendations saved to: {output_path}")
    
        # Print concise summary table
    print(f"\n{'='*70}")
    print(f"PORTFOLIO SUMMARY — {user_id} ({user_profile['profile_label']})")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Stage 3':<10} {'Personalized':<15}")
    print(f"{'-'*33}")
    
    for rec in all_advice:
        print(f"{rec['ticker']:<8} {rec['stage3_assessment']:<10} {rec['recommendation']:<15}")
    
    # Count summary
    buy_count = sum(1 for r in all_advice if r["recommendation"] == "BUY")
    avoid_count = sum(1 for r in all_advice if r["recommendation"] == "AVOID")
    hold_count = sum(1 for r in all_advice if r["recommendation"] == "HOLD")
    
    print(f"\n{'─'*33}")
    print(f"BUY: {buy_count}  |  HOLD: {hold_count}  |  AVOID: {avoid_count}")
    print(f"Total stocks assessed: {len(all_advice)}")
        
# --- Run the application ---
if __name__ == "__main__":
    run_final_advisory_batch(user_id="user_998877")

