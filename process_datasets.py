import pandas as pd
import config as cfg 

def prepare_regression_datasets(input_csv, output_csv):
    """
    Loads the cleaned feature dataset, calculates the target Percentile Rank,
    and formats the data for a Machine Learning Regressor.
    """
    print(f"Processing {input_csv}...")
    
    # 1. Load the data
    df = pd.read_csv(input_csv)
    
    # 2. Drop rows where we don't have a future return (e.g., the 6 months hasn't passed yet)
    initial_len = len(df)
    df = df.dropna(subset=['Forward_6m_Return'])
    if len(df) < initial_len:
        print(f"  -> Dropped {initial_len - len(df)} rows missing forward returns.")
        
    # 3. Calculate the Target Variable: Percentile Rank per Screening Date
    # pct=True scales the ranks from 0.000 (Worst performing) to 1.000 (Best performing)
    df['Target_Percentile'] = df.groupby('Screening_Date')['Forward_6m_Return'].rank(pct=True)
    
    # 4. Final Cleanup
    # drop the raw return so our model can't cheat by looking at it!
    df = df.drop(columns=['Forward_6m_Return'])
    
    # 5. Save the ML-ready dataset
    df.to_csv(output_csv, index=False)
    print(f"  -> Saved {len(df)} rows to {output_csv}\n")

if __name__ == "__main__":
    # Ensure reports/data directories exist
    cfg.DATA_DIR.mkdir(exist_ok=True)
    
    # Define input and output paths using your config
    train_input = cfg.DATA_DIR / "ML_Training_Data.csv"
    train_output = cfg.DATA_DIR / "ML_Training_Regression.csv"
    
    test_input = cfg.DATA_DIR / "ML_Testing_Data.csv"
    test_output = cfg.DATA_DIR / "ML_Testing_Regression.csv"
    
    print("=== Engineering Machine Learning Targets ===\n")
    prepare_regression_datasets(train_input, train_output)
    prepare_regression_datasets(test_input, test_output)
    
    print("Regression Datasets Ready for Training!")