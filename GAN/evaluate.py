import pandas as pd
from data_prep import load_and_preprocess_data

def evaluate_realism(real_data_path, synthetic_data_path):
    """
    Evaluate the realism of synthetic data by comparing distributions.
    """
    real_df = load_and_preprocess_data(real_data_path)
    synthetic_df = pd.read_json(synthetic_data_path, lines=True)
    
    # Category distribution
    real_cat = real_df['Category'].value_counts(normalize=True)
    syn_cat = synthetic_df['Category'].value_counts(normalize=True)
    
    print("Real category distribution (top 5):")
    print(real_cat.head())
    print("\nSynthetic category distribution (top 5):")
    print(syn_cat.head())
    
    # Average text length
    real_len = real_df['Resume_str'].str.len().mean()
    syn_len = synthetic_df['Resume_str'].str.len().mean()
    
    print(f"\nReal average text length: {real_len:.2f}")
    print(f"Synthetic average text length: {syn_len:.2f}")
    
    # Perhaps add more metrics, like unique words, etc.
    # For now, basic

if __name__ == "__main__":
    real_path = 'raw/Resume/Resume.csv'
    syn_path = 'data/synthetic/resumes.json'
    evaluate_realism(real_path, syn_path)