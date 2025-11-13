import pandas as pd
import re

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Resume.csv file.
    Assumes pipe-separated values with columns: ID, Resume_str, Resume_html, Category
    """
    # Load the data, skipping the malformed first line
    df = pd.read_csv(file_path, sep='|', header=None, names=['ID', 'Resume_str', 'Resume_html', 'Category'], skiprows=1)
    
    # Clean the Resume_str: remove extra whitespaces
    df['Resume_str'] = df['Resume_str'].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()))
    
    # Clean Category
    df['Category'] = df['Category'].str.strip()
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df