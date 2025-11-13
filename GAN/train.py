from data_prep import load_and_preprocess_data
from model import create_ctgan_model

def train_model(data_path, save_path='ctgan_model.pkl'):
    """
    Load data, create synthesizer, train, and save the CTGAN model.
    """
    df = load_and_preprocess_data(data_path)
    print(f"Loaded {len(df)} resumes for training.")
    
    synthesizer = create_ctgan_model()
    print("Training CTGAN synthesizer...")
    synthesizer.fit(df)
    
    synthesizer.save(save_path)
    print(f"Synthesizer saved to {save_path}")
    return synthesizer

if __name__ == "__main__":
    data_path = 'raw/Resume/Resume.csv'
    train_model(data_path)