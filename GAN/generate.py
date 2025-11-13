from sdv.single_table import CTGANSynthesizer
import os

def generate_synthetic_data(model_path, num_samples=10000, output_path='data/synthetic/resumes.json'):
    """
    Load trained synthesizer, generate synthetic data, and save to JSON.
    """
    synthesizer = CTGANSynthesizer.load(model_path)
    
    synthetic_data = synthesizer.sample(num_rows=num_samples)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON lines
    synthetic_data.to_json(output_path, orient='records', lines=True)
    
    print(f"Generated {num_samples} synthetic resumes saved to {output_path}")
    return synthetic_data

if __name__ == "__main__":
    model_path = 'ctgan_model.pkl'
    generate_synthetic_data(model_path)