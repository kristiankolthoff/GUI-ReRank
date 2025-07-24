import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / 'resources' / 'evaluation' / 'results'

# Model names to analyze (longest first for correct matching)
MODEL_NAMES = [
    'model_gpt_4_1_nano',
    'model_gpt_4_1_mini',
    'model_gpt_4_1',
]

# Find all relevant result files
text_result_files = list(RESULTS_DIR.glob('rerank_results_*_text_temp*.csv'))
image_result_files = list(RESULTS_DIR.glob('rerank_results_*_image_temp*.csv'))

print("Text result files found:")
for f in text_result_files:
    print(f"  {f}")
print("Image result files found:")
for f in image_result_files:
    print(f"  {f}")

# Helper to extract model name from filename
def extract_model_name(filename):
    for name in MODEL_NAMES:
        if name in filename:
            return name
    return None

def analyze_results(result_files, model_type_label):
    model_results = {name: None for name in MODEL_NAMES}
    for file in result_files:
        model_name = extract_model_name(str(file))
        if model_name:
            model_results[model_name] = pd.read_csv(file)
    print(f"\n==== {model_type_label.upper()} MODELS ====")
    for model_name, df in model_results.items():
        if df is None:
            print(f"No results found for model: {model_name}")
            continue
        print(f"\nModel: {model_name}")
        for col in ['rerank_time_seconds', 'input_tokens', 'output_tokens', 'total_tokens']:
            if col in df.columns:
                norm_col = df[col] / 20.0
                mean = np.mean(norm_col)
                std = np.std(norm_col)
                print(f"  {col} (per GUI): mean = {mean:.2f}, std = {std:.2f}")
            else:
                print(f"  {col}: column not found in results.")

if __name__ == "__main__":
    analyze_results(text_result_files, "text")
    analyze_results(image_result_files, "image")
