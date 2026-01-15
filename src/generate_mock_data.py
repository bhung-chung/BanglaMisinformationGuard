import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join("data", "raw")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def generate_mock_data():
    print("Generating mock data for pipeline verification...")
    
    # Authentic
    auth_data = {
        'content': [f"This is authentic news content {i} with some Bangla words আমি আছি" for i in range(100)],
        'domain': ['authentic.com'] * 100,
        'date': ['2023-01-01'] * 100
    }
    pd.DataFrame(auth_data).to_csv(os.path.join(DATA_DIR, 'Authentic-48K.csv'), index=False)
    
    # Fake
    fake_data = {
        'content': [f"This is fake news content {i} with rumors" for i in range(20)],
        'domain': ['fake.com'] * 20,
        'date': ['2023-01-01'] * 20
    }
    pd.DataFrame(fake_data).to_csv(os.path.join(DATA_DIR, 'Fake-1K.csv'), index=False)
    
    print(f"Mock data saved to {DATA_DIR}")

if __name__ == "__main__":
    generate_mock_data()
