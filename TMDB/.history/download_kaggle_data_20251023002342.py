import kaggle
import os

# Download the TMDB box office prediction dataset
try:
    kaggle.api.competition_download_files('tmdb-box-office-prediction', path='.', unzip=True)
    print("Dataset downloaded successfully!")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Make sure you have:")
    print("1. Kaggle API credentials in ~/.kaggle/kaggle.json")
    print("2. Accepted the competition rules on Kaggle website")