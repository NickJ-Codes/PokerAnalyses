# Preprocess and clean the data from an all players report
import numpy as np
import re
from data_load import load_player_data
from cluster_config import (
    filename_AllPlayers,
    MIN_HANDS
)

def standardize_column_names(df):
    """
    Standardize column names:
    - Convert to lowercase
    - Replace spaces and special characters with underscores
    - Remove any remaining special characters
    - Remove duplicate underscores
    """
    def clean_column_name(col):
        # Convert to lowercase
        col = col.lower()
        # Replace spaces and special characters with underscore
        col = re.sub(r'[^a-z0-9]', '_', col)
        # Remove duplicate underscores
        col = re.sub(r'_+', '_', col)
        # Remove leading/trailing underscores
        col = col.strip('_')
        return col

    df.columns = [clean_column_name(col) for col in df.columns]
    return df

def clean_player_data(file = filename_AllPlayers):
    """
    Load and clean the player data
    """
    # Load the data using data_load.py
    df = load_player_data(file)

    if df is not None:
        try:
            # Standardize column names
            df = standardize_column_names(df)

            # Add additional cleaning steps here, for example:
            # 1. Remove duplicate entries
            df = df.drop_duplicates()

            # 2. Handle missing values appropriately
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)

            # 3. Convert data types if needed
            # Example: Convert percentage strings to floats
            percentage_columns = [col for col in df.columns if 'percentage' in col or 'pct' in col]
            for col in percentage_columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.rstrip('%').astype('float') / 100.0

            # 4. Clean hands column (remove commas)
            df['hands'] = df['hands'].str.replace(',', '').astype(float)

            # 5. Replace dashes with zeros in cbet columns and convert to numeric
            df['cbet_f'] = df['cbet_f'].replace('-', '0').astype(float)
            df['3bet_f'] = df['3bet_f'].replace('-', '0').astype(float)

            # 6. Filter out players with less than MIN_HANDS
            df = df[df['hands'] >= MIN_HANDS]

            print("Data cleaning completed successfully")
            return df

        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return None

    return None

def get_clean_data(file = filename_AllPlayers):
    """
    Wrapper function to get cleaned data ready for analysis
    """
    return clean_player_data(file)

if __name__ == "__main__":
    # Test the cleaning process
    cleaned_data = get_clean_data(file = filename_AllPlayers)

    if cleaned_data is not None:
        print("\nFirst few rows of cleaned data:")
        print(cleaned_data.head())

        print("\nCleaned columns:")
        print(cleaned_data.columns.tolist())

        print("\nData info:")
        print(cleaned_data.info())

        print("\nMissing values after cleaning:")
        print(cleaned_data.isnull().sum())
