import numpy as np
import re
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from cluster_config import FILE_TRAINING_DATA, MIN_HANDS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning parameters"""
    min_hands: int = MIN_HANDS
    default_fill_value: float = 0.0
    special_columns: dict = None

    def __post_init__(self):
        self.special_columns = {
            'hands': {'comma_remove': True},
            'cbet_f': {'dash_to_zero': True},
            '3bet_f': {'dash_to_zero': True}
        }

class DataCleaner:
    """Class to handle data loading and cleaning operations"""

    def __init__(self, config: DataCleaningConfig = None):
        self.config = config or DataCleaningConfig()

    @staticmethod
    def _clean_column_name(col: str) -> str:
        """Clean individual column names"""
        col = col.lower()
        col = re.sub(r'[^a-z0-9]', '_', col)
        col = re.sub(r'_+', '_', col)
        return col.strip('_')

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize all column names in the DataFrame"""
        df.columns = [self._clean_column_name(col) for col in df.columns]
        return df

    def _handle_special_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process columns that need special handling"""
        for col, operations in self.config.special_columns.items():
            if col not in df.columns:
                continue

            if operations.get('comma_remove'):
                df[col] = df[col].str.replace(',', '').astype(float)

            if operations.get('dash_to_zero'):
                df[col] = df[col].replace('-', '0').astype(float)

        return df

    def _convert_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert percentage strings to float values"""
        percentage_columns = [col for col in df.columns
                              if 'percentage' in col or 'pct' in col]

        for col in percentage_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.rstrip('%').astype('float') / 100.0

        return df

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found at: {path}")

            df = pd.read_csv(path)
            if df.empty:
                raise ValueError("The CSV file is empty")

            logger.info(f"Successfully loaded {len(df)} player records")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def clean_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and process the loaded data"""
        try:
            # Standardize column names
            df = self.standardize_column_names(df)

            # Remove duplicates
            df = df.drop_duplicates()

            # Handle missing values in numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(self.config.default_fill_value)

            # Process percentage columns
            df = self._convert_percentages(df)

            # Handle special columns
            df = self._handle_special_columns(df)

            # Filter by minimum hands
            df = df[df['hands'] >= self.config.min_hands]

            logger.info("Data cleaning completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            return None

def get_clean_data(file_path: str = FILE_TRAINING_DATA) -> Optional[pd.DataFrame]:
    """Main function to load and clean data"""
    cleaner = DataCleaner()
    df = cleaner.load_data(file_path)
    if df is not None:
        return cleaner.clean_data(df)
    return None

if __name__ == "__main__":
    cleaned_data = get_clean_data()

    if cleaned_data is not None:
        logger.info("\nFirst few rows of cleaned data:")
        print(cleaned_data.head())

        logger.info("\nCleaned columns:")
        print(cleaned_data.columns.tolist())

        logger.info("\nData info:")
        print(cleaned_data.info())

        logger.info("\nMissing values after cleaning:")
        print(cleaned_data.isnull().sum())