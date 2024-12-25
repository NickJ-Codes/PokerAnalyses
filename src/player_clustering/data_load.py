import pandas as pd
from pathlib import Path
from cluster_config import  filename_AllPlayers

def load_player_data(file = filename_AllPlayers):
    try:
        # Get the project root directory (assuming we're in src/player_clustering)
        project_root = Path(__file__).parent.parent.parent

        # Construct the path to the data file
        data_path = project_root / 'data' / file

        # Check if file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        # Read the CSV file
        df = pd.read_csv(data_path)

        # Basic data validation
        if df.empty:
            raise ValueError("The CSV file is empty")

        print(f"Successfully loaded {len(df)} player records")
        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty or corrupted")
        return None
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        return None

if __name__ == "__main__":
    # Test the function
    player_data = load_player_data()
    if player_data is not None:
        print("\nFirst few rows of the data:")
        print(player_data.head())
        print("\nColumns in the dataset:")
        print(player_data.columns.tolist())
