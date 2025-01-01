import logging
from pathlib import Path
from clusterer import PokerPlayerClusterer
from clean_and_load_data import get_clean_data
from cluster_config import (
    CLUSTERING_FEATURES,
    MODEL_DIR,
    OUTPUT_DIR,
    FILE_NEWDATA
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_on_new_data():

    """
    Apply the pre-existing clustering model on new data and save it.
    Requirements:
        Config file has FILE_NEWDATA specified and existing
        Config file has MODEL_FILE specified and existing
    """
    try:
        # Load and clean training data
        logger.info(f"Loading and cleaning training data from {FILE_NEWDATA}")
        data = get_clean_data(FILE_NEWDATA)

        if data is None:
            raise ValueError("Failed to load new data")

        # Validate features
        missing_features = [feat for feat in CLUSTERING_FEATURES if feat not in data.columns]
        if missing_features:
            raise ValueError(f"Training data missing required features: {missing_features}")

        # Log data shape and features
        logger.info(f"Training data shape: {data.shape}")
        logger.info(f"Available features: {data.columns.tolist()}")

        # load pre-trained model
        logger.info("Validating saved model...")
        model_path = Path(MODEL_DIR)
        model_path.mkdir(parents=True, exist_ok=True)
        model_path = model_path / 'poker_clusterer.pkl'
        loaded_model = PokerPlayerClusterer.load_model(model_path)

        # Apply pre-trained model to new dataset
        logger.info("Making predictions on new data set...")
        df_with_clusters = loaded_model.predict(data)
        output_path = Path(OUTPUT_DIR)
        classifiedDataPath = output_path / 'NewData_with_clusters.csv'
        df_with_clusters.to_csv(classifiedDataPath, index = 'false')

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

def main():
    try:
        predict_on_new_data()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
