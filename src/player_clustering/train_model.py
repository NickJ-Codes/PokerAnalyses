import logging
from pathlib import Path
from clusterer import PokerPlayerClusterer
from clean_and_load_data import get_clean_data
from cluster_config import (
    CLUSTERING_FEATURES,
    MODEL_DIR,
    OUTPUT_DIR,
    N_CLUSTERS,
    FILE_TRAINING_DATA
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model():
    """
    Train the clustering model on the training dataset and save it.
    """
    try:
        # Load and clean training data
        logger.info(f"Loading and cleaning training data from {FILE_TRAINING_DATA}")
        train_data = get_clean_data(FILE_TRAINING_DATA)

        if train_data is None:
            raise ValueError("Failed to load training data")

        # Validate features
        missing_features = [feat for feat in CLUSTERING_FEATURES if feat not in train_data.columns]
        if missing_features:
            raise ValueError(f"Training data missing required features: {missing_features}")

        # Log data shape and features
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Available features: {train_data.columns.tolist()}")

        # Initialize clusterer
        logger.info(f"Initializing clusterer with {N_CLUSTERS} clusters")
        clusterer = PokerPlayerClusterer(n_clusters=N_CLUSTERS)

        # Fit the model
        logger.info("Training model...")
        clusterer.fit(
            df=train_data,
            clustering_features=CLUSTERING_FEATURES
        )

        # Create model directory if it doesn't exist
        model_path = Path(MODEL_DIR)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save the trained model
        model_path = model_path / 'poker_clusterer.pkl'
        logger.info(f"Saving trained model to {model_path}")
        clusterer.save_model(model_path)

        # Validate the saved model
        logger.info("Validating saved model...")
        loaded_model = PokerPlayerClusterer.load_model(model_path)
        test_pred = loaded_model.predict(train_data.head())

        logger.info("Model training completed successfully!")

        # Calculate and display cluster information
        results = clusterer.predict(train_data)
        cluster_means = clusterer.get_cluster_means(results)

        logger.info("\nCluster Means:\n%s", cluster_means.to_string())

        # Optionally save cluster means
        output_path = Path(OUTPUT_DIR)
        cluster_means_path = output_path / 'cluster_means_trainingSet.csv'
        cluster_means.to_csv(cluster_means_path, index=False)
        logger.info(f"Saved cluster means to {cluster_means_path}")

        #output original DF with columns for cluster names to output directory using the predict function in clusterer.py, class pokerplayerclusterer
        df_with_clusters = clusterer.predict(train_data)
        classified_trainingdata_path = output_path / 'training_data_with_clusters.csv'
        df_with_clusters.to_csv(classified_trainingdata_path, index = 'false')


        # Log cluster sizes
        cluster_sizes = results['cluster'].value_counts().sort_index()
        logger.info("\nCluster sizes:\n%s", cluster_sizes.to_string())

        #output original DF with columns for cluster names to output directory



    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

def main():
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
