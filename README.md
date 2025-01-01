# Poker Player Clustering

## Project Structure
```poker_analyses/
├── src/
│   ├── player_clustering/        # Trains clustering model and applies it to new data set
│   │   ├── cluster_config.py     # Specify key paramaters for training and test data
│   │   ├── clusterer.py          # Core clustering logic
│   │   ├── clean_and_load_data.py# Data preprocessing
│   │   ├── train_model.py        # Training pipeline
│   │   └── predict_on_newdata.py # Prediction pipeline
├── docs/
│   └── usage.md                  # Detailed usage instructions
└── README.md                     # Project overview```

## Quick Start
For end users:
1. To classify new players: Use `predict_on_newdata.py`
2. To retrain the model: Use `train_model.py`

## Usage Flow
1. `predict_on_newdata.py` and `train_model.py` are the main entry points
2. These files use `clusterer.py` which contains the core clustering logic
3. `clusterer.py` uses `clean_and_load_data.py` for data preprocessing
