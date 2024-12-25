# Configuration for clustering parameters
import os
#Load file name
filename_AllPlayers = 'All Players Report 20241225.csv'

#Minimum hands to worth classifying, min sample size
MIN_HANDS = 1000

# Number of clusters (k parameter)
N_CLUSTERS = 7

# Dictionary mapping cluster numbers to meaningful names
CLUSTER_NAMES = {
    0: 'The Loose Passive Fish',
    1: 'The Solid Regs',
    2: 'The Hyper-aggro Maniacs',
    3: 'The Tight Passive',
    4: 'The Loose Aggro Fish',
    5: 'The Moderate Aggressor',
    6: 'The Big-Betting Fish'
}

# If CLUSTER_NAMES size doesn't match N_CLUSTERS, generate generic names
if len(CLUSTER_NAMES) != N_CLUSTERS:
    CLUSTER_NAMES = {i: f'Cluster_TBD_Name' for i in range(N_CLUSTERS)}


# Color scheme for visualization
CLUSTER_COLORS = {
    0: '#e41a1c',    # red
    1: '#377eb8',    # blue
    2: '#4daf4a',    # green
    3: '#984ea3',    # purple
    4: '#ff7f00',    # orange
    5: '#ffff33',    # yellow
    6: '#a65628',    # brown
    7: '#f781bf',    # pink
    8: '#999999',    # grey
    9: '#66c2a5',    # mint
    10: '#fc8d62',   # salmon
    11: '#8da0cb',   # light blue
    12: '#e78ac3',   # rose
    13: '#a6d854',   # lime
    14: '#ffd92f'    # golden yellow
}

# Feature columns used for clustering
CLUSTERING_FEATURES = [
    'bb_100',        # Big Bets per 100 hands - overall profitability
    'vpip',          # Voluntarily Put Money in Pot - basic preflop playing style
    'pfr',           # PreFlop Raise - aggression metric
    '3bet_f',        # 3-bet frequency - preflop aggression
    'fold_to_3bet',  # Chance the person folds to 3B after 2Betting
    'flop_af',       # Flop aggression factor
    'donk_f',        # donk the flop
    'cbet_f',        # Flop c-bet - postflop passivity
    'fold_to_f_cbet',# Fold to flop c-bet - postflop passivity
    'xr_flop',       # Check raise the flop
    'bet_t',         # bet the turn
    'bet_r',         # bet the turn
    'call_r_bet',    # call river bet
    'wtsd',          # Went to ShowDown - postflop tendency
    'wsd',           # Won money at ShowDown - showdown success
]

EXCLUDE_CLUSTERING_FEATURES = [
    'player',
    'hands',
]

# Optional: Clustering algorithm parameters
RANDOM_STATE = 42  # For reproducibility
MAX_ITERATIONS = 300
N_INIT = 10

# Output directory and file path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Output directory and file path
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_FILE_PLAYERS = os.path.join(OUTPUT_DIR, "clusteredplayers.csv")
OUTPUT_FILE_CLUSTER_MEANS = os.path.join(OUTPUT_DIR, "player_cluster_means.csv")