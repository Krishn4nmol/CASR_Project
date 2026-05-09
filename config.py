# config.py
# All settings for CASR project
# This is K=4 experiment version

# ─────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────
DATA_PATH  = "data/"
TRAIN_DAYS = [1, 2, 3, 4, 5]
TEST_DAYS  = [6, 7]

# ─────────────────────────────────────────
# S-CACHE SETTINGS
# K=4 EXPERIMENT VERSION
# Original paper used K=3
# We test K=4 to see if finer
# granularity improves performance
# ─────────────────────────────────────────

# CHANGED FROM 3 TO 4 FOR EXPERIMENT
NUM_QUEUES = 4

# CHANGED: Split Queue 1 (1-60s) into
# Queue 1 (1-30s) and Queue 2 (30-60s)
# Original: [0, 1, 60, inf]
# New:      [0, 1, 30, 60, inf]
QUEUE_BOUNDARIES = [0, 1, 30, 60, float('inf')]

# CHANGED: Added 4th queue capacity
# Original: [5000, 500, 100]
# New:      [5000, 500, 200, 50]
INITIAL_QUEUE_CAPACITY = [5000, 500, 200, 50]

WINDOW_CACHE_RATIO = 0.2

# ─────────────────────────────────────────
# SERVER SETTINGS
# ─────────────────────────────────────────
SERVER_MEMORY_MB            = 4096
DEFAULT_CONTAINER_MEMORY_MB = 128

# ─────────────────────────────────────────
# KEY SETTING: NUMBER OF FUNCTIONS
# ─────────────────────────────────────────
NUM_FUNCTIONS = 2000

# ─────────────────────────────────────────
# KEY SETTING: CALLS PER WORKLOAD
# ─────────────────────────────────────────
EVAL_CALLS = 100000

# ─────────────────────────────────────────
# REINFORCEMENT LEARNING SETTINGS
# ─────────────────────────────────────────
THETA          = 0.8
DELTA          = 10000
SCALING_FACTOR = 0.25

# ─────────────────────────────────────────
# PPO SETTINGS
# Same as paper Table 2
# ─────────────────────────────────────────
LEARNING_RATE_ACTOR  = 0.001
LEARNING_RATE_CRITIC = 0.001
HIDDEN_LAYER_SIZE    = 128
DISCOUNT_FACTOR      = 0.63
GAE_LAMBDA           = 0.95
PPO_CLIP             = 0.2
ENTROPY_COEFF        = 0.01
MINI_BATCH_SIZE      = 20
REPLAY_BUFFER_SIZE   = 1000
EPOCHS_PER_UPDATE    = 10

# ─────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────
MAX_EPISODES      = 200
CALLS_PER_EPISODE = 100000

# CHANGED: Separate folder for K=4 model
# So K=3 trained model is not overwritten!
MODEL_SAVE_PATH   = "trained_model_k4/"

PRINT_EVERY       = 10

# ─────────────────────────────────────────
# BASELINE SETTINGS
# ─────────────────────────────────────────
FIXED_KEEPALIVE_SECONDS = 600

# ─────────────────────────────────────────
# RESULTS SETTINGS
# CHANGED: Separate folder for K=4 results
# So K=3 results are not overwritten!
# ─────────────────────────────────────────
THETA_VALUES_TO_TEST = [0.2, 0.4, 0.6, 0.8]
RESULTS_PATH         = "results_k4/"

# ─────────────────────────────────────────
# COOLING SETTINGS
# ─────────────────────────────────────────
COOLING_BETWEEN_ALGORITHMS = 30
COOLING_BETWEEN_WORKLOADS  = 120

# ─────────────────────────────────────────
# K=4 EXPERIMENT NOTES
# ─────────────────────────────────────────
# Original paper K=3:
#   Queue 0: 0-1s   (lightweight HTTP)
#   Queue 1: 1-60s  (medium functions)
#   Queue 2: 60+s   (heavy ML)
#
# Our K=4 experiment:
#   Queue 0: 0-1s   (lightweight HTTP)
#   Queue 1: 1-30s  (medium-light)
#   Queue 2: 30-60s (medium-heavy)
#   Queue 3: 60+s   (heavy ML)
#
# Hypothesis: Splitting the dominant
# Queue 1 (90% of Azure calls) into
# two groups allows agent to manage
# them with different strategies
# potentially improving performance