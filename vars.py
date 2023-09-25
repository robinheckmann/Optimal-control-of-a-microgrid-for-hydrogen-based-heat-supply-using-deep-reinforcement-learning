from ctypes import memmove
import torch

### General settings

NUM_TIME_STEPS = 8783
# TRAIN 168 TEST 8760

##### RL Agent parameters
NUM_EPISODES = 4000 # Number of episodes
EPSILON = 1 # For epsilon-greedy approach
EPS_DECAY = 0.997 # old: 0.997
LEARNING_RATE = 0.0001
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
TARGET_UPDATE = 10
BATCH_SIZE = 64 # 64
N_ACTIONS = 2
INPUT_DIMS = 5
FC_1_DIMS = 300
FC_2_DIMS = 600
FC_3_DIMS = FC_2_DIMS # If we don't want a third layer, set this to FC_2_DIMS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TAU = 1e-3 # For soft update
MEMORY_SIZE = 100000

#100000
##### Environment parameters #####

# REWARDS
SELL_PENALTY = 0.8 # Percentage of buying price, so selling price = SELL_PRICE_DISCOUNT*buying_price

# BATTERY STORAGE
C_MAX = 5000000 # Power in watt that the charging can provide divided by the time step size
D_MAX = 5000000 # Power in watt that the discharging can provide divided by the time step size
BATTERY_DEPRECIATION = 100 # When using the battery some depreciation is created
ETA = 0.95 # Battery charging efficiency
INITIAL_STORAGE = 1000000 # Set here what is initially stored inside of the battery
STORAGE_CAPACITY = 5000000 # Number of Watts that can be stored in the battery, 5 MWh
MIN_STORAGE = 10

# HYDROGEN STORAGE
HYDROGEN_STORAGE_CAPACITY = 10000 ## 6,000 kgH2 storage capacity of hydrogen tanks
INITIAL_HYDROGE_STORAGE = 0

# PHOTOVOLTAIK
PV_EFFICIENCY = 0.20
PV_CAPACITY = 100000000 
PV_SURFACE = PV_CAPACITY / 550 * 2.583252

# WIND TURBINES
NUM_WINDTURBINES = 1000

# ELECTROLYZER
ELECTROLYZER_POWER = 20000000   # (W) Watt
T = 180.0                       # (°C) Electrolyzer operating temperature
R = 8.31445                     # (J/mol-K) universal constant of gases
F = 96485.34                    # (C/mol) Faraday's constant
V_TANK = 200000                 # (m^3) volume of the tank
Nt = 1                          # number of tanks to be charged

# CONSUMPTION
AMOUNT_HOMES = 65000
