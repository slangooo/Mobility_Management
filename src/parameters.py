from src.math_tools import db2lin, wh_to_joules, lin2db
from src.types_constants import LinkType
from src.data_structures import Coords3d
from numpy import exp
from numpy import log

#Obstacles
EXTEND_TIMES_FOUR = True

# Users Mobility Model
NUM_OF_USERS = 140
USER_SPEED = [0.5, 0.8]
PAUSE_INTERVAL = [0, 60]
TIME_STEP = 100  # ms Between subsequent users mobility model updates
TIME_SLEEP = 2  # Sleep between updates to allow plotting to keep up
BUILDINGS_AREA_MARGIN = 50
SIMULATION_TIME = 60 * 60 * 2
USER_SPEED_DIVISOR = 1



# Channel model PLOS
PLOS_AVG_LOS_LOSS = 1
PLOS_AVG_NLOS_LOSS = 20
# PLOS_A_PARAM = 9.61
# PLOS_B_PARAM = 0.16
PLOS_A_PARAM = 5.05 #Obtained using the method
PLOS_B_PARAM = 0.38

# Channel model RF
DRONE_TX_POWER_RF = 0.2  # W
USER_BANDWIDTH = 500e3 #*2  # Hz
DRONE_BANDWIDTH = 20e6  #+5e6 # Hz
MBS_BANDWIDTH = 20e6  # Hz
DEFAULT_CARRIER_FREQ_MBS = 2e9  # Hz
DEFAULT_CARRIER_FREQ_DRONE = 2e9 + MBS_BANDWIDTH  # Hz
NOISE_SPECTRAL_DENSITY = -174  # dBm/Hz
NOISE_POWER_RF = db2lin(NOISE_SPECTRAL_DENSITY - 30 + lin2db(USER_BANDWIDTH))  # dBm input -> linear in W
DEFAULT_SNR_THRESHOLD = db2lin(25)  # linear
MBS_TX_POWER_RF = 0.5  # W
DEFAULT_SINR_THRESHOLD = db2lin(10)
ASSOCIATION_SCHEME = 'SINR'
DEFAULT_SINR_SENSITIVITY_LEVEL = db2lin(-10)
DEFAULT_RX_POWER_SENSITIVITY = db2lin(-300)

#Mobility Management
DEFAULT_A3_INDIV_OFFSET = 0 #dB
DEFAULT_A3_SINR_OFFSET = 5 #dB
DEAFULT_A3_SINR_HYST = 2 #dB
DEFAULT_TTT_MS = 1024 #e 0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, and 5120 ms.
DEFAULT_HO_SINR_THRESHOLD = db2lin(-10)

# Channel model FSO
RX_DIAMETER = 0.2  # m
DIVERGENCE_ANGLE = 0.06  # rads
RX_RESPONSIVITY = 0.5
AVG_GML = 3
WEATHER_COEFF = 4.3 * 10 ** -4  # /m
POWER_SPLIT_RATIO = 0.005
FSO_ENERGY_HARVESTING_EFF = 0.2
TX_POWER_FSO_DRONE = 0.2  # W
TX_POWER_FSO_MBS = 380  # W
BANDWIDTH_FSO = 1e9  # Hz
NOISE_VARIANCE_FSO = 0.8 * 1e-9
NOISE_POWER_FSO = 1e-6
EMPIRICAL_SNR_LOSSES = db2lin(15)  # Linear
BEAMWAIST_RADIUS = 0.25e-3
WAVELENGTH = 1550e-9
AvgGmlLoss = {LinkType.A2G: 3, LinkType.A2A: 3 / 1.5, LinkType.G2G: 5}  # TODO: Get refs


# BEAMWAIST_RADII = [0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015]
BEAMWAIST_RADII = [0.01, 0.01, 0.02, 0.02]

UAV_HEIGHT = 25 #m
MBS_HEIGHT = 25  # m
UE_HEIGHT = 1.5  # To conform with channel models

MBS_LOCATIONS = [Coords3d(400, 0, MBS_HEIGHT)]

# USER_MOBILITY_SAVE_NAME = 'users_200_truncated'
USER_MOBILITY_SAVE_NAME = 'extended_4_madrids_500_users'