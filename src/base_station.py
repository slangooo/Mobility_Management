#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from itertools import count
from src.data_structures import Coords3d
from src.apparatus.rf_transciever import StationRfTransceiver
from src.parameters import MBS_BANDWIDTH, MBS_TX_POWER_RF, DEFAULT_CARRIER_FREQ_MBS, MBS_LOCATIONS

from src.types_constants import StationType


class BaseStation:
    _ids = count(0)

    def __init__(self, mbs_id: int = None, coords: Coords3d = MBS_LOCATIONS[0], station_type=StationType.UMa):
        self.id = next(self._ids) if mbs_id is None else mbs_id
        self.coords = coords.copy()
        self.rf_transceiver = StationRfTransceiver(coords=self.coords, t_power=MBS_TX_POWER_RF, bandwidth=MBS_BANDWIDTH,
                                                   bs_id=self.id, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS,
                                                   station_type=station_type)
        self.station_type = StationType.UMa
