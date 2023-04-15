from src.parameters import *
import numpy as np
from src.channel_model.rf_a2g import PlosModel
from src.channel_model.rf_g2g import UmaCellular, UmiCellular
from src.data_structures import Coords3d
from src.types_constants import StationType
import types
from scipy.ndimage.interpolation import shift
from itertools import compress


class RfTransceiver:
    received_powers = []
    bs_sinr_cochannel_mask = None
    bs_cochannel_mask = None
    bs_list = []
    cochannel_bs_list = []
    sinr_cochannel_bs_list = []
    received_sinrs = []
    received_snrs = []
    received_powers = []
    serving_bss = []
    max_sinr_idx = None
    serving_bs = None
    serving_sinr = None
    serving_snr = None
    serving_rx_power = None

    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS):

        self.tx_power = t_power
        self.coords = coords
        self.noise_power = NOISE_POWER_RF
        self.bandwidth = bandwidth
        self.user_id = user_id
        self.bs_id = bs_id
        self.carrier_frequency = carrier_frequency
        self.get_path_loss = get_path_loss_function(StationType.UE)
        self.available_bandwidth = 0

    def get_received_powers(self):
        for bs_idx, bs in enumerate(self.sinr_cochannel_bs_list):
            path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
            self.received_powers[bs_idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)

    def get_received_sinrs(self):
        self.get_received_powers()
        for bs_idx, bs in enumerate(self.sinr_cochannel_bs_list):
            rx_powers = self.received_powers
            interferences = np.sum(rx_powers) - rx_powers[bs_idx]
            self.received_sinrs[bs_idx] = rx_powers[bs_idx] / (interferences + self.noise_power)
            self.received_snrs[bs_idx] = rx_powers[bs_idx] / self.noise_power

    def get_received_snrs(self):
        self.get_received_powers()
        for bs_idx, bs in enumerate(self.sinr_cochannel_bs_list):
            self.received_snrs[bs_idx] = self.received_powers[bs_idx] / self.noise_power

    def update_cochannel_bs_mask(self, frequency_condition=None):
        if frequency_condition is None:
            self.bs_cochannel_mask = np.ones(len(self.stations_list), dtype=bool)
        else:
            for idx, _station in enumerate(self.stations_list):
                if _station.carrier_frequency == frequency_condition:
                    self.bs_cochannel_mask[idx] = 1
        self.cochannel_bs_list = list(compress(self.stations_list, self.bs_cochannel_mask))

    def update_sufficient_sinr_cochannel_bs_mask(self, sinr_condition=None):
        self.bs_sinr_cochannel_mask = np.copy(self.bs_cochannel_mask)
        self.sinr_cochannel_bs_list = list(compress(self.stations_list, self.bs_cochannel_mask))

        rx_powers = np.zeros(len(self.stations_list))
        for bs_idx, bs in enumerate(self.stations_list):
            if not self.bs_cochannel_mask[bs_idx]:
                rx_powers[bs_idx] = 0
            path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
            rx_powers[bs_idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)

        rx_sinr = np.zeros(len(self.stations_list))

        for bs_idx, bs in enumerate(self.stations_list):
            if not self.bs_cochannel_mask[bs_idx]:
                rx_powers[bs_idx] = 0
            interferences = np.sum(rx_powers) - rx_powers[bs_idx]
            rx_sinr[bs_idx] = rx_powers[bs_idx] / (interferences + self.noise_power)

        for idx, sinr in enumerate(rx_sinr):
            if sinr < sinr_condition:
                self.bs_sinr_cochannel_mask[idx] = False

        self.sinr_cochannel_bs_list = list(compress(self.stations_list, self.bs_sinr_cochannel_mask))
        self.received_sinrs = rx_sinr[self.bs_sinr_cochannel_mask]
        self.received_powers = rx_powers[self.bs_sinr_cochannel_mask]

    def set_available_base_stations(self, stations_list, sinr_condition=None, frequency_condition=None):
        self.stations_list = stations_list
        self.update_cochannel_bs_mask(frequency_condition)
        self.update_sufficient_sinr_cochannel_bs_mask(sinr_condition)

    def calculate_serving_bs(self, recalculate=False):
        """Return base_station ID, SINR, SNR, Received Power, Capacity"""
        if recalculate:
            self.get_received_sinrs()

        self.max_sinr_idx = self.received_sinrs.argmax()
        self.serving_bs = [self.sinr_cochannel_bs_list[self.max_sinr_idx]]
        self.serving_sinr = self.received_sinrs[self.max_sinr_idx]
        self.serving_snr = self.received_powers[self.max_sinr_idx] / self.noise_power
        self.serving_rx_power = self.received_powers[self.max_sinr_idx]

        return self.serving_bs.bs_id, self.received_sinr, self.received_snr, self.received_power

    def is_sinr_satisfied(self):
        return self.received_sinr >= DEFAULT_SINR_THRESHOLD

    def is_snr_satisfied(self):
        return self.received_snr >= DEFAULT_SNR_THRESHOLD


class UserRfTransceiver(RfTransceiver):
    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS):
        super().__init__(coords, t_power=t_power, bandwidth=bandwidth, bs_id=bs_id, user_id=user_id,
                         carrier_frequency=carrier_frequency)
        self.get_path_loss = None


class MacroRfTransceiver(RfTransceiver):
    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS, station_type=StationType.UMa,
                 macro_type=StationType.UMa):
        super().__init__(coords, t_power=t_power, bandwidth=bandwidth, bs_id=bs_id, user_id=user_id,
                         carrier_frequency=carrier_frequency, station_type=station_type)
        if macro_type == StationType.UMa:
            self.get_path_loss = UmaCellular.get_path_loss
        elif macro_type == StationType.UMi:
            self.get_path_loss = UmiCellular.get_path_loss
        elif macro_type == StationType.DBS:
            self.get_path_loss = PlosModel.get_path_loss
        else:
            self.get_path_loss = None


def get_path_loss_function(station_type):
    if station_type == StationType.UE:
        return None
    elif station_type == StationType.UMa:
        return UmaCellular.get_path_loss
    elif station_type == StationType.UMi:
        return UmiCellular.get_path_loss
    elif station_type == StationType.DBS:
        return PlosModel.get_path_loss
    else:
        return None


if __name__ == "__main__":
    rf1 = RfTransceiver(Coords3d(0, 1, 2))
    rf2 = RfTransceiver(Coords3d(5, 6, 7))
    rf3 = RfTransceiver(rf1.coords)
