from parameters import *
from src.environment.user_mobility import ObstaclesMobilityModel
from src.users import User
from src.data_structures import Coords3d
from src.types_constants import StationType
from src.base_station import BaseStation
from src.mobility_management_ctrl.ho_events_controller import A3Event
import numpy as np
import random

class SimulationController:
    user_model = None
    time_step = 0
    def __init__(self):
        self.n_users = NUM_OF_USERS
        self.reset_users_model(n_users=self.n_users)
        self.users = [User(self.user_model.users[i])
                      for i in range(self.user_model.n_users)]
        self.base_stations = []
        self.time_ms = 0
        self.a3_event = None

    def reset_users_model(self, load=True, n_users=NUM_OF_USERS):
        if self.user_model is None:
            self.user_model = ObstaclesMobilityModel(number_of_users=self.n_users)
            # Comment below to not load
            if load:
                self.user_model.load_model(save_name=USER_MOBILITY_SAVE_NAME, max_num_users=n_users,
                                               speed_divisor=self.user_model.speed_divisor)
            else:
                self.user_model.reset_users()
        else:
            self.user_model.reset_users(self.user_model.speed_divisor)

    def add_base_station(self, coords: Coords3d = MBS_LOCATIONS[0], station_type: StationType = StationType.UMa):
        self.base_stations.append(BaseStation(coords=coords, station_type=station_type))
        self.update_ue_bs_rf_list()

    def update_ue_bs_rf_list(self):
        bs_rf_list = [_bs.rf_transceiver for _bs in self.base_stations]
        [_user.rf_transceiver.set_available_base_stations(bs_rf_list) for _user in self.users]

    def simulate_time_step(self, time_step=TIME_STEP, static_users=False):
        self.time_ms += time_step
        self.time_step = time_step
        if not self.user_model.generate_model_step(time_step, static_users):
            print("No more UE mobility data")
            return False

    def update_ues_serving_bs(self):
        [_user.rf_transceiver.calculate_serving_bs(True) for _user in self.users]

    def update_ues_received_sinrs(self):
        [_user.rf_transceiver.get_received_sinrs() for _user in self.users]

    def get_ues_received_sinr(self):
        return np.array([_user.rf_transceiver.received_sinrs for _user in self.users])

    def get_ues_serving_bs_id(self):
        return np.array([_user.rf_transceiver.serving_bs.id for _user in self.users])

    def check_a3_event(self):
        if not self.a3_event:
            self.a3_event = A3Event(self.n_users, len(self.base_stations))
        self.a3_event.check_users_state(self.get_ues_received_sinr(), self.get_ues_serving_bs_id(), self.time_step)

    def perform_ho(self):
        self.a3_event.perform_ho_from_mr(self.users, self.base_stations)

    def randomly_associate_ues(self):
        for _user in self.users:
            _user.rf_transceiver.serving_bs = random.choice(self.base_stations)

if __name__ == '__main__':
    a = SimulationController()
    a.add_base_station()
    a.add_base_station(Coords3d(0,400,MBS_HEIGHT))
    a.simulate_time_step()
    a.update_ue_bs_rf_list()
    a.update_ues_received_sinrs()
    a.randomly_associate_ues()
    a.check_a3_event()
    # a.simulate_time_step()
    # a.update_ues_received_sinrs()
    # a.check_a3_event()
    # a.perform_ho()
    # a.check_a3_event()