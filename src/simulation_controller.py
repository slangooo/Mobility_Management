from parameters import *
from src.environment.user_mobility import ObstaclesMobilityModel
from src.users import User
from src.data_structures import Coords3d
from src.types_constants import StationType

class SimulationController:
    def __init__(self):
        self.n_users = NUM_OF_USERS
        self.reset_mobility_model(n_users=self.n_users)
        self.users = [User(self.mobility_model.users[i])
                      for i in range(self.mobility_model.n_users)]
        self.base_stations = []

    def reset_users_model(self, load=True, n_users=NUM_OF_USERS):
        if self.user_model is None:
            self.user_model = ObstaclesMobilityModel(number_of_users=self.n_users)
            # Comment below to not load
            if load:
                self.mobility_model.load_model(save_name=USER_MOBILITY_SAVE_NAME, max_num_users=self.n_users,
                                               speed_divisor=self.speed_divisor)
            else:
                self.mobility_model.reset_users()
        else:
            self.mobility_model.reset_users(self.speed_divisor)

    def add_base_station(self, coords: Coords3d=Coords3d(0, 0, 15), ):