#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.environment.user_modeling import UserWalker
import numpy as np
from src.parameters import *
from src.environment.user_modeling import UserSpatialModel
from src.environment.obstacles import Obstacles, get_madrid_buildings


import os

import time as timee


class ObstaclesMobilityModel(UserSpatialModel):
    loaded = False
    seeks = []
    loaded_duration = 0

    def __init__(self, number_of_users=NUM_OF_USERS):
        self.obstacles_objects = get_madrid_buildings()
        self.users = []
        self.current_time = 0
        self.n_users = number_of_users
        self.speed_divisor = USER_SPEED_DIVISOR

    def reset_users(self, speed_divisor=1):
        self.current_time = 0
        self.load_model(save_directory_in=self.loaded, duration=self.loaded_duration, max_num_users=self.n_users,
                        speed_divisor=speed_divisor)

    def update_users_locations(self, time_step):
        for _user in self.users:
            if _user.update_location(time_step):
                with open(os.path.join(self.loaded, f"user_{_user.id}"), 'rb') as _file:
                    _file.seek(self.seeks[_user.id])
                    _wayps = np.load(_file, allow_pickle=True)
                    waypoints = [Coords3d.from_array(_arr) for _arr in _wayps]
                    next_pause_time = int(np.load(_file, allow_pickle=True))
                    self.seeks[_user.id] = _file.tell()
                _user.set_waypoints(waypoints.copy(), 0)
                _user.last_waypoint = waypoints[-1]
                _user.next_pause_time = next_pause_time

    def generate_model(self, duration=None):
        while duration is None or self.current_time < duration:
            if not self.generate_model_step():
                return

    def generate_model_step(self, time_step=TIME_STEP, static_users=False):
        self.current_time += time_step / 1000
        if self.current_time > self.loaded_duration > 0:
            return False
        if not static_users:
            self.update_users_locations(time_step / 1000)
        return True

    def get_obstacles(self):
        return self.obstacles_objects.obstaclesList

    def add_base_stations_to_plot(self, base_stations_coords):
        self.base_stations_coords = base_stations_coords

    def load_model(self, folder_name='mobility_model_saved',
                   save_name="default", duration=60 * 60 * 2, max_num_users=None, save_directory_in=None,
                   speed_divisor=USER_SPEED_DIVISOR):
        save_directory = os.path.join(os.getcwd(), "environment", folder_name,  save_name) if not save_directory_in else save_directory_in
        self.seeks = []
        n_users = len(os.listdir(save_directory))
        self.n_users = min(n_users, max_num_users if max_num_users else n_users)
        for uid in range(self.n_users):
            with open(os.path.join(save_directory, f"user_{uid}"), 'rb') as _file:
                _file.seek(0)
                speed = float(np.load(_file, allow_pickle=True)) / speed_divisor
                wyps = np.load(_file, allow_pickle=True)
                next_pause_time = int(np.load(_file, allow_pickle=True))
                self.seeks.append(_file.tell())
                if wyps.size == 0:
                    wyps = [Coords3d.from_array(_arr) for _arr in np.load(_file, allow_pickle=True)]
                    waypoints = [wyps[0]]
                else:
                    waypoints = [Coords3d.from_array(_arr) for _arr in wyps]
            starting_coords = Coords3d.from_array(waypoints[0])
            if len(self.users) > uid:
                self.users[uid].current_coords.set(starting_coords)
                self.users[uid].next_pause_time = next_pause_time
                self.users[uid].set_waypoints(waypoints.copy(), 0)
                self.users[uid].remaining_pause_time = 0
            else:
                _user = UserWalker(uid, initial_coords=starting_coords)
                _user.speed = speed
                _user.next_pause_time = next_pause_time
                _user.set_waypoints(waypoints.copy(), 0)
                self.users.append(_user)

        self.loaded = save_directory
        self.loaded_duration = duration
        # self.users = tuple(self.users)


if __name__ == '__main__':
    mobility_model = ObstaclesMobilityModel()
    time1 = timee.time()
    mobility_model.load_model(save_name='extended_4_madrids_500_users', max_num_users=20)
    mobility_model.generate_model_step(TIME_STEP)
    # mobility_model.generate_model(duration=60 * 60 * 4)
    # print(timee.time() - time1)
    #
    # for j in range(100):
    #     mobility_model.generate_model_step(TIME_STEP)
