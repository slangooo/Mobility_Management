from src.parameters import DEFAULT_A3_SINR_OFFSET, DEAFULT_A3_SINR_HYST, DEFAULT_TTT_MS, DEFAULT_A3_INDIV_OFFSET,\
DEFAULT_HO_SINR_THRESHOLD
from src.math_tools import lin2db
import numpy as np


class A3Event:
    def __init__(self, n_users, n_stations):
        self.offset = DEFAULT_A3_SINR_OFFSET
        self.hysteresis = DEAFULT_A3_SINR_HYST
        self.ttt = DEFAULT_TTT_MS
        self.n_users = n_users
        self.a3_cells_indiv_offset = DEFAULT_A3_INDIV_OFFSET * np.ones((n_stations))
        self.a3_triggered_matrix = np.zeros((n_users, n_stations), dtype=bool)
        self.a3_ttt_matrix = np.zeros((n_users, n_stations), dtype=int)
        self.a3_mr_matrix = np.zeros((n_users, n_stations), dtype=float)

    def check_users_state(self, ue_sinrs, ue_associations, timestep):
        self.a3_mr_matrix.fill(0)
        for ue_idx, serving_bs_idx in enumerate(ue_associations):
            serving_sinr = ue_sinrs[ue_idx, serving_bs_idx]
            for target_bs_idx, target_sinr in enumerate(ue_sinrs[ue_idx]):
                if target_bs_idx == serving_bs_idx:
                    continue
                if not self.a3_triggered_matrix[ue_idx, target_bs_idx]:
                    if self.test_trigger_condition(serving_sinr, target_sinr,
                                                   self.a3_cells_indiv_offset[serving_bs_idx],
                                                   self.a3_cells_indiv_offset[target_bs_idx]):
                        self.a3_triggered_matrix[ue_idx, target_bs_idx] = True
                        self.a3_ttt_matrix[ue_idx, target_bs_idx] = self.ttt
                    continue

                elif self.test_cancel_condition(serving_sinr, target_sinr, self.a3_cells_indiv_offset[serving_bs_idx],
                                                self.a3_cells_indiv_offset[target_bs_idx]):
                    self.a3_triggered_matrix[ue_idx, target_bs_idx] = False
                    self.a3_mr_matrix[ue_idx, target_bs_idx] = 0
                    self.a3_ttt_matrix[ue_idx, target_bs_idx] = 0
                    continue

                elif self.a3_ttt_matrix[ue_idx, target_bs_idx] > 0:
                    self.a3_ttt_matrix[ue_idx, target_bs_idx] -= timestep
                    if self.a3_ttt_matrix[ue_idx, target_bs_idx] < 0:
                        self.a3_ttt_matrix[ue_idx, target_bs_idx] = 0
                    else:
                        continue
                self.a3_mr_matrix[ue_idx, target_bs_idx] = target_sinr
                print(f'UE {ue_idx} is sending A3 report with target cell {target_bs_idx}')

    def perform_ho_from_mr(self, users, base_stations):
        for ue_idx, user in enumerate(users):
            max_arg = self.a3_mr_matrix[ue_idx].argmax()
            if DEFAULT_HO_SINR_THRESHOLD < self.a3_mr_matrix[ue_idx, max_arg]:
                print(f"Starting HO of UE {ue_idx} to target cell {max_arg}")
                # TODO: make delay in HO
                user.rf_transceiver.serving_bs = base_stations[max_arg]
                self.a3_mr_matrix[ue_idx].fill(0)
                self.a3_triggered_matrix[ue_idx].fill(False)

    def test_trigger_condition(self, serving_sinr, target_sinr, serving_indiv_offset, target_indiv_offset):
        return lin2db(target_sinr) + target_indiv_offset - self.hysteresis > lin2db(
            serving_sinr) + serving_indiv_offset + self.offset

    def test_cancel_condition(self, serving_sinr, target_sinr, serving_indiv_offset, target_indiv_offset):
        return lin2db(target_sinr) + target_indiv_offset + self.hysteresis < lin2db(
            serving_sinr) + serving_indiv_offset + self.offset

# class HoEventsController:
#     events = []
#
#     def __init__(self, n_users, n_stations):
#         self.time_ms = 0
#         self.events.append(A3Event(n_users, n_stations))
#         self.n_users = n_users
#         self.n_stations = n_stations
