import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import data_preprocess
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm
from time import time
import pickle

class Dataset(TensorDataset):

    path_coral_gables = 'data/coral_gables/raw.txt'
    info_coral_gables = {'path': path_coral_gables}
    path_los_angeles = 'data/los_angeles/raw.txt'
    info_los_angeles = {'path': path_los_angeles}

    dataset_names = {'coral_gables'}
    info_dict = {'coral_gables': info_coral_gables}
    info_dict['los_angeles'] = info_los_angeles

    def get_dataset_names():
        print(Dataset.dataset_names)

    def _setup(
        self,
        num_days,
        loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np,
        home_rank, work_rank,
        min_locs_per_day=4,
        expansion=False
    ):
        if expansion:
            with open(f"trained_models/{self.params['model_name']}/data/validation_set.pkl", 'rb') as f:
                loc_x_val, time_x_val, poi_x_val, uid_val, loc_y_val, loc_y_exploration_val, is_commuter_val, rank_dist_val, selected_home_rank_val, selected_work_rank_val = pickle.load(f)
            return loc_x_val, time_x_val, poi_x_val, uid_val, loc_y_val, loc_y_exploration_val, is_commuter_val, rank_dist_val, selected_home_rank_val, selected_work_rank_val
        
        # Check whether data is already preloaded:
        try:
            with open(f"trained_models/{self.params['model_name']}/data/training_set.pkl", 'rb') as f:
                loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, selected_home_rank, selected_work_rank = pickle.load(f)
            return loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, selected_home_rank, selected_work_rank
        except (FileNotFoundError, EOFError):
            pass


        min_slot = min_locs_per_day * num_days
        n_days = loc_np.shape[1] / self.slots_per_day
        if n_days % 1 != 0:
            raise ValueError(
                f'total slots per user is {loc_np.shape[1]}, ' +
                'which is not divisible by slots_per_day, which is {self.slots_per_day}.'
            )
        n_days = int(n_days)
        n_users = loc_np.shape[0]

        length_of_sequence = int(self.slots_per_day * num_days)
        n_weeks =int(n_days // (length_of_sequence/self.slots_per_day))
        n_sample = int(n_users*n_weeks)
        n_selected = n_weeks*length_of_sequence
        
        loc_np = loc_np[:,:n_selected].reshape(n_sample, length_of_sequence)
        time_np = time_np[:,:n_selected].reshape(n_sample, length_of_sequence)
        poi_np = poi_np[:,:n_selected].reshape(n_sample, length_of_sequence)
        uid_np = np.repeat(uid_np, n_weeks)
        exp_np = exp_np[:,:n_selected].reshape(n_sample, length_of_sequence)
        is_commuter_np = is_commuter_np[:,:n_selected].reshape(n_sample, length_of_sequence)
        rank_dist_np = np.repeat(rank_dist_np, n_weeks, axis=0)
        home_rank_np = np.repeat(home_rank, n_weeks)
        work_rank_np = np.repeat(work_rank, n_weeks)

        loc_cnt = np.count_nonzero(loc_np, axis=1)
        selected_index = np.where((loc_cnt >= min_slot))[0]

        loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, selected_home_rank, selected_work_rank = \
            self._parse_input(selected_index, loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank_np, work_rank_np)

        selected_index_val = np.where((loc_cnt < min_slot))[0]
        loc_x_val, time_x_val, poi_x_val, uid_val, loc_y_val, loc_y_exploration_val, is_commuter_val, rank_dist_val, selected_home_rank_val, selected_work_rank_val = \
            self._parse_input(selected_index_val, loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank_np, work_rank_np)
        
        with open(f"trained_models/{self.params['model_name']}/data/validation_set.pkl", 'wb') as f:
            pickle.dump((loc_x_val, time_x_val, poi_x_val, uid_val, loc_y_val, loc_y_exploration_val, is_commuter_val, rank_dist_val, selected_home_rank_val, selected_work_rank_val), f)
        with open(f"trained_models/{self.params['model_name']}/data/training_set.pkl", 'wb') as f:
            pickle.dump((loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, selected_home_rank, selected_work_rank), f)

        return loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, selected_home_rank, selected_work_rank

    def _parse_input(self, selected_index, loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank):
        selected_loc = loc_np[selected_index]
        selected_time = time_np[selected_index]
        selected_poi = poi_np[selected_index]
        selected_uid = uid_np[selected_index]
        selected_exp = exp_np[selected_index]
        selected_is_commuter = is_commuter_np[selected_index]
        selected_rank_dist = rank_dist_np[selected_index]
        selected_work_rank = work_rank[selected_index]
        selected_home_rank = home_rank[selected_index]

        loc_x = torch.tensor(selected_loc).long()
        time_x = torch.tensor(selected_time).long()
        poi_x = torch.tensor(selected_poi).long()
        uid = torch.tensor(selected_uid)
        loc_y = torch.zeros(size=(loc_x.shape[0], loc_x.shape[1], self.num_rank+1))
        loc_y_exploration = torch.zeros(size=(loc_x.shape[0], loc_x.shape[1], 2))
        is_commuter = torch.tensor(selected_is_commuter).long()
        selected_home_rank = torch.tensor(selected_home_rank).long()
        selected_work_rank = torch.tensor(selected_work_rank).long()

        for i in tqdm(range(selected_loc.shape[0])):
            for j in range(selected_loc.shape[1]):
                if selected_loc[i, j] != -1:
                    loc_y[i, j, int(selected_loc[i, j])] = 1
                    loc_y_exploration[i, j, int(selected_exp[i, j])] = 1

        rank_dist = torch.tensor(selected_rank_dist)

        return loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, selected_home_rank, selected_work_rank

    

    def _inherit(self, parent, users):
        self.dataset_name = parent.dataset_name
        self.lat_lon_resolution = parent.lat_lon_resolution
        self.slots_per_day= parent.slots_per_day
        self.num_rank = parent.num_rank
        self.info = Dataset.info_dict[self.dataset_name]
        self.sample_length = parent.sample_length
        self.n_locations = parent.n_locations
        self.n_days = parent.n_days
        if self.dataset_name == 'coral_gables':
            self.loc_x, self.time_x, self.poi_x, self.uid, self.loc_y, self.loc_y_exploration, self.is_commuter, self.rank_dist, self.home_rank, self.work_rank = parent[users]
            super.__init__(self.loc_x, self.time_x, self.poi_x, self.uid, self.loc_y, self.loc_y_exploration, self.is_commuter, self.rank_dist, self.home_rank, self.work_rank)
  
    def __init__(
            self,
            params,
            expansion=False,
            lat_lon_resolution=3,
            slots_per_day=24,
    ):
        self.params = params
        self.dataset_name = params['dataset_params']['dataset_name']
        self.sample_length = params['dataset_params']['sample_length']
        self.num_rank = params['dataset_params']['num_rank']


        self.lat_lon_resolution = lat_lon_resolution
        self.slots_per_day=slots_per_day
        self.info = Dataset.info_dict[self.dataset_name]

        print('---- Preprocessing data ----')
        if self.dataset_name in ['coral_gables', 'los_angeles']:
            loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank = data_preprocess.preprocess_coral_gables(params)
        else:
            raise ValueError(
                f'{self.dataset_name} not among available datasets. Please choose one from {Dataset.dataset_names}'
            )

        loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, home_rank, work_rank = \
            self._setup(self.sample_length, loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank, expansion=expansion)

        self.loc_x, self.time_x, self.poi_x, self.uid, self.loc_y, self.loc_y_exploration, self.is_commuter, self.rank_dist, self.home_rank, self.work_rank = \
            loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, home_rank, work_rank

        self.n_days = self.loc_y.shape[1]
        self.n_users = self.loc_y.shape[0]
        self.n_locations = self.num_rank

        print('---- Data preprocessing Completed ----')
        super().__init__(self.loc_x, self.time_x, self.poi_x, self.uid, self.loc_y, self.loc_y_exploration, self.is_commuter, self.rank_dist, self.home_rank, self.work_rank)
        