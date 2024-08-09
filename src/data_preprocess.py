import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import pickle
from tqdm import tqdm
import warnings
import time
from utils import haversine
warnings.filterwarnings('ignore')



def _pickle_coral_gables(df, df_user_ranked, slots_per_day, dataset_params, start_processing_time):
    df_rank_merged = df.merge(df_user_ranked, left_on=['id','lon','lat'], right_on=['id','lon_x','lat_x'], how='left')
    df_rank_merged = df_rank_merged.dropna(subset=['rank'])
    df_rank_merged['rank'] = df_rank_merged['rank'].astype(int)

    n_users = len(df['id'].unique())
    n_slots = (dataset_params['N_DAYS'])*slots_per_day
    loc_np = np.zeros(shape=(n_users, n_slots))

    one_line_time = np.tile(np.arange(slots_per_day), dataset_params['N_DAYS'])
    time_np = np.tile(one_line_time, n_users).reshape(n_users, n_slots)
    poi_np = np.zeros(shape=(n_users, n_slots))
    uid_np = np.array(df['id'].unique())
    home_rank = np.zeros(shape=(n_users,))
    work_rank = np.zeros(shape=(n_users,))

    exp_np = np.zeros(shape=(n_users, n_slots))
    is_commuter_np = np.zeros(shape=(n_users, n_slots))

    rank_dist_np = np.zeros(shape=(n_users, dataset_params['num_rank']))
    commuter_id = []
    print('---- Filling the Input Array ----')
    for i in tqdm(range(n_users)):
        user = df_rank_merged['id'].unique()[i]
        user_record = df_rank_merged[df_rank_merged['id'] == user]
        try:
            home_rank[i] = user_record[user_record['poi'] == 1]['rank'].unique()[0]
        except:
            pass
        try:
            work_rank[i] = user_record[user_record['poi'] == 2]['rank'].unique()[0]
        except:
            pass

        if user_record['poi'].unique().sum() == 3:
            commuter_id.append(i)
            is_commuter_np[i,:] = 1

        for j in range(len(user_record)):
            # filling the input array
            loc_np[i, user_record['slot'].iloc[j]] = user_record['rank'].iloc[j]
            poi_np[i, user_record['slot'].iloc[j]] = user_record['poi'].iloc[j]
            exp_np[i, user_record['slot'].iloc[j]] = user_record['exploration'].iloc[j]

        # calculating the rank distribution
        df_user = df_user_ranked[df_user_ranked['id'] == user]
        df_user['percentage'] = df_user['num_visits'] / df_user['num_visits'].sum()
        user_i_rank_dist = df_user['percentage'].to_numpy()
        if len(user_i_rank_dist) < dataset_params['num_rank']:
            user_i_rank_dist = np.pad(user_i_rank_dist, (0, dataset_params['num_rank']-len(user_i_rank_dist)))
        rank_dist_np[i] = user_i_rank_dist


    sparsity = 1 - np.count_nonzero(loc_np) / (n_users * n_slots)

    # Calculate pr1 and pr2
    loc_np_new = loc_np[:, 0:slots_per_day*(dataset_params['N_DAYS']//7)*7].reshape(loc_np.shape[0]*(dataset_params['N_DAYS']//7), slots_per_day*7)
    home_rank_new = np.repeat(home_rank, dataset_params['N_DAYS']//7)
    commuter_id = np.array(commuter_id)

    pr1 = _calc_pr(loc_np_new, home_rank_new)
    loc_np_new = loc_np[commuter_id, 0:slots_per_day*(dataset_params['N_DAYS']//7)*7].reshape(commuter_id.shape[0]*(dataset_params['N_DAYS']//7), slots_per_day*7)
    work_rank_idx = np.where(work_rank != 0)[0]
    work_rank_new = np.repeat(work_rank[work_rank_idx], dataset_params['N_DAYS']//7)
    pr2 = _calc_pr(loc_np_new, work_rank_new)

    # Save the data files
    with open(dataset_params['PICKLE_PATH_FOLDER'] + f"/{dataset_params['dataset_name']}.pkl", 'wb') as f:
        pickle.dump((loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank), f)

    # Save pr1 and pr2
    np.save(dataset_params['PICKLE_PATH_FOLDER'] + 'pr1.npy', pr1)
    np.save(dataset_params['PICKLE_PATH_FOLDER'] + 'pr2.npy', pr2)

    DATA_PARAM_FILE = open(dataset_params['PICKLE_PATH_FOLDER'] + 'params.txt', 'w')
    for k, v in dataset_params.items():
        if k in ['num_rank', 'exploration_visits', 'PICKLE_PATH_FOLDER','MIN_STAYS', 'work_threshold_num_visits', 'work_threshold_distance']:
            DATA_PARAM_FILE.write(f"{k:<{20}}: {' ' * 0}{v}\n\n")
    DATA_PARAM_FILE.write(f"{'num_users':<{20}}: {' ' * 0}{n_users}\n\n")
    DATA_PARAM_FILE.write(f"{'num_commuters':<{20}}: {' ' * 0}{commuter_id.shape[0]}\n\n")
    DATA_PARAM_FILE.write(f"{'num_non_commuters':<{20}}: {' ' * 0}{n_users - commuter_id.shape[0]}\n\n")
    DATA_PARAM_FILE.write(f"{'sparsity':<{20}}: {' ' * 0}{sparsity}\n\n")
    DATA_PARAM_FILE.close()

    df_user_ranked.to_csv(dataset_params['PICKLE_PATH_FOLDER'] + f"/{dataset_params['dataset_name']}_dict.csv")

    end_processing_time = time.time()
    print(f'Data Preprocessing Time: {round((end_processing_time - start_processing_time)/60, 2)} minutes')
    
    return loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank
    
def preprocess_coral_gables(
        params,
        lat_lon_resolution=3,
        slots_per_day=24,
    ):
    dataset_params = params['dataset_params']
    # Check whether data is already pickled:
    try:
        with open(dataset_params['PICKLE_PATH_FOLDER'] + f"/{dataset_params['dataset_name']}.pkl", 'rb') as f:
            loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank = pickle.load(f)
        return loc_np, time_np, poi_np, uid_np, exp_np, is_commuter_np, rank_dist_np, home_rank, work_rank
    except (FileNotFoundError, EOFError):
        print('This version of the CCG/LA dataset does not exist. Now creating it.')
        start_processing_time = time.time()
        pass

    # Load data from file
    print('---- Loading Data ----')
    if dataset_params['dataset_name'] == 'los_angeles':
        df = pd.read_table(dataset_params['DATA_SOURCE'], sep='\t')
    elif dataset_params['dataset_name'] == 'coral_gables':
        df = pd.read_table(data_path, sep=' ', header=None)

    df.columns = ['id', 'time', 'loc', 'old_id', 'duration', 'lon', 'lat']
    #df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize(dataset_params['TIMEZONE']) # do this for coral gables, since it's loaded in local time
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) # do this for utc timestamps not for coral gables
    df['time'] = df['time'].dt.tz_convert(dataset_params['TIMEZONE']) # do this for utc timestamps not for coral gables

    df['diff'] = df['time'] - dataset_params['START_DATE']
    df['slot'] = df['diff'].apply(lambda x: x.total_seconds() // 3600).astype(int)

    # Rasterize geo coordinates and group stays by pixel
    df['lon'] = df['lon'].apply(lambda x: round(x, lat_lon_resolution))
    df['lat'] = df['lat'].apply(lambda x: round(x, lat_lon_resolution))
    
    # Preliminary user selection based on number of stays
    print('---- Preliminary User Selection ----')
    num_stays = df.groupby('id').size().reset_index().sort_values(0, ascending=False).reset_index()
    selected_users = num_stays[num_stays[0] > dataset_params['MIN_STAYS']]
    df = df[df['id'].isin(selected_users['id'])].reset_index(drop=True)

    max_location_index = df.groupby(['id','slot'])['duration'].idxmax()
    df = df.loc[max_location_index].reset_index(drop=True)

    # Rank encodings
    print('---- Rank Encoding ----')
    df_user_rank = df.groupby(['id','lon','lat']).size().reset_index()
    df_user_rank.columns = ['id','lon','lat','num_visits']
    df_user_ranked = df_user_rank.groupby('id').apply(lambda x: x.nlargest(dataset_params['num_rank'], 'num_visits')).reset_index(drop=True)
    df_user_ranked['rank'] = df_user_ranked.groupby('id').cumcount() + 1

    # POI Ecoding
    print('---- POI Encoding ----')
    df_home = _home_detection(df, dataset_params)
    df_work = _work_detection(df, df_home, dataset_params)
    df_user_ranked = _combine_home_work(df_user_ranked, df_home, df_work, dataset_params)

    return _pickle_coral_gables(df, df_user_ranked, slots_per_day, dataset_params, start_processing_time)


def _combine_home_work(df_user_ranked, df_home, df_work, dataset_params):
    df_user_ranked = df_user_ranked.merge(df_home, on='id', how='left')
    df_user_ranked['poi'] = df_user_ranked.apply(lambda x: 1 if (x['lon_x'] == x['lon_y']) & (x['lat_x'] == x['lat_y']) else 0, axis=1)
    df_user_ranked = df_user_ranked.drop(['lon_y', 'lat_y'], axis=1)
    df_user_ranked = df_user_ranked.merge(df_work, on='id', how='left')
    df_user_ranked['poi'] = df_user_ranked.apply(lambda x: 2 if (x['lon_x'] == x['lon_y']) & (x['lat_x'] == x['lat_y']) else x['poi'], axis=1)
    df_user_ranked = df_user_ranked.drop(['lon_y', 'lat_y'], axis=1)

    df_user_ranked['exploration'] = df_user_ranked['num_visits'].apply(lambda x: 1 if x <= dataset_params['exploration_visits'] else 0).reset_index(drop=True)
    
    return df_user_ranked


def _home_detection(df, dataset_params):
    df['time'] = pd.to_datetime(df['time'])
    df_home_time = df[(df['time'].dt.hour >= 19) | (df['time'].dt.hour <= 7)]
    df_home_grouped = df_home_time.groupby(['id', 'lon', 'lat']).count().reset_index()
    idx = df_home_grouped.groupby(['id'])['time'].idxmax()
    df_home = df_home_grouped.loc[idx].reset_index(drop=True)
    df_home = df_home[['id', 'lon', 'lat']]

    return df_home

def _work_detection(df, df_home, dataset_params):
    df['time'] = pd.to_datetime(df['time'])
    df_during_work = df[(df['time'].apply(lambda x: x.hour).isin(np.arange(7, 19))) & (df['time'].apply(lambda x: x.weekday() in np.arange(0, 5)))]
    df_work_grouped = df_during_work.groupby(['id', 'lon', 'lat']).count().reset_index()

    idx = df_work_grouped.groupby('id')['time'].idxmax()
    df_work = df_work_grouped.loc[idx].reset_index(drop=True)
    df_work = df_work[df_work['time'] > dataset_params['work_threshold_num_visits']]
    df_work = df_work[['id', 'lon', 'lat']].reset_index(drop=True)

    df_work = pd.merge(df_home, df_work, on='id')
    df_work['distance'] = df_work.apply(lambda x: haversine(x['lon_x'], x['lat_x'], x['lon_y'], x['lat_y']), axis=1)
    df_work = df_work[df_work['distance'] > dataset_params['work_threshold_distance']]
    df_work = df_work[['id', 'lon_y', 'lat_y']].reset_index(drop=True)

    return df_work

def _calc_pr(loc_np_new, rank):
    count = np.zeros(loc_np_new.shape[1])
    count_non_zero = np.zeros(loc_np_new.shape[1])

    for col in range(loc_np_new.shape[1]):
        count[col] = np.sum(loc_np_new[:, col] == rank)
        count_non_zero[col] = np.sum(loc_np_new[:, col] != 0)
    pr = count / count_non_zero

    return pr


