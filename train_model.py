import os
import pickle
import torch
import torch.nn as nns
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import pytz
from datetime import datetime
sys.path.insert(0, 'src')
from src.dataset import Dataset
from src.train import train_model
from src.models.DTGTransformer import DTGTransformer
from src.generate import generate
from src.viz import viz_loss, viz_stay_duration_dist, viz_departure_time_dist, viz_num_daily_loc_dist
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model and Dataset Parameters')

    parser.add_argument('--model_name', type=str, default='los_angeles_0808', help='Name of the model')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lambda_1', type=float, default=1, help='Weight for the main task')
    parser.add_argument('--lambda_2', type=float, default=25, help='Weight for Regularizer 1')
    parser.add_argument('--lambda_3', type=float, default=0.1, help='Weight for Regularizer 2')
    parser.add_argument('--lambda_4', type=float, default=0.05, help='Weight for Regularizer 3')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--mask_commuter_weekend', default=False, help='Mask commuter weekend')
    parser.add_argument('--aux_learning', default=True, help='Enable auxiliary learning')

    # Dataset parameters
    parser.add_argument('--num_rank', type=int, default=45, help='Number of ranks, Lc')
    parser.add_argument('--sample_length', type=int, default=14, help='Sample length in days')
    parser.add_argument('--exploration_visits', type=int, default=4, help='Exploration visits, nc')
    parser.add_argument('--dataset_name', type=str, default='los_angeles', help='Name of the dataset')
    parser.add_argument('--PICKLE_PATH_FOLDER', type=str, default='data/los_angeles/0808/', help='Path to the pickle folder')
    parser.add_argument('--MIN_STAYS', type=int, default=50, help='Minimum number of stays')
    parser.add_argument('--START_DATE', type=str, default='2019-01-01 00:07:00', help='Start date (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--DATA_SOURCE', type=str, default='data/los_angeles/raw.txt', help='Data source path')
    parser.add_argument('--N_DAYS', type=int, default=31, help='Number of days')
    parser.add_argument('--work_threshold_num_visits', type=int, default=8, help='Work threshold number of visits')
    parser.add_argument('--work_threshold_distance', type=float, default=0.5, help='Work threshold distance')

    args = parser.parse_args()

    args.START_DATE = datetime.strptime(args.START_DATE, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.timezone('US/Pacific'))
    args.device = torch.device(args.device)

    # Convert Namespace to dictionaries for model and dataset parameters
    model_params = {
        'model_name': args.model_name,
        'num_epochs': args.num_epochs,
        'lambda_1': args.lambda_1,
        'lambda_2': args.lambda_2,
        'lambda_3': args.lambda_3,
        'lambda_4': args.lambda_4,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'batch_size': args.batch_size,
        'mask_commuter_weekend': args.mask_commuter_weekend,
        'aux_learning': args.aux_learning
    }

    dataset_params = {
        'num_rank': args.num_rank,
        'sample_length': args.sample_length,
        'exploration_visits': args.exploration_visits,
        'dataset_name': args.dataset_name,
        'PICKLE_PATH_FOLDER': args.PICKLE_PATH_FOLDER,
        'MIN_STAYS': args.MIN_STAYS,
        'START_DATE': args.START_DATE,
        'DATA_SOURCE': args.DATA_SOURCE,
        'N_DAYS': args.N_DAYS,
        'work_threshold_num_visits': args.work_threshold_num_visits,
        'work_threshold_distance': args.work_threshold_distance
    }

    return model_params, dataset_params



def main(params):
    if not os.path.exists(params['dataset_params']['PICKLE_PATH_FOLDER']):
        os.makedirs(params['dataset_params']['PICKLE_PATH_FOLDER'])

    # Model Paths
    MODEL_PATH = f"trained_models/{params['model_name']}/model.pt"
    PARAMS_PATH = f"trained_models/{params['model_name']}/params.txt"

    if not os.path.exists(f"trained_models/{params['model_name']}"):
        os.makedirs(f"trained_models/{params['model_name']}")
        os.makedirs(f"trained_models/{params['model_name']}/viz/")
        os.makedirs(f"trained_models/{params['model_name']}/loss/")
        os.makedirs(f"trained_models/{params['model_name']}/simulation/")
        os.makedirs(f"trained_models/{params['model_name']}/data/")

    dataset = Dataset(params)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [1, 0, 0])
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model=DTGTransformer(
        dataset.n_locations, 
        dataset.slots_per_day,
    )

    params['n_locations'] = dataset.n_locations
    params['slots_per_day'] = dataset.slots_per_day
    
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    model.to(params['device'])

    train_single_pred_loss_long, train_mobility_loss_long, aux_loss_long = train_model(model, train_loader, val_loader, optimizer, params)

    generated_output, uid = generate(model, dataset, params)

    expansion_dataset = Dataset(params, expansion=True)
    expansion_output, expansion_uid = generate(model, expansion_dataset, params)

    # Saving Outputs
    torch.save(generated_output, f"trained_models/{params['model_name']}/simulation/generated_output.pt")
    torch.save(uid, f"trained_models/{params['model_name']}/simulation/uid.pt")

    torch.save(expansion_output, f"trained_models/{params['model_name']}/simulation/expansion_output.pt")
    torch.save(expansion_uid, f"trained_models/{params['model_name']}/simulation/expansion_uid.pt")

    # Saving Losses
    with open(f"trained_models/{params['model_name']}/loss/loss.pkl", "wb") as f:
        pickle.dump((train_single_pred_loss_long, train_mobility_loss_long, aux_loss_long), f)
    viz_loss(train_single_pred_loss_long, train_mobility_loss_long, aux_loss_long, model_name=params['model_name'])

    # Visualizing Mobility Patterns
    print('---- Visualizing Mobility Patterns ----')
    start_time = time.time()
    viz_stay_duration_dist(generated_output, dataset.loc_x, model_name=params['model_name'])
    viz_departure_time_dist(generated_output, dataset.loc_x, model_name=params['model_name'])
    viz_num_daily_loc_dist(generated_output, dataset.loc_x, model_name=params['model_name'])

    # viz_stay_duration_dist(expansion_output, expansion_dataset.loc_x, model_name=params['model_name'], expansion=True)
    # viz_departure_time_dist(expansion_output, expansion_dataset.loc_x, model_name=params['model_name'], expansion=True)
    # viz_num_daily_loc_dist(expansion_output, expansion_dataset.loc_x, model_name=params['model_name'], expansion=True)

    end_time = time.time()
    print(f'Visualization Time: {round((end_time - start_time)/60, 2)} minutes')

    # Saving Model
    torch.save(model.state_dict(), MODEL_PATH)
    param_file = open(PARAMS_PATH, "w")
    for k, v in params.items():
        if k == 'dataset_params':
            for k1, v1 in v.items():
                param_file.write(f"{k1:<{20}}: {' ' * 0}{v1}\n\n")
        else:
            param_file.write(f"{k:<{20}}: {' ' * 0}{v}\n\n")
    param_file.close()

if __name__ == "__main__":
    params, dataset_params = parse_args()
    dataset_params['TIMEZONE'] = 'US/Pacific'
    params['dataset_params'] = dataset_params
    main(params)
    print('---- Training Completed ----')
    print('Model Name:  ', params['model_name'])



