import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def get_stay_duration(output):
    stay_duration = []
    print('Calculating Stay Duration')
    for i in tqdm(range(output.shape[0])):
        init = output[i][0]
        duration = 1
        for j in range(1, output.shape[1]):
            if output[i][j] == init:
                duration += 1
            else:
                stay_duration.append(duration)
                duration = 1
                init = output[i][j]
        
    stay_duration = np.array(stay_duration)
    return stay_duration

def get_departure_time(output, time_interval=24):
    departure_time = []
    print('Calculating Departure Time')
    for i in tqdm(range(output.shape[0])):
        init = output[i][0]
        for j in range(1, output.shape[1]):
            if output[i][j] == 0:
                pass
            elif (output[i][j] != init):
                departure_time.append(j % time_interval)
                init = output[i][j]
                
    departure_time = np.array(departure_time)
    return departure_time

def get_daily_visited_locations(output):
    numpy_output = np.array(output.cpu(), dtype=int).reshape(-1, 24)
    unique_counts = []
    print('Calculating Daily Visited Locations')
    for row in tqdm(numpy_output):
        unique_elements = np.unique(row)
        unique_counts.append(len(unique_elements))
    daily_visited_locations = np.array(unique_counts)

    return daily_visited_locations


def project_cosine_similarity(model, main_task_loss_batch, mobility_loss_batch, aux_task_loss_batch, alpha=0.5):
    main_task = main_task_loss_batch + mobility_loss_batch
    main_task.backward(retain_graph=True)
    main_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    aux_task_loss_batch.backward()
    aux_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    for name, param in model.named_parameters():
        if name in main_grads and name in aux_grads and param.grad is not None:
            main_grad = main_grads[name]
            aux_grad = aux_grads[name]
            main_grad_flat = main_grad.view(-1)
            aux_grad_flat = aux_grad.view(-1)

            cos_sim = F.cosine_similarity(main_grad_flat, aux_grad_flat, dim=0)
            if cos_sim <= 0:
                combined_grad = main_grad_flat
            elif cos_sim >0:
                combined_grad = cos_sim * aux_grad_flat * (1-alpha) + main_grad_flat * alpha
            param.grad = combined_grad.view(param.grad.shape)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def mask_commuter_weekends(main_task_prob, is_commuter, work_rank, params):
    start_weekday = params['dataset_params']['START_DATE_CG'].weekday()
    output_weekday = np.tile(np.arange(7), int(params['dataset_params']['sample_length']/7+1))[start_weekday:-7+start_weekday]
    output_weekday = np.repeat(output_weekday, params['slots_per_day'])
    weekend_slots = np.where((output_weekday == 5) | (output_weekday == 6))[0]
    weekend_slots = torch.tensor(weekend_slots).to(params['device'])

    commuter_selected = torch.where(is_commuter)[0].unique()

    work_rank_index = torch.where(work_rank != 0)[0]
    work_rank_to_fill = work_rank[work_rank_index]-1

    indices = torch.zeros_like(main_task_prob, dtype=torch.long)

    weekend_mesh, commuter_mesh = torch.meshgrid(weekend_slots, commuter_selected, indexing='ij')
    selected_values = indices[weekend_mesh, commuter_mesh, :]

    indices[weekend_mesh, commuter_mesh, work_rank_to_fill.unsqueeze(0).expand(weekend_mesh.shape[0], weekend_mesh.shape[1])] = 1
    mask = (indices == 1)
    large_number = -1000000
    main_task_prob[mask] = large_number

    return main_task_prob 