from tqdm import tqdm
from loss import CustomLoss
from utils import project_cosine_similarity
import torch
import torch.nn.functional as F
import numpy as np
import time
        
def run_epoch(model, data_loader, LossFunction, optimizer, params, train=True):
    device = params['device']
    regular_loss = 0.0
    mobility_loss = 0.0
    auxilary_loss = 0.0

    for loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, home_rank, work_rank in tqdm(data_loader):
        loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, home_rank, work_rank = \
            loc_x.to(device), time_x.to(device), poi_x.to(device), uid.to(device), \
            loc_y.to(device), loc_y_exploration.to(device), is_commuter.to(device), rank_dist.to(device), \
            home_rank.to(device), work_rank.to(device)
        
        optimizer.zero_grad()

        main_output, aux_output = model(loc_x, time_x, poi_x, is_commuter)
        main_task_loss_batch, mobility_loss_batch, aux_task_loss_batch = LossFunction(main_output, loc_y, aux_output, loc_y_exploration, is_commuter, rank_dist, loc_x, home_rank, work_rank)

        if train:
            #https://vivien000.github.io/blog/journal/learning-though-auxiliary_tasks.html
            if params['aux_learning']:
                project_cosine_similarity(model, main_task_loss_batch, mobility_loss_batch, aux_task_loss_batch)
            else:
                main_task = main_task_loss_batch + mobility_loss_batch
                main_task.backward()
            optimizer.step()
        
        regular_loss += main_task_loss_batch.item()
        mobility_loss += mobility_loss_batch.item()
        auxilary_loss += aux_task_loss_batch.item()
        
    return regular_loss, mobility_loss, auxilary_loss

def train_model(model, train_loader, val_loader, optimizer, params):
    MyLoss = CustomLoss(params)
    model.train()
    print('---- Training Starts ----')
    start_time = time.time()

    main_task_loss_epoch = []
    mobility_regularization_epoch = []
    aux_task_loss_epoch = []

    for epoch in range(params['num_epochs']): 
        model.train()
        train_main_task_loss, train_mobility_loss, train_aux_loss = run_epoch(model, train_loader, MyLoss, optimizer, params)

        main_task_loss_epoch.append(train_main_task_loss)
        mobility_regularization_epoch.append(train_mobility_loss)
        aux_task_loss_epoch.append(train_aux_loss)

        model.eval()
        val_main_task_loss, val_mobility_loss, val_aux_loss = run_epoch(model, val_loader, MyLoss, optimizer, params, train=False)

        print(f"Epoch [{epoch+1}/{params['num_epochs']:2}] \
            Single Prediction Loss: {train_main_task_loss/len(train_loader):<15.10f} \
            Mobility Loss: {train_mobility_loss/len(train_loader):<15.10f} \
            Aux Loss: {train_aux_loss/len(train_loader):<15.10f} \
            Training Loss: {(train_main_task_loss+train_mobility_loss)/len(train_loader):<15.10f} \
            Validation Loss: {(val_main_task_loss+val_mobility_loss)/len(val_loader):<15.10f}")

    end_time = time.time()
    print(f'Training Time: {round((end_time - start_time)/60, 2)} minutes')

    return np.array(main_task_loss_epoch), np.array(mobility_regularization_epoch), np.array(aux_task_loss_epoch)
    