import torch
from torch.utils.data import DataLoader
from utils import mask_commuter_weekends

def generate(model, dataset, params):
    print('---- Generating ----')
    combined_outputs = torch.zeros((1, dataset.loc_x.shape[1])).to(params['device'])
    eval_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    list_of_uids = []

    for loc_x, time_x, poi_x, uid, loc_y, loc_y_exploration, is_commuter, rank_dist, home_rank, work_rank in eval_loader:
            loc_x, time_x, poi_x, uid, loc_y, is_commuter, rank_dist, work_rank = \
                loc_x.to(params['device']), time_x.to(params['device']), poi_x.to(params['device']), uid.to(params['device']), \
                    loc_y.to(params['device']), is_commuter.to(params['device']), rank_dist.to(params['device']), work_rank.to(params['device'])

            model.eval()
            main_task_prob, aux_task_prob = model(loc_x, time_x, poi_x, is_commuter)
            # Masking commuter weekends
            if params['mask_commuter_weekend']:
                main_task_prob = mask_commuter_weekends(main_task_prob, is_commuter, work_rank, params)

            main_task_output = torch.argmax(main_task_prob[:,:,1:], dim=2)+1
            # if params['mask_commuter_weekends']:
            #     main_task_output[torch.where(main_task_output == 0)] = 1

            combined_outputs = torch.concatenate((combined_outputs, main_task_output.T), axis=0)
            list_of_uids = list_of_uids + uid.tolist()

    combined_outputs = combined_outputs[1:,:]
    list_of_uids = torch.tensor(list_of_uids)

    return combined_outputs, list_of_uids