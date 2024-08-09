import torch
import torch.nn as nn
import numpy as np


class CustomLoss(nn.Module):
    def __init__(self, params):
        super(CustomLoss, self).__init__()
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.KLDivLoss()
        self.criterion3 = nn.MSELoss()
        self.params = params

        pr1 = np.load(params['dataset_params']['PICKLE_PATH_FOLDER'] + 'pr1.npy')*0.55 + 0.4
        pr1 = torch.tensor(np.tile(pr1, int(params['dataset_params']['sample_length']/7)), dtype=torch.float32).to(torch.device(params['device']))
        self.pr1 = pr1/pr1.sum()

        pr2 = np.load(params['dataset_params']['PICKLE_PATH_FOLDER'] + 'pr2.npy')
        pr2 = torch.tensor(np.tile(pr2, int(params['dataset_params']['sample_length']/7)), dtype=torch.float32).to(torch.device(params['device']))
        self.pr2 = pr2/pr2.sum()

        self.pt_weekday = torch.tensor([0.00494407, 0.00253384, 0.00129782, 0.00129782, 0.00568568,
                                0.02064149, 0.05030591, 0.08361659, 0.06501452, 0.04319881,
                                0.04573265, 0.0495643 , 0.06890798, 0.05537359, 0.06044126,
                                0.07403745, 0.07885792, 0.09022928, 0.06927878, 0.04703047,
                                0.03473209, 0.02373154, 0.01563562, 0.00791051])

        self.pt_weekend = torch.tensor([0.00538377, 0.00324406, 0.0021397 , 0.00207068, 0.00200166,
                                0.00752347, 0.02056875, 0.03582275, 0.06205135, 0.06853948,
                                0.07654611, 0.07302595, 0.08786582, 0.07061016, 0.07205964,
                                0.06957482, 0.07302595, 0.0658476 , 0.06426008, 0.04548592,
                                0.03402816, 0.02871342, 0.01815295, 0.01145776])

    def forward(self, main_output, loc_y, aux_output, loc_y_exploration, is_commuter, rank_dist, loc_x, home_rank, work_rank, train=True):
        L1, L_aux = self._single_prediction_loss(main_output, loc_y, aux_output, loc_y_exploration, loc_x)
        L2 = self._rank_distribution_loss(main_output, rank_dist)
        L3, standardized_prob = self._pt_loss(main_output)
        L4 = self._pr_loss(standardized_prob, is_commuter, home_rank, work_rank)

        main_task_loss = L1 * self.params['lambda_1']
        mobility_regularization = L2 * self.params['lambda_2'] + L3 * self.params['lambda_3'] + L4 * self.params['lambda_4']
        auxilary_task_loss = L_aux * self.params['lambda_1']

        return main_task_loss, mobility_regularization, auxilary_task_loss


    def _single_prediction_loss(self, main_output, loc_y, aux_output, loc_y_exploration, loc_x):
        masked = torch.where(loc_x.T != 0)
        masked_y = loc_y[masked[1], masked[0], :]
        masked_y_exploration = loc_y_exploration[masked[1], masked[0], :]

        masked_main_output = main_output[masked[0], masked[1],:]
        masked_aux_output = aux_output[masked[0], masked[1],:]

        # Single prediction cross entropy loss
        L1 = self.criterion1(torch.exp(masked_main_output), masked_y)
        L_aux = self.criterion1(torch.exp(masked_aux_output), masked_y_exploration)

        return L1, L_aux

    def _rank_distribution_loss(self, main_output, rank_dist):
        # Rank distribution kl-divergence loss
        standardized_prob = torch.exp(main_output[:,:,1:]).sum(axis=0)
        standardized_prob = standardized_prob/standardized_prob.sum(axis=1, keepdim=True)
        L2 = self.criterion2(torch.log(standardized_prob), rank_dist)

        return L2

    def _pt_loss(self, main_output):
        start_weekday = self.params['dataset_params']['START_DATE'].weekday()
        output_weekday = np.tile(np.arange(7), int(self.params['dataset_params']['sample_length']/7+1))[start_weekday:-7+start_weekday]
        output_weekday = np.repeat(output_weekday, self.params['slots_per_day'])
        weekend_slots = np.where((output_weekday == 5) | (output_weekday == 6))[0]

        # pt MSE loss
        transformed = main_output.reshape(self.params['dataset_params']['sample_length'], self.params['slots_per_day'], main_output.shape[1], self.params['n_locations']+1)
        transformed = torch.exp(transformed[:,:,:,1:])
        standardized_prob = transformed / transformed.sum(axis=-1, keepdim=True)

        sse_weekday = torch.zeros(self.params['slots_per_day'])
        sse_weekend = torch.zeros(self.params['slots_per_day'])
        for i in range(transformed.shape[1]-1):
            current = standardized_prob[:,i, :, :]
            next_one = standardized_prob[:,i+1, :, :]
            if i in weekend_slots:
                sse_weekend[i] = self.criterion3(current, next_one)
            else:
                sse_weekday[i] = self.criterion3(current, next_one)

        L3 = (sse_weekday/self.pt_weekday).sum() + (sse_weekend/self.pt_weekend).sum()

        return L3, standardized_prob

    def _pr_loss(self, standardized_prob, is_commuter, home_rank, work_rank):
        # pr1 KL divergence loss
        a = standardized_prob.reshape(self.params['slots_per_day']*self.params['dataset_params']['sample_length'], standardized_prob.shape[2], standardized_prob.shape[3])
        batch_indices = torch.arange(a.size(1)).unsqueeze(0).repeat(a.size(0), 1)
        home_rank_indices = home_rank.unsqueeze(0).repeat(a.size(0), 1)
        a = a[torch.arange(a.size(0)).unsqueeze(1), batch_indices, home_rank_indices]
        a = a.mean(axis=1) / a.mean(axis=1).sum()
        L4a = self.criterion2(torch.log(a), self.pr1)

        # pr2 KL divergence loss
        commuter_idx = torch.where(is_commuter)[0].unique()
        commuter_standardized_prob = standardized_prob[:,:,commuter_idx,:]
        work_rank = work_rank[commuter_idx]

        b = commuter_standardized_prob.reshape(self.params['slots_per_day']*self.params['dataset_params']['sample_length'], commuter_standardized_prob.shape[2], commuter_standardized_prob.shape[3])
        batch_indices = torch.arange(b.size(1)).unsqueeze(0).repeat(b.size(0), 1)
        work_rank_indices = work_rank.unsqueeze(0).repeat(b.size(0), 1)
        b = b[torch.arange(b.size(0)).unsqueeze(1), batch_indices, work_rank_indices]
        b = b.mean(axis=1) / b.mean(axis=1).sum()
        L4b = self.criterion2(torch.log(b), self.pr2)

        L4 = L4a + L4b
        return L4



