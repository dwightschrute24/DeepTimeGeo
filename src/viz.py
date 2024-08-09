import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from utils import get_stay_duration, get_departure_time, get_daily_visited_locations

sns.set(style ="whitegrid", font_scale=1.5)
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
color_palette = ['#93003a', '#00429d', '#93c4d2', '#6ebf7c']

def viz_loss(single_loss, mobility_loss, aux_loss, model_name):
    x = np.arange(1, len(single_loss)+1, dtype=int)
    total_loss = single_loss + mobility_loss

    fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=200)
    sns.lineplot(x=x, y=single_loss, ax=ax[0], color=color_palette[2], marker='o', label='Cross Entropy Loss')
    sns.lineplot(x=x, y=mobility_loss, ax=ax[1], color=color_palette[1], marker='o', label='KL Divergence Loss')
    sns.lineplot(x=x, y=aux_loss, ax=ax[2], color=color_palette[3], marker='o', label='Auxiliary Task Loss')
    sns.lineplot(x=x, y=total_loss, ax=ax[3], color=color_palette[0], marker='o', label='Total Loss')

    ax[0].set(xticklabels=[], title='Single Prediction Training Loss')
    ax[1].set(xticklabels=[], title='Mobility Training Loss')
    ax[2].set(xticklabels=[], title='Auxiliary Task Training Loss')
    ax[3].set(title='Total Training Loss')

    for i in range(4):
        ax[i].grid(True, alpha=0.25)
        ax[i].set(xlim=(1, len(x)))

    fig.text(0.03, 0.5, 'Loss', ha='center', va='center', rotation='vertical', size=20)
    fig.text(0.5, 0.02, 'Epoch', ha='center', va='center', size=20)
    plt.tight_layout()

    plt.savefig(f'trained_models/{model_name}/viz/loss.png')


def viz_stay_duration_dist(generated_output, original_input, model_name, expansion=False):
    stay_duration_calc = get_stay_duration(generated_output.cpu())
    stay_duration_calc_input = get_stay_duration(original_input.cpu())

    hist, bins = np.histogram(stay_duration_calc, bins=np.arange(0,25,1))
    stay_duration = hist/hist.sum()

    hist_input, bins_input = np.histogram(stay_duration_calc_input, bins=np.arange(0,25,1))
    stay_duration_input = hist_input/hist_input.sum()

    fig1, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot(np.arange(0,len(stay_duration)), stay_duration, 
                      marker='o', linestyle='-', color=color_palette[1], label='Generated')
    ax.plot(np.arange(0,len(stay_duration_input)), stay_duration_input, 
                      marker='o', linestyle='-', color=color_palette[0], label='Input')

    ax.set(xlabel='Waiting Time $\Delta t$ in Hours', 
        ylabel='Probability Density $P(\Delta t)$',
        title='Stay Duration Distribution',
        xticks=np.arange(0,28,4),
        xlim=(0,24))
    ax.set_yscale('log')
    ax.grid(which='both', color='#dddddd')
    ax.legend()
    plt.tight_layout()
    if expansion:
        plt.savefig(f'trained_models/{model_name}/viz/expansion_stay_duration.png')
    else:
        plt.savefig(f'trained_models/{model_name}/viz/original_stay_duration.png')

    
def viz_departure_time_dist(generated_output, original_input, model_name, expansion=False):
    departure_time_calc = get_departure_time(generated_output.cpu())
    departure_time_calc_input = get_departure_time(original_input.cpu())

    hist, bins = np.histogram(departure_time_calc, bins=np.arange(0,25,1))
    departure_time = hist/hist.sum()

    hist_input, bins_input = np.histogram(departure_time_calc_input, bins=np.arange(0,25,1))
    departure_time_input = hist_input/hist_input.sum()

    fig1, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot(np.arange(len(departure_time)), departure_time, 
                      marker='o', linestyle='-', color=color_palette[1], label='Generated')
    ax.plot(np.arange(len(departure_time_input)), departure_time_input, 
                      marker='o', linestyle='-', color=color_palette[0], label='Input')

    ax.set(xlabel='Depature Time $t$', 
        ylabel='Probability Density $P(t)$',
        title='Departure Time Distribution',
        xticks=np.arange(0,28,4),
        xlim=(0,24))
    ax.grid(which='both', color='#dddddd')
    ax.legend()
    plt.tight_layout()
    if expansion:
        plt.savefig(f'trained_models/{model_name}/viz/expansion_depature_time.png')
    else:
        plt.savefig(f'trained_models/{model_name}/viz/original_depature_time.png')

def viz_num_daily_loc_dist(generated_output, original_input, model_name, expansion=False):
    daily_visited_locations = get_daily_visited_locations(generated_output.cpu())
    daily_visited_locations_input = get_daily_visited_locations(original_input.cpu())

    hist, _ = np.histogram(daily_visited_locations, bins=np.arange(2,14,1))
    num_loc = hist/hist.sum()

    hist_input, _ = np.histogram(daily_visited_locations_input, bins=np.arange(2,14,1))
    num_loc_input = hist_input/hist_input.sum()

    fig1, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot(np.arange(len(num_loc)), num_loc, 
                      marker='o', linestyle='-', color=color_palette[1], label='Generated')
    ax.plot(np.arange(len(num_loc_input)), num_loc_input,
                      marker='o', linestyle='-', color=color_palette[0], label='Input')

    ax.set(xlabel='Number of Daily Locations, $N$', 
        ylabel='Probability Density $P(N)$',
        title='Daily Visited Locations Distribution',
        xticks=np.arange(2,14,2),
        xlim=(0,12))
    ax.set_yscale('log')
    ax.grid(which='both', color='#dddddd')
    ax.legend()
    plt.tight_layout()
    if expansion:
        plt.savefig(f'trained_models/{model_name}/viz/expansion_daily_num_locations.png')
    else:
        plt.savefig(f'trained_models/{model_name}/viz/original_daily_num_locations.png')