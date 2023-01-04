import os
import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--score_type", type=str, required=True)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0, 7))

import sys

sys.path.append("/")

from data import get_data
from sampling import ddim_score_update2, pc_sampler2, ode_score_update, ode_sampler
from util import dict2namespace
from train import train
from torchlevy.approx_score import _get_c_t

import ot

from ax.service.ax_client import AxClient


def get_config(score_type):
    config_dict = {
        # "dataset": "gaussian",  # "gaussian", "mixture of gaussian"
        "beta_min": 0.1,
        "beta_max": 10,
        "max_iter": 20,
        "score_type": score_type,  # 'rectified_with_hyperparam', 'real+linear'
        "score_coeff1": None,
        "score_coeff2": None,
        "noise_type": 'clamp',  # 'levy', 'brownian', 'clamp', 'resampling'
        "loss_type": 'l2',  # 'l2', 'smooth_l1', 'l1'
        "clamp_size": 10,  # 1000, 3, 5, 10
        "init_clamp_size": 10,  # 1000, 3, 5
    }
    return dict2namespace(config_dict)


def distance_func(alpha, score_type, hpo_config):
    config = get_config(score_type)
    if score_type in ['rectified+linear', 'real+linear']:
        config.score_coeff1 = hpo_config["score_coeff1"]
        config.score_coeff2 = hpo_config["score_coeff2"]
    elif score_type == 'rectified_with_hyperparam':
        config.c_hat = hpo_config["c_hat"]
        config.beta_hat = hpo_config["beta_hat"]
    elif score_type == "professor_suggestion":
        config.change_point = hpo_config["change_point"]
        config.c_hat = hpo_config["c_hat"]
        config.beta_hat = alpha - 1
    else:
        raise NotImplementedError()

    total_distance = 0
    num_iter = 10
    for i in range(num_iter):
        for dataset in ["gaussian", "two moon", "swiss roll"]:
            train_data, valid_data, test_data = get_data(dataset)
            valid_data = valid_data.cpu().numpy()

            score_model = train(alpha, train_data, config)
            output = pc_sampler2(score_model, config, show_image=False)
            output = output.cpu().numpy()

            total_distance += ot.sliced_wasserstein_distance(valid_data, output, n_projections=10)

    distance = total_distance / 3 / num_iter
    return {"distance": distance}


if __name__ == "__main__":

    ax_client = AxClient()

    if args.score_type in ['rectified+linear', 'real+linear']:
        parameters = [
            {"name": "score_coeff1", "type": "range", "bounds": [0., 1.], "value_type": "float"},
            {"name": "score_coeff2", "type": "range", "bounds": [0., 1.], "value_type": "float"},
        ]
    elif args.score_type == 'rectified_with_hyperparam':
        parameters = [
            {"name": "c_hat", "type": "range", "bounds": [0.1, 2.], "value_type": "float"},
            {"name": "beta_hat", "type": "range", "bounds": [0.1, 1.1], "value_type": "float"},
        ]
    elif args.score_type == 'professor_suggestion':
        parameters = [
            {"name": "c_hat", "type": "range", "bounds": [0.1, 2.], "value_type": "float"},
            {"name": "change_point", "type": "range", "bounds": [1., 5.], "value_type": "float"},
        ]

    else:
        NotImplementedError()

    ax_client.create_experiment(
        name=args.score_type + "_experiment",
        parameters=parameters,
        objective_name="distance",
        minimize=True,
    )

    if args.score_type in ['rectified+linear', 'real+linear']:
        re_score_hp = {"score_coeff1": 1., "score_coeff2": 0.}
        ax_client.attach_trial(re_score_hp)
        ax_client.complete_trial(trial_index=0, raw_data=distance_func(args.alpha, args.score_type, re_score_hp))

        linear_score_hp = {"score_coeff1": 0., "score_coeff2": 1.}
        ax_client.attach_trial(linear_score_hp)
        ax_client.complete_trial(trial_index=1, raw_data=distance_func(args.alpha, args.score_type, linear_score_hp))


    elif args.score_type == 'rectified_with_hyperparam':

        linear_score_hp = {"c_hat": 0.5, "beta_hat": 1.}
        ax_client.attach_trial(linear_score_hp)
        ax_client.complete_trial(trial_index=0, raw_data=distance_func(args.alpha, args.score_type, linear_score_hp))

        _, c, t = _get_c_t(alpha=args.alpha)
        reels_hp = {"c_hat": c.item(), "beta_hat": t.item()}
        ax_client.attach_trial(reels_hp)
        ax_client.complete_trial(trial_index=1, raw_data=distance_func(args.alpha, args.score_type, reels_hp))
    else:
        NotImplementedError()

    for i in range(30 - 2):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index,
                                 raw_data=distance_func(args.alpha, args.score_type, parameters))

    best_parameters, values = ax_client.get_best_parameters()
    print("args:", args)
    print(f"Best parameters: {best_parameters}")
    print(f"Corresponding mean: {values[0]}, covariance: {values[1]}")
