import multiprocessing
import os

from sklearn.grid_search import ParameterGrid
import random

from train_slim import train_slim
from utils import merge_dicts, easy_parallize


def local_train_slim(args):
    config, q = args
    train_slim(config)

    if q is not None:
        q.put(q)

if __name__ == '__main__':

    # make local dir the working dir, st paths are working
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    grid_search = False
    nb_random_samples = 100
    cores = multiprocessing.cpu_count()

    params = {
        'fit_intercept': [True, False],
        'ignore_negative_weights': [True, False],
        'l1_reg': [0.0001, 0.0003, 0.001, 0.003, 0.01],
        'l2_reg': [0.00003, 0.0001, 0.0003, 0.001, 0.003],
        'nb_epochs': [3, 5, 8, 10],
        'binarize': [True, False]
    }

    param_comb = list(ParameterGrid(params))

    if not grid_search and nb_random_samples < len(param_comb): # random search
        param_comb = random.sample(param_comb, nb_random_samples)

    config = {}
    config['ratings_path'] = 'data/splits/ml-100k/ratings.csv'
    config['sparse_item'] = True
    config['train_test_split'] = 0.7
    config['train_path'] = 'data/splits/ml-100k/sparse-item/0.7-0.8-train.csv'
    config['test_path'] = 'data/splits/ml-100k/sparse-item/0.7-test.csv'
    config['test'] = True
    config['val'] = True
    if config['val']:
        config['train_val_split'] = 0.8
        config['val_path'] = 'data/splits/ml-100k/sparse-item/0.7-0.8-val.csv'

    config['run_eval'] = True
    config['precision_recall_at_n'] = 20
    config['verbose'] = 0
    config['hit_threshold'] = 4
    config['top_n_predictions'] = 100
    config['run_movie_metrics'] = False
    config['eval_in_parallel'] = False
    config['pool_size'] = 2

    all_configs = []
    for i, p in enumerate(param_comb):
        curr_config = merge_dicts(config, p)

        curr_config['alpha'] = curr_config['l1_reg'] + curr_config['l2_reg']
        curr_config['l1_ratio'] = curr_config['l1_reg'] / curr_config['alpha']

        if curr_config['binarize']:
            curr_config['binarize_threshold'] = 1
            curr_config['binarize_pos'] = 1
            curr_config['binarize_neg'] = 0

        curr_config['experiment_name'] = 'no-si_ml-100k_e{}_tt-0.7_task-{}'.format(curr_config['nb_epochs'], i)

        all_configs.append(curr_config)

    easy_parallize(local_train_slim, all_configs, p=cores)
