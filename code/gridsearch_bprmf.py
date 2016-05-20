import multiprocessing
import os

from sklearn.grid_search import ParameterGrid
import random

from train_eval_save import train_eval_save
from train_bprmf import train_bprmf
from utils import merge_dicts, easy_parallize


def local_train_bprmf(args):
    config, q = args
    train_eval_save(config, train_bprmf)

    if q is not None:
        q.put(1)

if __name__ == '__main__':

    # make local dir the working dir, st paths are working
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    grid_search = False
    nb_random_samples = 24
    cores = multiprocessing.cpu_count()

    params = {
        'lr': [0.003, 0.01, 0.03],
        'lr_decay': [5e-4, 2e-2],
        'reg_lambda': [0.001, 0.003, 0.01],
        'nb_latent_f': [64, 96, 128],
        'triplet_sample_factor': [3, 5],
    }

    param_comb = list(ParameterGrid(params))

    if not grid_search: # random search
        param_comb = random.sample(param_comb, nb_random_samples)

    experiment_name = 'bprmf_ml-1m_e{}_tt-0.2_task-{}'

    config = {}
    config['init_params_scale'] = 0.001
    config['nb_epochs'] = 20
    config['ratings_path'] = 'data/splits/ml-1m/ratings.csv'
    config['sparse_item'] = True
    config['train_test_split'] = 0.2
    config['train_path'] = 'data/splits/ml-1m/sparse-item/0.2-train.csv'
    config['test_path'] = 'data/splits/ml-1m/sparse-item/0.2-test.csv'
    config['test'] = True
    config['val'] = False
    if config['val']:
        config['train_val_split'] = 0.8
        config['val_path'] = 'data/splits/ml-100k/sparse-item/0.7-0.8-val.csv'

    config['adagrad'] = True
    if config['adagrad']:
        config['ada_eps'] = 1e-6

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1
        config['binarize_pos'] = 1
        config['binarize_neg'] = 0

    config['model_save_dir'] = 'models/bprmf'
    config['metrics_save_dir'] = 'metrics/bprmf'

    config['run_eval'] = True
    config['precision_recall_at_n'] = 20
    config['verbose'] = 0
    config['hit_threshold'] = 4
    config['top_n_predictions'] = 100
    config['run_movie_metrics'] = False
    config['eval_in_parallel'] = False

    all_configs = []
    for i, p in enumerate(param_comb):
        curr_config = merge_dicts(config, p)

        curr_config['experiment_name'] = experiment_name.format(curr_config['nb_epochs'], i)

        all_configs.append(curr_config)

    easy_parallize(local_train_bprmf, all_configs, p=cores)
