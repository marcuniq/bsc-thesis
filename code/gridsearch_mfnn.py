import multiprocessing
import os

from gensim.models import Doc2Vec
from sklearn.grid_search import ParameterGrid
import random

from train_eval_save import train_eval_save
from train_mfnn import train_mfnn
from utils import merge_dicts, easy_parallize


def local_train_mfnn(args):
    config, q = args
    train_eval_save(config, train_mfnn)

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
        'binarize': [True, False],
        'use_avg_rating': [True, False],
        'zero_sample_factor': [3, 5],
        'd2v_model': ['doc2vec-models/2016-04-14_17.36.08_20e_pv-dbow_size50_lr0.025_window8_neg5'
                      ],
        'user_pref_lr': [0.003, 0.01, 0.03],
        'user_pref_lr_decay': [5e-4, 2e-2],
        'user_pref_lambda_grad': [0.01, 0.03, 0.1, 0.3],
        'user_pref_reg_lambda': [0.0003, 0.001, 0.003],
        'user_pref_hidden_dim': [[4, 1], [10, 1], [100, 1]]
    }

    param_comb = list(ParameterGrid(params))

    if not grid_search and nb_random_samples < len(param_comb): # random search
        param_comb = random.sample(param_comb, nb_random_samples)

    experiment_name = 'mfnn_ml-100k_e{}_tt-0.7_task-{}'

    config = {}
    config['nb_epochs'] = 20
    config['ratings_path'] = 'data/splits/ml-100k/ratings.csv'
    config['sparse_item'] = True
    config['train_test_split'] = 0.7
    config['train_path'] = 'data/splits/ml-100k/sparse-item/0.7-train.csv'
    config['test_path'] = 'data/splits/ml-100k/sparse-item/0.7-test.csv'
    config['test'] = True
    config['val'] = False
    if config['val']:
        config['train_val_split'] = 0.8
        config['val_path'] = 'data/splits/ml-100k/sparse-item/0.7-0.8-val.csv'

    config['adagrad'] = True
    if config['adagrad']:
        config['ada_eps'] = 1e-6

    config['model_save_dir'] = 'models/mfnn'
    config['metrics_save_dir'] = 'metrics/mfnn'

    config['user_pref_input_user_id'] = True
    config['user_pref_input_movie_id'] = True
    config['user_pref_input_movie_d2v'] = True

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

        if curr_config['binarize']:
            curr_config['binarize_threshold'] = 1
            curr_config['binarize_pos'] = 1
            curr_config['binarize_neg'] = 0

        curr_config['experiment_name'] = experiment_name.format(curr_config['nb_epochs'], i)

        all_configs.append(curr_config)

    easy_parallize(local_train_mfnn, all_configs, p=cores)
