import multiprocessing
import os

from gensim.models import Doc2Vec
from sklearn.grid_search import ParameterGrid
import random

from train_eval_save import train_eval_save
from train_mpcf import train_mpcf
from utils import merge_dicts, easy_parallize


def local_train_mpcf(args):
    config, q = args
    train_eval_save(config, train_mpcf)

    if q is not None:
        q.put(q)

if __name__ == '__main__':

    # make local dir the working dir, st paths are working
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    grid_search = False
    nb_random_samples = 20
    cores = multiprocessing.cpu_count()

    params = {
        'lr': [0.003, 0.01, 0.03],
        'lr_decay': [5e-4, 2e-2],
        'reg_lambda': [0.001, 0.003, 0.01],
        'nb_latent_f': [64, 96, 128],
        'nb_user_pref': [2, 4, 8],
        'binarize': [True, False],
        'use_avg_rating': [True, False],
        'zero_sample_factor': [3, 5],
        'd2v_model': ['doc2vec-models/2016-04-14_17.36.08_20e_pv-dbow_size50_lr0.025_window8_neg5'
                      ],
        'si_lr': [0.0003, 0.001, 0.003, 0.01],
        'si_lr_decay': [5e-4, 2e-2],
        'si_lambda_delta_qi': [0.0003, 0.001, 0.003],
        'si_reg_lambda': [0.001, 0.003],
        'si_nn_hidden': [[], [160]]
    }

    param_comb = list(ParameterGrid(params))

    if not grid_search and nb_random_samples < len(param_comb): # random search
        param_comb = random.sample(param_comb, nb_random_samples)

    experiment_name = 'si_ml-1m_e{}_tt-0.2_task-{}'

    config = {}
    config['nb_epochs'] = 20
    config['init_params_scale'] = 0.001
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

    config['model_save_dir'] = 'models/mpcf-si'
    config['metrics_save_dir'] = 'metrics/mpcf-si'

    config['si_model'] = True

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

        d2v_model = Doc2Vec.load(curr_config['d2v_model'])
        curr_config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])

        si_nn = list(curr_config['si_nn_hidden'])
        si_nn.insert(0, curr_config['nb_latent_f'])
        si_nn.append(curr_config['nb_d2v_features'])
        curr_config['si_nn'] = si_nn
        curr_config.pop('si_nn_hidden', None)

        all_configs.append(curr_config)

    easy_parallize(local_train_mpcf, all_configs, p=cores)
