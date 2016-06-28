import os

import pandas as pd

from rec_si.train_eval_save import train_eval_save
from rec_si.recommender.mpcf import MPCFModel
from rec_si.recommender.si_model import create_si_item_model, create_si_user_model
from rec_si.sampler.zero_sampler import ZeroSampler
from rec_si.utils import binarize_ratings, create_lookup_tables


def train_mpcf(config):
    ratings = pd.read_csv(config['ratings_path'])
    train = pd.read_csv(config['train_path'])
    test = pd.read_csv(config['test_path'])
    val = None
    if config['val']:
        val = pd.read_csv(config['val_path'])

    zero_sampler = None
    if 'zero_sample_factor' in config:
        config['zero_samples_total'] = len(train) * config['zero_sample_factor']
        zero_sampler = ZeroSampler(ratings)

    if config['binarize']:
        train = binarize_ratings(train, pos=config['binarize_pos'], neg=config['binarize_neg'],
                                 threshold=config['binarize_threshold'])
        test = binarize_ratings(test, pos=config['binarize_pos'], neg=config['binarize_neg'],
                                threshold=config['binarize_threshold'])
        if val is not None:
            val = binarize_ratings(val, pos=config['binarize_pos'], neg=config['binarize_neg'],
                                   threshold=config['binarize_threshold'])

    si_item_model = None
    if 'si_item_model' in config and config['si_item_model']:
        si_item_model, config = create_si_item_model(config, ratings)

    si_user_model = None
    if 'si_user_model' in config and config['si_user_model']:
        si_user_model, config = create_si_user_model(config, ratings)

    if config['verbose'] > 0:
        print "experiment: ", config['experiment_name']
        print config

    users, items = create_lookup_tables(ratings)

    model = MPCFModel(users, items, config)
    model.si_item_model = si_item_model
    model.si_user_model = si_user_model

    loss_history = model.fit(train, val=val, test=test, zero_sampler=zero_sampler)

    return model, config, loss_history


if __name__ == "__main__":

    # make local dir the working dir, st paths are working
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    config = {}

    config['verbose'] = 1

    config['lr'] = 0.01
    config['lr_decay'] = 0.02
    #config['lr_power_t'] = 0.25
    config['reg_lambda'] = 0.01
    config['nb_latent_f'] = 128
    config['nb_user_pref'] = 4

    config['adagrad'] = True
    if config['adagrad']:
        config['ada_eps'] = 1e-6

    config['init_params_scale'] = 0.001

    config['nb_epochs'] = 5

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

    config['model_save_dir'] = 'models/mpcf'

    config['zero_sample_factor'] = 3

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1
        config['binarize_pos'] = 1
        config['binarize_neg'] = 0

    config['use_avg_rating'] = False

    config['experiment_name'] = 'si_ml-100k_e5_tt-0.7_three-similar'

    config['si_item_model'] = True
    config['si_user_model'] = False

    if config['si_item_model']:
        config['si_item_d2v_model'] = 'doc2vec-models/2016-04-14_17.36.08_20e_pv-dbow_size50_lr0.025_window8_neg5'
        #config['si_item_vector_dict'] = ''
        config['si_item_valid_id'] = 2
        config['si_item_lr'] = 0.01
        config['si_item_lr_decay'] = 5e-4
        config['si_item_lambda_d_item_f'] = 0.001
        config['si_item_reg_lambda'] = 0.01
        config['si_item_cosine_lambda'] = 0.1
        config['si_item_nn_hidden'] = []

        config['si_item_topn_similar'] = 0

        config['model_save_dir'] = 'models/mpcf-si'

    if config['si_user_model']:
        config['si_user_d2v_model'] = ''
        #config['si_user_vector_dict'] = ''
        config['si_user_valid_id'] = 1
        config['si_user_lr'] = 0.01
        config['si_user_lr_decay'] = 5e-4
        config['si_user_lambda_d_item_f'] = 0.1
        config['si_user_reg_lambda'] = 0.0001
        config['si_user_cosine_lambda'] = 0.5
        config['si_user_nn_hidden'] = []

        config['model_save_dir'] = 'models/mpcf-si'

    config['run_eval'] = True
    if config['run_eval']:
        config['precision_recall_at_n'] = 20
        config['hit_threshold'] = 4
        config['top_n_predictions'] = 100
        config['run_movie_metrics'] = True

        config['eval_in_parallel'] = True
        config['pool_size'] = 4

        if config['si_item_model']:
            config['metrics_save_dir'] = 'metrics/mpcf-si'
        else:
            config['metrics_save_dir'] = 'metrics/mpcf'

    train_eval_save(config, train_mpcf)
