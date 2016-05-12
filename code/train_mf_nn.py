import os

import pandas as pd
from gensim.models import Doc2Vec

from train_eval_save import train_eval_save
from utils import binarize_ratings
from user_pref_model import UserPrefModel
from calc_metrics import run_eval
from mf_nn import MFNNModel
from zero_sampler import ZeroSampler


def train_mf_nn(config):
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

    d2v_model = Doc2Vec.load(config['d2v_model'])
    config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])
    user_pref_model = UserPrefModel(config)

    if config['verbose'] > 0:
        print "experiment: ", config['experiment_name']
        print config

    model = MFNNModel(ratings, config)
    model.user_pref_model = user_pref_model
    model.d2v_model = d2v_model
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
    config['lr_decay'] = 5e-4
    #config['lr_power_t'] = 0.25
    config['reg_lambda'] = 0.001
    config['nb_latent_f'] = 128

    config['adagrad'] = True
    if config['adagrad']:
        config['ada_eps'] = 1e-6

    config['nb_epochs'] = 20

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

    config['model_save_dir'] = 'models/mfnn'

    config['zero_sample_factor'] = 3

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1
        config['binarize_pos'] = 1
        config['binarize_neg'] = 0

    config['experiment_name'] = 'mf-nn_ml-100k_e20_tt-0.7'

    config['use_avg_rating'] = True

    config['d2v_model'] = 'doc2vec-models/2016-04-14_17.36.08_20e_pv-dbow_size50_lr0.025_window8_neg5'
    config['user_pref_lr'] = 0.003
    config['user_pref_lr_decay'] = 5e-4
    config['user_pref_lambda_grad'] = 0.003
    config['user_pref_reg_lambda'] = 0.0001
    config['user_pref_hidden_dim'] = [4, 1]
    config['user_pref_movie_d2v'] = True

    config['run_eval'] = True
    if config['run_eval']:
        config['precision_recall_at_n'] = 20
        config['hit_threshold'] = 4
        config['top_n_predictions'] = 100
        config['run_movie_metrics'] = False

        config['eval_in_parallel'] = True
        config['pool_size'] = 2

        config['metrics_save_dir'] = 'metrics/mfnn'

    train_eval_save(config, train_mf_nn)
