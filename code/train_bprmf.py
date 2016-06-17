import os

import pandas as pd

from rec_si.recommender.bprmf import BPRMFModel
from rec_si.sampler.triplet_sampler import TripletSampler
from rec_si.train_eval_save import train_eval_save
from rec_si.utils import binarize_ratings, create_lookup_tables


def train_bprmf(config):
    ratings = pd.read_csv(config['ratings_path'])
    train = pd.read_csv(config['train_path'])

    triplet_sampler = TripletSampler(ratings['movie_id'].unique())

    if config['binarize']:
        train = binarize_ratings(train, pos=config['binarize_pos'], neg=config['binarize_neg'],
                                 threshold=config['binarize_threshold'])

    if config['verbose'] > 0:
        print "experiment: ", config['experiment_name']
        print config

    users, items = create_lookup_tables(ratings)

    model = BPRMFModel(users, items, config)
    model.fit(train, triplet_sampler)

    return model, config, None


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

    config['init_params_scale'] = 0.001

    config['nb_epochs'] = 1

    config['triplet_sample_factor'] = 5

    config['ratings_path'] = 'data/splits/ml-100k/ratings.csv'

    config['sparse_item'] = False
    config['train_test_split'] = 0.2
    config['train_path'] = 'data/splits/ml-100k/no-sparse-item/0.2-train.csv'
    config['test_path'] = 'data/splits/ml-100k/no-sparse-item/0.2-test.csv'
    config['test'] = True

    config['val'] = False
    if config['val']:
        config['train_val_split'] = 0.8
        config['val_path'] = 'data/splits/ml-100k/sparse-item/0.7-0.8-val.csv'

    config['model_save_dir'] = 'models/bprmf'

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1
        config['binarize_pos'] = 1
        config['binarize_neg'] = 0

    config['experiment_name'] = 'bprmf_ml-100k_e1_test'

    config['run_eval'] = True
    if config['run_eval']:
        config['precision_recall_at_n'] = 20
        config['hit_threshold'] = 4
        config['top_n_predictions'] = 100
        config['run_movie_metrics'] = True

        config['eval_in_parallel'] = True
        config['pool_size'] = 4

        config['metrics_save_dir'] = 'metrics/bprmf'

    train_eval_save(config, train_bprmf)
