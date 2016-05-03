import pandas as pd
import numpy as np

from calc_metrics import run_eval
from slim import SLIM
from utils import binarize_ratings
from zero_sampler import ZeroSampler


def train_slim(config):
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
        zero_samples = zero_sampler.sample(config['zero_samples_total'], verbose=1)
        train = train.append(zero_samples).reset_index(drop=True)

    if config['binarize']:
        train = binarize_ratings(train, pos=config['binarize_pos'], neg=config['binarize_neg'], threshold=config['binarize_threshold'])
        test = binarize_ratings(test, pos=config['binarize_pos'], neg=config['binarize_neg'], threshold=config['binarize_threshold'])
        if val is not None:
            val = binarize_ratings(val, pos=config['binarize_pos'], neg=config['binarize_neg'],
                                     threshold=config['binarize_threshold'])

    print "experiment: ", config['experiment_name']
    print config

    model = SLIM(ratings, config)
    model.fit(train)

    if config['run_eval']:
        train = pd.read_csv(config['train_path'])
        test = pd.read_csv(config['test_path'])
        run_eval(model, train, test, ratings, config)


if __name__ == "__main__":
    config = {}

    config['ratings_path'] = 'data/splits/ml-100k/ratings.csv'

    config['sparse_item'] = True
    config['train_test_split'] = 0.2
    config['train_path'] = 'data/splits/ml-100k/sparse-item/0.2-train.csv'
    config['test_path'] = 'data/splits/ml-100k/sparse-item/0.2-test.csv'
    config['test'] = True

    config['val'] = False
    if config['val']:
        config['train_val_split'] = 0.9
        config['val_path'] = 'data/splits/ml-100k/sparse-item/0.2-0.9-val.csv'

    #config['zero_sample_factor'] = 1

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1
        config['binarize_pos'] = 1
        config['binarize_neg'] = 0

    config['experiment_name'] = 'slim_e5_train-intercepts'

    config['nb_epochs'] = 5

    config['fit_intercept'] = True
    config['ignore_negative_weights'] = False
    config['l1_reg'] = 0.001
    config['l2_reg'] = 0.0001
    config['alpha'] = config['l1_reg'] + config['l2_reg']
    config['l1_ratio'] = config['l1_reg'] / config['alpha']

    config['run_eval'] = True
    if config['run_eval']:
        config['precision_recall_at_n'] = 20
        config['verbose'] = 1
        config['hit_threshold'] = 4
        config['debug_eval'] = False

    train_slim(config)
