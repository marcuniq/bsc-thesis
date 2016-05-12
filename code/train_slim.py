import pandas as pd
import os

from slim import SLIMModel
from train_eval_save import train_eval_save
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

    if config['verbose'] > 0:
        print "experiment: ", config['experiment_name']
        print config

    model = SLIMModel(ratings, config)
    model.fit(train)

    return model, config, None


if __name__ == "__main__":

    # make local dir the working dir, st paths are working
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    config = {}

    config['verbose'] = 1

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

    config['model_save_dir'] = 'models/slim'

    #config['zero_sample_factor'] = 1

    config['binarize'] = True
    if config['binarize']:
        config['binarize_threshold'] = 1
        config['binarize_pos'] = 1
        config['binarize_neg'] = 0

    config['experiment_name'] = 'slim_e5_tt-0.7'

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
        config['hit_threshold'] = 4
        config['debug_eval'] = False
        config['top_n_predictions'] = 100
        config['run_movie_metrics'] = False

        config['eval_in_parallel'] = True
        config['pool_size'] = 2

        config['metrics_save_dir'] = 'metrics/slim'

    train_eval_save(config, train_slim)
