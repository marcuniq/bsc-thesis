import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from ratings import get_ratings, get_train_test_split
from mpcf import MPCFModel
from build_si_model import build_si_model
from calc_metrics import run_eval
from zero_sampler import ZeroSampler

def binarize_ratings(df):
    df.loc[df['rating'] != 0, 'rating'] = 1


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
        binarize_ratings(train)
        binarize_ratings(test)
        if val:
            binarize_ratings(val)

    d2v_model, si_model = None, None
    if config['si_model']:
        d2v_model = Doc2Vec.load(config['d2v_model'])
        config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])
        si_model = build_si_model(config['si_nn'], config['si_reg_lambda'])

    print "experiment: ", config['experiment_name']
    print config

    model = MPCFModel(ratings, config)
    model.fit(train, val=val, test=test, d2v_model=d2v_model, si_model=si_model, zero_sampler=zero_sampler)

    if config['run_eval']:
        run_eval(model, train, test, ratings, config)


if __name__ == "__main__":
    config = {}

    config['lr'] = 0.001
    config['lr_decay'] = 5e-4
    config['reg_lambda'] = 0.06
    config['nb_latent_f'] = 128
    config['nb_user_pref'] = 2

    config['nb_epochs'] = 1

    config['save_on_epoch_end'] = False

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

    config['zero_sample_factor'] = 3

    config['binarize'] = True

    config['experiment_name'] = 'si_ml-100k_e1_tt-0.2_zero-samp-3_sparse-item_binarize_no-val'

    config['si_model'] = True
    if config['si_model']:
        config['d2v_model'] = 'doc2vec-models/2016-04-14_17.36.08_20e_pv-dbow_size50_lr0.025_window8_neg5'
        d2v_model = Doc2Vec.load(config['d2v_model'])
        config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])
        config['lr_si'] = 0.001
        config['lr_si_decay'] = 5e-4
        config['lr_delta_qi'] = 0.0001
        config['lr_delta_qi_decay'] = 5e-4
        config['si_reg_lambda'] = 0.01
        config['si_nn'] = [config['nb_latent_f'], config['nb_d2v_features']]

    config['run_eval'] = False
    if config['run_eval']:
        config['precision_recall_at_n'] = 20
        config['verbose'] = 1

    train_mpcf(config)
