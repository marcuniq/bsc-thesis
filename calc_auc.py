import numpy as np
import pandas as pd
import itertools
import datetime
import time
import progressbar
from gensim.models.doc2vec import Doc2Vec

from ratings import get_ratings, get_train_test_split
from train_mpcf import MPCFModel
from build_si_model import build_si_model


def calc_auc(args):
    params, q = args
    user_id, model, train, test, movie_ids = params
    movies_in_train = train[train['user_id'] == user_id]['movie_id'].unique()
    nb_train_movies = movies_in_train.size
    movies_in_test = test[test['user_id'] == user_id]['movie_id'].unique()
    nb_test_movies = movies_in_test.size

    movies_not_rated = np.setdiff1d(movie_ids, movies_in_train)
    nb_movie_not_rated = movies_not_rated.size

    # print "making predictions for user", user_id
    predictions = model.predict_for_user(user_id, movies_not_rated)
    predictions['user_id'] = user_id

    rankings = predictions[predictions['movie_id'].isin(movies_in_test)].index

    nb_all_pairs = (nb_movie_not_rated - nb_test_movies) * nb_test_movies
    nb_correct_pairs = 0
    for rank in rankings:
        nb_correct_pairs += nb_movie_not_rated - rank + 1
    nb_correct_pairs -= nb_test_movies

    auc = float(nb_correct_pairs) / nb_all_pairs
    data = {'user_id': user_id, 'nb_train_movies': nb_train_movies, 'nb_test_movies': nb_test_movies,
            'nb_movies_not_rated': nb_movie_not_rated, 'rankings': [list(rankings)], 'auc': auc}

    q.put(user_id) # put into queue to indicate job done for this user

    return data, predictions


if __name__ == '__main__':

    def easy_parallize(f, params, p=8):
        from multiprocessing import Pool, Manager
        pool = Pool(processes=p)
        m = Manager()
        q = m.Queue()

        nb_jobs = len(params)
        bar = progressbar.ProgressBar(max_value=nb_jobs)

        args = [(i, q) for i in params]
        results = pool.map_async(f, args)

        print "done dispatching..."
        bar.start()

        while not results.ready():
            complete_count = q.qsize()
            bar.update(complete_count)
            time.sleep(.5)

        bar.finish()

        pool.close()
        pool.join()

        return results.get()

    config = {'lr': 0.001, 'lr_decay': 5e-4, 'reg_lambda': 0.06, 'nb_latent_f': 128, 'nb_user_pref': 2,
              'nb_epochs': 200, 'save_on_epoch_end': False, 'train_test_split': 0.2}

    # ratings_path = 'data/ml-1m/processed/ratings.csv'
    # movies_path = 'data/ml-1m/processed/movies-enhanced.csv'
    # all_subs_path = 'data/subs/all.txt'
    # ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    # train, test = get_train_test_split(ratings, train_size=config['train_test_split'], sparse_item=False)

    ratings = pd.read_csv('data/splits/ratings.csv')
    train = pd.read_csv('data/splits/0.2-train.csv')
    test = pd.read_csv('data/splits/0.2-test.csv')

    config['experiment_name'] = 'test'
    side_info_model = False

    d2v_model = None
    si_model = None

    if side_info_model:
        config['d2v_model'] = 'doc2vec-models/2016-04-14_17.36.08_20e_pv-dbow_size50_lr0.025_window8_neg5'
        d2v_model = Doc2Vec.load(config['d2v_model'])
        config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])
        config['si_model'] = True
        config['lr_si'] = 0.001
        config['lr_si_decay'] = 2e-2
        config['lr_delta_qi'] = 0.00001
        config['lr_delta_qi_decay'] = 5e-4

        si_model = build_si_model(config['nb_latent_f'], config['nb_d2v_features'], len(train), config['reg_lambda'])

    #model = MPCFModel(ratings, config)
    #model.fit(train, test=test, d2v_model=d2v_model, si_model=si_model)

    model = MPCFModel()
    model.load('mpcf-models/2016-04-18_16.32.34_no-si_e20_tt-0.2_zero-samp-3.h5')

    # calc AUC #
    print "Calculate AUC..."
    movie_ids = ratings['movie_id'].unique()
    user_ids = ratings['user_id'].unique()

    nb_movies = movie_ids.size
    nb_users = user_ids.size

    params = zip(user_ids,
                 itertools.repeat(model),
                 itertools.repeat(train),
                 itertools.repeat(test),
                 itertools.repeat(movie_ids))

    result = easy_parallize(calc_auc, params)

    print "Saving results ..."
    dt = datetime.datetime.now()
    log, predictions = map(list, zip(*result))
    log = pd.DataFrame(log)
    log.to_csv('data/results/{:%Y-%m-%d_%H.%M.%S}_{}_log.csv'.format(dt, config['experiment_name']), index=False)
    predictions = reduce(lambda x,y: x.append(y), predictions)
    predictions.to_csv('data/results/{:%Y-%m-%d_%H.%M.%S}_{}_predictions.csv'.format(dt, config['experiment_name']), index=False)


