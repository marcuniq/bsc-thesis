import numpy as np
import pandas as pd
import itertools

from gensim.models.doc2vec import Doc2Vec
from ratings import get_ratings, get_train_test_split
from train_mpcf import MPCFModel


def calc_auc(params):
    user_id, model, train, test, movie_ids = params
    movies_in_train = train[train['user_id'] == user_id]['movie_id'].unique()
    nb_train_movies = movies_in_train.size
    movies_in_test = test[test['user_id'] == user_id]['movie_id'].unique()
    nb_test_movies = movies_in_test.size

    movies_not_rated = np.setdiff1d(movie_ids, movies_in_train)
    nb_movie_not_rated = movies_not_rated.size

    print "making predictions for user", user_id
    predictions = model.predict_for_user(user_id, movies_not_rated)

    rankings = predictions[predictions['movie_id'].isin(movies_in_test)].index

    nb_all_pairs = (nb_movie_not_rated - nb_test_movies) * nb_test_movies
    nb_correct_pairs = 0
    for rank in rankings:
        nb_correct_pairs += nb_movie_not_rated - rank + 1
    nb_correct_pairs -= nb_test_movies

    auc = float(nb_correct_pairs) / nb_all_pairs
    data = {'user_id': user_id, 'nb_train_movies': nb_train_movies, 'nb_test_movies': nb_test_movies,
            'nb_movies_not_rated': nb_movie_not_rated, 'rankings': [list(rankings)], 'auc': auc}

    return data


if __name__ == '__main__':

    def easy_parallize(f, sequence, p=8):
        # I didn't see gains with .dummy; you might
        from multiprocessing import Pool
        pool = Pool(processes=p)
        #from multiprocessing.dummy import Pool
        #pool = Pool(16)

        # f is given sequence. Guaranteed to be in order
        result = pool.map(f, sequence)
        cleaned = [x for x in result if x is not None]
        # not optimal but safe
        pool.close()
        pool.join()
        return cleaned

    config = {'lr': 0.001, 'lr_decay': 5e-4, 'lambda_bi': 0.06, 'lambda_p': 0.06, 'nb_latent_f': 128, 'nb_user_pref': 2,
              'nb_epochs': 1, 'save_on_epoch_end': False, 'train_test_split': 0.75}
    ratings_path = 'data\\ml-1m\\processed\\ratings.csv'
    movies_path = 'data\\ml-1m\\processed\\movies-enhanced.csv'
    all_subs_path = 'data\\subs\\all.txt'
    ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    train, test = get_train_test_split(ratings, train_size=config['train_test_split'], sparse_item=False)

    movie_ids = ratings['movie_id'].unique()
    user_ids = ratings['user_id'].unique()

    nb_movies = movie_ids.size

    # train model without side information
    ######################################
    config['experiment_name'] = 'compare_auc_1e_parallel'

    #model = MPCFModel(ratings, config)
    #model.fit(train)
    model = MPCFModel()
    model.load('mpcf-models/2016-04-01_18.45.28_kabbur_best_e200.h5')

    # calc AUC #
    params = zip(user_ids[:100], itertools.repeat(model), itertools.repeat(train), itertools.repeat(test), itertools.repeat(movie_ids))
    result = easy_parallize(calc_auc, params)
    log = pd.DataFrame(result)
    log.to_csv('model_without_sideinfo_parallel_test.csv')

    quit()

    print
    print "Running model with side information..."

    # train model with side information
    ######################################
    d2v_model = Doc2Vec.load('doc2vec-models\\doc2vec-model_stopwords-removed')
    config['nb_d2v_features'] = int(d2v_model.docvecs['107290.txt'].shape[0])
    config['si_model'] = True
    config['lr_si'] = 0.01
    config['experiment_name'] = 'compare_auc_100e'

    model = MPCFModel(ratings, config)
    model.fit(train, d2v_model=d2v_model)

    # calc AUC #
    log = calc_auc(model)
    log.to_csv('model_with_sideinfo_100e.csv')
