import numpy as np
import pandas as pd
import itertools
import datetime
import time
import progressbar
from gensim.models.doc2vec import Doc2Vec
import multiprocessing

from ratings import get_ratings, get_train_test_split
from train_mpcf import MPCFModel
from build_si_model import build_si_model


def add_rank(df):
    df['rank'] = np.mean(df.index)
    return df


def get_possible_pairs(rank_movie_tuples):
    all_comb = itertools.product(rank_movie_tuples, rank_movie_tuples)
    for (rank_i, movie_i), (rank_j, movie_j) in all_comb:
        if rank_i < rank_j:
            yield movie_i, movie_j


def get_predicted_ranks(predictions, movie_ids_in_test):
    predicted_rank = {}
    for movie_id in movie_ids_in_test:
        rank = predictions[predictions['movie_id'] == movie_id].index[0]
        predicted_rank[movie_id] = rank
    return predicted_rank


def get_perfect_ranks(df_movies_in_test, movie_ids_in_test):
    perfect_rank = {}
    for movie_id in movie_ids_in_test:
        perfect_rank[movie_id] = float(df_movies_in_test[df_movies_in_test['movie_id'] == movie_id]['rank'])
    return perfect_rank


def calc_fcp(predict_rank, perfect_rank, movie_ids_in_test):
    nb_test_movies = len(movie_ids_in_test)
    nb_all_pairs = nb_test_movies * (nb_test_movies - 1) / 2
    nb_correct_pairs = 0
    for movie_i, movie_j in itertools.product(movie_ids_in_test, movie_ids_in_test):
        predict_rank_i = predict_rank[movie_i]
        predict_rank_j = predict_rank[movie_j]

        perfect_rank_i = perfect_rank[movie_i]
        perfect_rank_j = perfect_rank[movie_j]

        if predict_rank_i > predict_rank_j and perfect_rank_i >= perfect_rank_j:
            nb_correct_pairs += 1

    fcp = float(nb_correct_pairs) / nb_all_pairs
    return fcp


def calc_spearman_rank_corr(predict_rank, perfect_rank, movie_ids_in_test):
    mean_predict_rank = np.mean(predict_rank.values())
    mean_perfect_rank = np.mean(perfect_rank.values())

    covariance = float(0)
    std_dev_predict = 0
    std_dev_perfect = 0
    for movie_id in movie_ids_in_test:
        pred = predict_rank[movie_id] - mean_predict_rank
        perf = perfect_rank[movie_id] - mean_perfect_rank
        covariance += pred * perf
        std_dev_predict += pred ** 2
        std_dev_perfect += perf ** 2
    std_dev_predict = np.sqrt(std_dev_predict)
    std_dev_perfect = np.sqrt(std_dev_perfect)

    spearman_rank_corr = covariance / (std_dev_predict * std_dev_perfect)
    return spearman_rank_corr


def calc_precision_recall(df_hits_in_prediction, n):
    nb_hits = len(df_hits_in_prediction)
    rankings = df_hits_in_prediction.index

    true_pos = float(0)
    for rank in rankings:
        if rank < n:
            true_pos += 1
        else:
            break
    precision = true_pos / n
    recall = true_pos / nb_hits
    return precision, recall


def calc_avg_precision(df_hits_in_prediction):
    nb_hits = len(df_hits_in_prediction)
    rankings = df_hits_in_prediction.index

    avg_precision = 0
    for i, rank in enumerate(rankings, 1):
        avg_precision += float(i) / (rank + 1)
    avg_precision /= nb_hits
    return avg_precision


def calc_reciprocal_rank(df_hits_in_prediction):
    rankings = df_hits_in_prediction.index
    mrr = float(1) / (1 + rankings[0])
    return mrr


def calc_auc(nb_movies_not_in_train, nb_test_movies, rankings):
    nb_all_pairs = (nb_movies_not_in_train - nb_test_movies) * nb_test_movies
    nb_correct_pairs = 0
    for i, rank in enumerate(rankings):
        nb_correct_pairs += nb_movies_not_in_train - rank - (nb_test_movies - i)

    auc = float(nb_correct_pairs) / nb_all_pairs
    return auc


def calc_metrics(args):
    params, q = args
    user_id, model, train, test, movie_ids, metrics_config = params

    if 'verbose' in metrics_config and metrics_config['verbose'] >= 2:
        print "processing user_id", user_id

    df_movies_in_train = train[train['user_id'] == user_id]
    movie_ids_in_train = df_movies_in_train['movie_id'].unique()
    nb_train_movies = movie_ids_in_train.size

    df_movies_in_test = test[test['user_id'] == user_id]
    df_movies_in_test = df_movies_in_test.sort_values('rating', ascending=False)
    df_movies_in_test = df_movies_in_test.reset_index(drop=True)
    df_movies_in_test = df_movies_in_test.groupby('rating').apply(lambda df: add_rank(df))

    movie_ids_in_test = df_movies_in_test['movie_id'].unique()
    nb_test_movies = movie_ids_in_test.size

    movie_ids_of_hits = df_movies_in_test[df_movies_in_test['rating'] >= 4]['movie_id'].unique()
    nb_hits = movie_ids_of_hits.size

    if 'verbose' in metrics_config and metrics_config['verbose'] >= 2:
        print "# hits:", nb_hits, "for user", user_id
    elif 'verbose' in metrics_config and metrics_config['verbose'] >= 1 and nb_hits == 0:
        print "0 hits for user", user_id

    movie_ids_not_in_train = np.setdiff1d(movie_ids, movie_ids_in_train)
    nb_movies_not_in_train = movie_ids_not_in_train.size

    # movie_ids_never_rated = np.setdiff1d(movie_ids_not_in_train, movie_ids_in_test)

    df_predictions = model.predict_for_user(user_id, movie_ids_not_in_train)
    df_predictions['user_id'] = user_id

    df_test_movies_in_prediction = df_predictions[df_predictions['movie_id'].isin(movie_ids_in_test)]
    rankings = df_test_movies_in_prediction.index

    df_test_only_predictions = df_test_movies_in_prediction.reset_index(drop=True)

    # calc actual metrics
    auc = calc_auc(nb_movies_not_in_train, nb_test_movies, rankings)
    avg_precision, precision, recall, f1, reciprocal_rank = 0, 0, 0, 0, 0
    precision_recall_at_n = metrics_config['precision_recall_at_n']
    if nb_hits > 0:
        df_hits_in_prediction = df_predictions[df_predictions['movie_id'].isin(movie_ids_of_hits)]

        avg_precision = calc_avg_precision(df_hits_in_prediction)
        precision, recall = calc_precision_recall(df_hits_in_prediction, precision_recall_at_n)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        reciprocal_rank = calc_reciprocal_rank(df_hits_in_prediction)

    # create dics of predicted ranks and perfect ranks for each movie_id in test
    pred_rank_of_test = get_predicted_ranks(df_test_only_predictions, movie_ids_in_test)
    perfect_rank_of_test = get_perfect_ranks(df_movies_in_test, movie_ids_in_test)

    fcp = calc_fcp(pred_rank_of_test, perfect_rank_of_test, movie_ids_in_test)
    spearman_rank_corr = calc_spearman_rank_corr(pred_rank_of_test, perfect_rank_of_test, movie_ids_in_test)

    metrics = {'user_id': user_id, 'nb_train_movies': nb_train_movies, 'nb_test_movies': nb_test_movies,
            'nb_movies_not_in_train': nb_movies_not_in_train, 'rankings': [list(rankings)],
            'auc': auc, 'avg_precision': avg_precision, 'f1': f1,
            'recall_at_{}'.format(precision_recall_at_n): recall,
            'precision_at_{}'.format(precision_recall_at_n): precision,
            'reciprocal_rank': reciprocal_rank,
            'fcp': fcp,
            'spearman_rank_corr': spearman_rank_corr
            }

    if q is not None:
        q.put(user_id) # put into queue to indicate job done for this user

    return metrics


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
              'nb_epochs': 1, 'save_on_epoch_end': False, 'train_test_split': 0.2, 'test': True}

    # ratings_path = 'data/ml-1m/processed/ratings.csv'
    # movies_path = 'data/ml-1m/processed/movies-enhanced.csv'
    # all_subs_path = 'data/subs/all.txt'
    # ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    # train, test = get_train_test_split(ratings, train_size=config['train_test_split'], sparse_item=False)

    ratings = pd.read_csv('data/splits/ml-1m/ratings.csv')
    train = pd.read_csv('data/splits/ml-1m/sparse-item/0.2-train.csv')
    test = pd.read_csv('data/splits/ml-1m/sparse-item/0.2-test.csv')

    config['experiment_name'] = 'no-si_e50_tt-0.2_zero-samp-3_no-val'
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
    model.load('mpcf-models/2016-04-21_15.42.48_no-si_e50_tt-0.2_zero-samp-3_no-val.h5')

    # calc AUC #
    print "Calculate metrics..."
    movie_ids = ratings['movie_id'].unique()
    user_ids = ratings['user_id'].unique()

    nb_movies = movie_ids.size
    nb_users = user_ids.size

    metrics_config = {'precision_recall_at_n': 20, 'verbose': 1}

    params = zip(user_ids,
                 itertools.repeat(model),
                 itertools.repeat(train),
                 itertools.repeat(test),
                 itertools.repeat(movie_ids),
                 itertools.repeat(metrics_config))

    result = easy_parallize(calc_metrics, params, p=multiprocessing.cpu_count())

    print "Saving results ..."
    dt = datetime.datetime.now()
    metrics = pd.DataFrame(result)
    metrics.to_csv('data/results/{:%Y-%m-%d_%H.%M.%S}_{}_metrics.csv'.format(dt, config['experiment_name']), index=False)
