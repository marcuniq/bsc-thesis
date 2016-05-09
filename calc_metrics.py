import numpy as np
import pandas as pd
import itertools
import datetime
import time
import progressbar
import multiprocessing

from utils import easy_parallize

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


def calc_user_metrics(args):
    params, q = args
    user_id, model, train, test, movie_ids, config = params

    if 'verbose' in config and config['verbose'] >= 2:
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

    movie_ids_of_hits = df_movies_in_test[df_movies_in_test['rating'] >= config['hit_threshold']]['movie_id'].unique()
    nb_hits = movie_ids_of_hits.size

    if 'verbose' in config and config['verbose'] >= 2:
        print "# hits:", nb_hits, "for user", user_id
    elif 'verbose' in config and config['verbose'] >= 1 and nb_hits == 0:
        print "0 hits for user", user_id

    movie_ids_not_in_train = np.setdiff1d(movie_ids, movie_ids_in_train)
    nb_movies_not_in_train = movie_ids_not_in_train.size

    # movie_ids_never_rated = np.setdiff1d(movie_ids_not_in_train, movie_ids_in_test)

    df_predictions = model.predict_for_user(user_id, movie_ids_not_in_train)

    df_test_movies_in_prediction = df_predictions[df_predictions['movie_id'].isin(movie_ids_in_test)]
    rankings = df_test_movies_in_prediction.index

    df_test_only_predictions = df_test_movies_in_prediction.reset_index(drop=True)

    # calc actual metrics
    auc, avg_precision, precision, recall, f1, reciprocal_rank, fcp, spearman_rank_corr = 0, 0, 0, 0, 0, 0, 0, 0

    auc = calc_auc(nb_movies_not_in_train, nb_test_movies, rankings)
    precision_recall_at_n = config['precision_recall_at_n']
    if nb_hits > 0:
        df_hits_in_prediction = df_predictions[df_predictions['movie_id'].isin(movie_ids_of_hits)]

        avg_precision = calc_avg_precision(df_hits_in_prediction)
        precision, recall = calc_precision_recall(df_hits_in_prediction, precision_recall_at_n)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        reciprocal_rank = calc_reciprocal_rank(df_hits_in_prediction)

    # create dics of predicted ranks and perfect ranks for each movie_id in test
    pred_rank_of_test = get_predicted_ranks(df_test_only_predictions, movie_ids_in_test)
    perfect_rank_of_test = get_perfect_ranks(df_movies_in_test, movie_ids_in_test)

    if nb_test_movies > 1:
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

    return metrics, df_predictions[:config['top_n_predictions']]


def calc_movie_metrics(args):
    params, q = args
    movie_id, model, train, test, user_ids, top_n_predictions, config = params

    if 'verbose' in config and config['verbose'] >= 2:
        print "processing movie_id", movie_id

    df_ratings_in_train = train[train['movie_id'] == movie_id]
    if len(df_ratings_in_train) > 0:
        user_ids_in_train = df_ratings_in_train['user_id'].unique()
        nb_train_ratings = user_ids_in_train.size
    else:
        user_ids_in_train = np.array([])
        nb_train_ratings = 0

    df_ratings_in_test = test[test['movie_id'] == movie_id]
    if len(df_ratings_in_test) > 0:
        df_ratings_in_test = df_ratings_in_test.sort_values('rating', ascending=False)
        df_ratings_in_test = df_ratings_in_test.reset_index(drop=True)
        df_ratings_in_test = df_ratings_in_test.groupby('rating').apply(lambda df: add_rank(df))

        user_ids_in_test = df_ratings_in_test['user_id'].unique()
        nb_test_ratings = user_ids_in_test.size

        user_ids_of_hits = df_ratings_in_test[df_ratings_in_test['rating'] >= config['hit_threshold']]['user_id'].unique()
        nb_hits = user_ids_of_hits.size
    else:
        user_ids_in_test = np.array([])
        nb_test_ratings = 0
        user_ids_of_hits = np.array([])
        nb_hits = 0

    if 'verbose' in config and config['verbose'] >= 2:
        print "movie_id", movie_id, "# train", nb_train_ratings, "# test", nb_test_ratings, "# hits", nb_hits
    elif 'verbose' in config and config['verbose'] >= 1:
        if nb_train_ratings == 0:
            print "0 train ratings for movie", movie_id
        if nb_test_ratings == 0:
            print "0 test ratings for movie", movie_id
        if nb_hits == 0:
            print "0 hits for movie", movie_id

    user_ids_not_in_train = np.setdiff1d(user_ids, user_ids_in_train)
    nb_users_not_in_train = user_ids_not_in_train.size

    # movie_ids_never_rated = np.setdiff1d(user_ids_not_in_train, movie_ids_in_test)

    df_predictions = model.predict_for_movie(movie_id, user_ids_not_in_train)

    auc = 0
    if nb_test_ratings > 0:
        df_test_ratings_in_prediction = df_predictions[df_predictions['user_id'].isin(user_ids_in_test)]
        rankings = df_test_ratings_in_prediction.index
        auc = calc_auc(nb_users_not_in_train, nb_test_ratings, rankings)

    df_ratings_in_top_n_predictions = top_n_predictions[top_n_predictions['movie_id'] == movie_id]
    nb_times_in_top_n_predictions = len(df_ratings_in_top_n_predictions)

    metrics = {'movie_id': movie_id, 'nb_train_ratings': nb_train_ratings, 'nb_test_ratings': nb_test_ratings,
               'nb_times_in_top_n_predictions': nb_times_in_top_n_predictions,
               'auc': auc}

    if q is not None:
        q.put(movie_id) # put into queue to indicate job done for this movie

    return metrics


def run_user_metrics(model, train, test, user_ids, movie_ids, config, verbose=0):
    if verbose:
        print "Calculate user metrics..."
    user_params = zip(user_ids,
                      itertools.repeat(model),
                      itertools.repeat(train),
                      itertools.repeat(test),
                      itertools.repeat(movie_ids),
                      itertools.repeat(config))

    if 'eval_in_parallel' in config and config['eval_in_parallel']:
        if 'pool_size' in config:
            pool_size = config['pool_size']
        else:
            pool_size = multiprocessing.cpu_count()
        result = easy_parallize(calc_user_metrics, user_params, p=pool_size)
    else:
        result = []
        for user_id in user_ids:
            user_metrics = calc_user_metrics(((user_id, model, train, test, movie_ids, config), None))
            result.append(user_metrics)

    user_metrics, top_n_predictions = map(list, zip(*result))
    user_metrics = pd.DataFrame(user_metrics)
    top_n_predictions = reduce(lambda x, y: x.append(y), top_n_predictions)

    return user_metrics, top_n_predictions


def run_movie_metrics(model, train, test, user_ids, movie_ids, top_n_predictions, config, verbose=0):
    if verbose:
        print "Calculate movie metrics..."
    movie_params = zip(movie_ids,
                       itertools.repeat(model),
                       itertools.repeat(train),
                       itertools.repeat(test),
                       itertools.repeat(user_ids),
                       itertools.repeat(top_n_predictions),
                       itertools.repeat(config))

    if 'eval_in_parallel' in config and config['eval_in_parallel']:
        if 'pool_size' in config:
            pool_size = config['pool_size']
        else:
            pool_size = multiprocessing.cpu_count()
        result = easy_parallize(calc_movie_metrics, movie_params, p=pool_size)
    else:
        result = []
        for movie_id in movie_ids:
            metrics = calc_movie_metrics(((movie_id, model, train, test, user_ids, top_n_predictions, config), None))
            result.append(metrics)

    movie_metrics = pd.DataFrame(result)

    return movie_metrics


def run_eval(model, train, test, ratings, config):
    movie_ids = ratings['movie_id'].unique()
    user_ids = ratings['user_id'].unique()

    verbose = 'verbose' in config and config['verbose'] > 0

    user_metrics, top_n_predictions = run_user_metrics(model, train, test, user_ids, movie_ids, config, verbose)

    if verbose:
        print "Saving user metrics and predictions ..."
    dt = datetime.datetime.now()
    user_metrics.to_csv('metrics/{:%Y-%m-%d_%H.%M.%S}_{}_user-metrics.csv'
                        .format(dt, config['experiment_name']), index=False)
    top_n_predictions.to_csv('metrics/{:%Y-%m-%d_%H.%M.%S}_{}_top-{}-predictions.csv'
                             .format(dt, config['experiment_name'], config['top_n_predictions']), index=False)

    if 'run_movie_metrics' in config and config['run_movie_metrics']:
        movie_metrics = run_movie_metrics(model, train, test, user_ids, movie_ids, top_n_predictions, config)

        if verbose:
            print "Saving movie metrics ..."
        movie_metrics.to_csv('metrics/{:%Y-%m-%d_%H.%M.%S}_{}_movie-metrics.csv'
                             .format(dt, config['experiment_name']), index=False)

if __name__ == '__main__':
    from mpcf import MPCFModel
    from slim import SLIMModel
    from mf_nn import MFNNModel

    model = MPCFModel()
    model.load('mpcf-models/2016-05-05_13.00.42_no-si_ml-100k_e10_tt-0.7_baseline.h5')

    ratings = pd.read_csv('data/splits/ml-100k/ratings.csv')
    train = pd.read_csv('data/splits/ml-100k/sparse-item/0.7-0.8-train.csv')
    test = pd.read_csv('data/splits/ml-100k/sparse-item/0.7-0.8-val.csv')

    movie_ids = ratings['movie_id'].unique()
    user_ids = ratings['user_id'].unique()

    config = {}
    config['precision_recall_at_n'] = 20
    config['verbose'] = 1
    config['experiment_name'] = 'no-si_ml-100k_e10_tt-0.7_baseline'
    config['hit_threshold'] = 4
    config['top_n_predictions'] = 100
    config['debug_eval'] = False
    config['run_movie_metrics'] = False

    run_eval(model, train, test, ratings, config)

    #top_n_predictions = pd.read_csv('metrics/2016-05-05_11.37.28_slim_e5_tt-0.7_top-100-predictions.csv')
    #movie_metrics = run_movie_metrics(model, train, test, user_ids, movie_ids, top_n_predictions, config)

    #dt = datetime.datetime.now()
    #movie_metrics.to_csv('metrics/{:%Y-%m-%d_%H.%M.%S}_{}_movie-metrics.csv'.format(dt, config['experiment_name']), index=False)
