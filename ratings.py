import pandas as pd


def get_ratings(ratings_path, movies_path, all_subs_path):
    """
    Get ratings for movies for which we have subtitles
    """
    subs = pd.read_csv(all_subs_path, header=None)
    subs.columns = ['imdb_id']
    subs['imdb_id'] = subs['imdb_id'].apply(lambda x: int(x.split('.')[0]))

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    ratings_enhanced = ratings.merge(movies, on='movie_id', how='left')
    ratings_enhanced = ratings_enhanced[['user_id', 'movie_id', 'rating', 'timestamp', 'imdb_id']]

    result = ratings_enhanced[ratings_enhanced['imdb_id'].isin(subs['imdb_id'])]
    return result


def get_train_test_split(ratings, train_size=0.2, sparse_item=True):
    if sparse_item:
        group_key = 'movie_id'
    else:
        group_key = 'user_id'

    # group by key, then sample a fraction for train
    grouped = ratings.groupby(group_key)

    def get_samples(x, train_size):
        if train_size < 1.0:
            return x.sample(frac=train_size)
        else:
            return x.sample(train_size)

    train = grouped.apply(lambda x: get_samples(x, train_size))
    train.reset_index(inplace=True, drop=True)

    # test = ratings - train
    train_set = set([ tuple(line) for line in train.values.tolist()])
    ratings_set = set([ tuple(line) for line in ratings.values.tolist()])
    test = pd.DataFrame(list(ratings_set.difference(train_set)))
    del train_set, ratings_set
    test.columns = ratings.columns

    return train, test
