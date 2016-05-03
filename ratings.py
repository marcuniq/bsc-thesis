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
    ratings_enhanced = ratings.merge(movies, on='movie_id', how='inner')
    ratings_enhanced = ratings_enhanced[['user_id', 'movie_id', 'rating', 'timestamp', 'imdb_id']]

    result = ratings_enhanced[ratings_enhanced['imdb_id'].isin(subs['imdb_id'])]
    return result


def retain_all_users(train, test, ratings):
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    user_ids = ratings['user_id'].unique()
    users_not_retained = set(user_ids)

    while len(users_not_retained) != 0:
        users_not_retained = set(user_ids)
        for user_id in user_ids:
            user_train_ratings = train[train['user_id'] == user_id]
            user_test_ratings = test[test['user_id'] == user_id]

            is_in_train = True if len(user_train_ratings) != 0 else False
            is_in_test = True if len(user_test_ratings) != 0 else False

            if is_in_train and is_in_test:
                users_not_retained.remove(user_id)
                continue

            users_not_retained.add(user_id)
            if is_in_train and not is_in_test:
                random_rating = user_train_ratings.sample(1)
                train = train.drop(random_rating.index)
                test = test.append(random_rating)

            elif is_in_test and not is_in_train:
                random_rating = user_test_ratings.sample(1)
                test = test.drop(random_rating.index)
                train = train.append(random_rating)

    return train, test


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

    # test = ratings - train
    train_set = set([ tuple(line) for line in train.values.tolist()])
    ratings_set = set([ tuple(line) for line in ratings.values.tolist()])
    test = pd.DataFrame(list(ratings_set.difference(train_set)))
    del train_set, ratings_set
    test.columns = ratings.columns

    if sparse_item:
        train, test = retain_all_users(train, test, ratings)

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    return train, test
