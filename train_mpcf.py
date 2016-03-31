import numpy as np
import sys
from ratings import get_ratings, get_train_test_split


def train_mpcf(config, train, val, test):

    # unpack config
    lr = config['lr']
    lambda_bi = config['lambda_bi']
    lambda_p = config['lambda_p']

    nb_latent_f = config['nb_latent_f'] # k
    nb_user_pref = config['nb_user_pref'] # T

    avg_train_rating = train['rating'].mean()

    users_unique = train['user_id'].unique()
    nb_users = len(users_unique)
    users = dict(zip(users_unique, range(nb_users))) # lookup table for user_id to zero indexed number

    movies_unique = train['movie_id'].unique()
    nb_movies = len(movies_unique)
    movies = dict(zip(movies_unique, range(nb_movies))) # lookup table for user_id to zero indexed number


    # init matrices
    P = np.random.uniform(low=-0.001, high=0.001, size=(nb_users, nb_latent_f)) # user latent factor matrix
    Q = np.random.uniform(low=-0.001, high=0.001, size=(nb_movies, nb_latent_f)) # item latent factor matrix
    W = np.random.uniform(low=-0.001, high=0.001, size=(nb_users, nb_latent_f, nb_user_pref)) # user latent factor tensor
    b_i = np.random.uniform(low=-0.001, high=0.001, size=(nb_movies, 1)) # item bias vector
    B_u = np.random.uniform(low=-0.001, high=0.001, size=(nb_users, nb_user_pref)) # user-interest bias matrix


    print "Start training ..."
    for epoch in range(config['nb_epochs']):
        print "epoch {}".format(epoch)

        # shuffle train

        errors = []
        total = len(train)
        point = total / 100
        increment = total / 20
        progress = 0

        # train / update model
        for row in train.itertuples():
            user_id, movie_id, rating = row[1], row[2], row[3]

            u = users[user_id]
            i = movies[movie_id]
            r_predict = avg_train_rating + b_i[i] + np.dot(P[u,:], Q[i,:].T)

            error = rating - r_predict
            errors.append(error**2)
            b_i[i] = b_i[i] + lr * (error - lambda_bi * b_i[i])
            P[u,:] = P[u,:] + lr * (error * Q[i,:] - lambda_p * P[u,:])
            Q[i,:] = Q[i,:] + lr * (error * P[u,:] - lambda_p * Q[i,:])

            # update progess bar
            if(progress % (5 * point) == 0):
                sys.stdout.write("\r[" + "=" * (progress / increment) +  " " * ((total - progress)/ increment) + "]" + str(progress / point) + "%")
                sys.stdout.flush()

            progress = progress + 1

        print ""
        # validation

        # report error
        print sum(errors)

        # save

    # report error on test set

if __name__ == "__main__":
    config = {'lr': 0.1, 'lambda_bi': 0.1, 'lambda_p':0.1, 'nb_latent_f': 5, 'nb_user_pref':3, 'nb_epochs':10}
    ratings_path = 'data\\ml-1m\\processed\\ratings.csv'
    movies_path = 'data\\ml-1m\\processed\\movies-enhanced.csv'
    all_subs_path = 'data\\subs\\all.txt'
    ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    train, test = get_train_test_split(ratings)

    train_mpcf(config, train, train, train)
