import numpy as np
import sys
from ratings import get_ratings, get_train_test_split


def train_mpcf(config, train, val=None, test=None, verbose=0):

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
    W = np.random.uniform(low=-0.001, high=0.001, size=(nb_users, nb_user_pref, nb_latent_f)) # user latent factor tensor
    b_i = np.random.uniform(low=-0.001, high=0.001, size=(nb_movies, 1)) # item bias vector
    B_u = np.random.uniform(low=-0.001, high=0.001, size=(nb_users, nb_user_pref)) # user-interest bias matrix

    def get_local_pref(u, i):
        max_score = False
        local_pref = 0
        for t in range(nb_user_pref):
            score = B_u[u,t] + np.dot(W[u,t,:], Q[i,:].T)
            if not max_score or score > max_score:
                max_score = score
                local_pref = t
        return local_pref, max_score

    print "Start training ..."
    for epoch in range(config['nb_epochs']):
        print "epoch {}".format(epoch)

        # shuffle train
        train = train.reindex(np.random.permutation(train.index))

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

            local_pref, local_pref_score = get_local_pref(u, i)
            r_predict = avg_train_rating + b_i[i] + np.dot(P[u,:], Q[i,:].T) + local_pref_score

            error = rating - r_predict
            errors.append(error**2)
            b_i[i] = b_i[i] + lr * (error - lambda_bi * b_i[i])
            B_u[u,local_pref] = B_u[u,local_pref] + lr * (error - lambda_p * B_u[u,local_pref])
            P[u,:] = P[u,:] + lr * (error * Q[i,:] - lambda_p * P[u,:])
            Q[i,:] = Q[i,:] + lr * (error * P[u,:] - lambda_p * Q[i,:])
            W[u,local_pref,:] = W[u,local_pref,:] + lr * (error * Q[i,:] - lambda_p * W[u,local_pref,:])

            # update progess bar
            if(progress % (5 * point) == 0):
                sys.stdout.write("\r[" + "=" * (progress / increment) +  " " * ((total - progress)/ increment) + "]" + str(progress / point) + "%")
                sys.stdout.flush()

            progress = progress + 1

        print ""
        # report error
        print "Train root mean squared error:", np.sqrt(np.mean(errors))

        # validation
        if val is not None and 'val' in config and config['val']:
            val_errors = []
            for row in val.itertuples():
                user_id, movie_id, rating = row[1], row[2], row[3]

                if user_id not in users:
                    if verbose > 0:
                        print "[Error] user_id", user_id, "not in train set"
                    continue
                if movie_id not in movies:
                    if verbose > 0:
                        print "[Error] movie_id", movie_id, "not in train set"
                    continue

                u = users[user_id]
                i = movies[movie_id]
                local_pref, local_pref_score = get_local_pref(u, i)
                r_predict = avg_train_rating + b_i[i] + np.dot(P[u,:], Q[i,:].T) + local_pref_score

                val_error = rating - r_predict
                val_errors.append(val_error**2)

            # report error
            print "Validation root mean squared error:", np.sqrt(np.mean(val_errors))

        # save

    # report error on test set

if __name__ == "__main__":
    config = {'lr': 0.1, 'lambda_bi': 0.1, 'lambda_p':0.1, 'nb_latent_f': 32, 'nb_user_pref':8, 'nb_epochs':20, 'val':True}
    ratings_path = 'data\\ml-1m\\processed\\ratings.csv'
    movies_path = 'data\\ml-1m\\processed\\movies-enhanced.csv'
    all_subs_path = 'data\\subs\\all.txt'
    ratings = get_ratings(ratings_path, movies_path, all_subs_path)
    train, test = get_train_test_split(ratings)
    train, val = get_train_test_split(train, train_size=0.8)

    train_mpcf(config, train, val, test)
