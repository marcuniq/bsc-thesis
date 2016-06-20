import numpy as np
import progressbar
from base_recommender import BaseRecommender


class BPRMFModel(BaseRecommender):
    def __init__(self, users=None, items=None, config=None):
        BaseRecommender.__init__(self, users, items, config)
        self.P = None
        self.Q = None
        self.b_i = None

        if config and self.users and self.items:
            nb_users = len(self.users)
            nb_movies = len(self.items)
            nb_latent_f = config['nb_latent_f']
            scale = config['init_params_scale'] if 'init_params_scale' in config else 0.001
            params = self._init_params(nb_users, nb_movies, nb_latent_f, scale=scale)
            self._set_params(params)

    def _init_params(self, nb_users, nb_movies, nb_latent_f, scale=0.001):
        P = np.random.normal(scale=scale, size=(nb_users, nb_latent_f))  # user latent factor matrix
        Q = np.random.normal(scale=scale, size=(nb_movies, nb_latent_f))  # item latent factor matrix
        b_i = np.zeros((nb_movies, 1)) # item bias vector

        params = {'P': P, 'Q': Q, 'b_i': b_i}
        return params

    def _get_params(self):
        return {'P': self.P, 'Q': self.Q, 'b_i': self.b_i}

    def _set_params(self, params):
        self.P = params['P']
        self.Q = params['Q']
        self.b_i = params['b_i']

    def predict(self, user_id, item_id):
        u = self.users[user_id]
        i = self.items[item_id]
        r_predict = self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)
        return r_predict

    def fit(self, train, triplet_sampler):

        config = self.config
        lr = config['lr']
        reg_lambda = config['reg_lambda']

        verbose = 'verbose' in config and config['verbose'] > 0

        # AdaGrad
        if 'adagrad' in config and config['adagrad']:
            ada_cache_P = np.zeros_like(self.P)
            ada_cache_Q = np.zeros_like(self.Q)
            ada_cache_b_i = np.zeros_like(self.b_i)
            ada_eps = config['ada_eps']

        if verbose:
            print "Start training ..."

        for epoch in range(config['nb_epochs']):
            if verbose:
                print "epoch {}, lr {}".format(epoch, lr)

            if triplet_sampler and 'triplet_sample_factor' in config:
                if verbose:
                    print "sampling triplets ..."
                train_for_epoch = triplet_sampler.sample(train, config['triplet_sample_factor'])
            else:
                train_for_epoch = train

            # shuffle train
            train_for_epoch = train_for_epoch.reindex(np.random.permutation(train_for_epoch.index))

            if verbose:
                total = len(train_for_epoch)
                bar = progressbar.ProgressBar(max_value=total)
                bar.start()
                progress = 0

            # train / update model
            for row in train_for_epoch.itertuples():
                user_id, movie_id1, rating, imdb_id, movie_id2 = row[1], row[2], row[3], row[5], row[6]

                u = self.users[user_id]
                i = self.items[movie_id1]
                j = self.items[movie_id2]

                r_predict = self.b_i[i] + np.dot(self.P[u,:], self.Q[i,:].T)

                x_uij = r_predict - (self.b_i[j] + np.dot(self.P[u,:], self.Q[j,:].T))
                one_over_one_plus_ex = 1.0 / (1.0 + np.exp(x_uij))

                # calc gradients
                dP_u = one_over_one_plus_ex * (self.Q[i,:] - self.Q[j,:]) - reg_lambda * self.P[u, :]
                dQ_i = one_over_one_plus_ex * (self.P[u, :]) - reg_lambda * self.Q[i, :]
                dQ_j = one_over_one_plus_ex * (self.P[u, :]) - reg_lambda * self.Q[j, :]
                db_i = one_over_one_plus_ex - reg_lambda * self.b_i[i]
                db_j = one_over_one_plus_ex - reg_lambda * self.b_i[j]

                # update AdaGrad caches and parameters
                if 'adagrad' in config and config['adagrad']:
                    ada_cache_P[u, :] += dP_u ** 2
                    ada_cache_Q[i, :] += dQ_i ** 2
                    ada_cache_Q[j, :] += dQ_j ** 2
                    ada_cache_b_i[i] += db_i ** 2
                    ada_cache_b_i[j] += db_j ** 2

                    # update parameters
                    self.P[u, :] += lr * dP_u / (np.sqrt(ada_cache_P[u, :]) + ada_eps)
                    self.Q[i, :] += lr * dQ_i / (np.sqrt(ada_cache_Q[i, :]) + ada_eps)
                    self.Q[j, :] += lr * dQ_j / (np.sqrt(ada_cache_Q[j, :]) + ada_eps)
                    self.b_i[i] += lr * db_i / (np.sqrt(ada_cache_b_i[i]) + ada_eps)
                    self.b_i[j] += lr * db_j / (np.sqrt(ada_cache_b_i[j]) + ada_eps)
                else:
                    # update parameters
                    self.P[u, :] += lr * dP_u
                    self.Q[i, :] += lr * dQ_i
                    self.Q[j, :] += lr * dQ_j
                    self.b_i[i] += lr * db_i
                    self.b_i[j] += lr * db_j

                # update progess bar
                if verbose:
                    progress += 1
                    bar.update(progress)

            if verbose:
                bar.finish()

            # lr decay
            if 'lr_decay' in config:
                lr *= (1.0 - config['lr_decay'])
            elif 'lr_power_t' in config:
                lr = config['lr'] / pow(epoch + 1, config['lr_power_t'])
