import numpy as np
import progressbar
from gensim.models import Doc2Vec

from base_recommender import BaseRecommender
from user_pref_model import UserPrefModel


class MFNNModelNumpy(BaseRecommender):

    def __init__(self, users=None, items=None, config=None, movie_to_imdb=None,d2v_model=None):
        BaseRecommender.__init__(self, users, items, config)

        self.user_factors = None
        self.item_factors = None
        self.item_bias = None

        self.global_bias = None

        self.nn_w1 = None
        self.nn_w2 = None

        self.d2v_model = d2v_model
        self.movie_to_imdb = movie_to_imdb

        if config and self.users and self.items:
            nb_users = len(self.users)
            nb_movies = len(self.items)
            self.nb_latent_f = config['nb_latent_f']
            nb_d2v_features = config['nb_d2v_features']
            nb_hidden_neurons = config['nb_hidden_neurons']
            params = self._init_params(nb_users, nb_movies, self.nb_latent_f, nb_d2v_features, nb_hidden_neurons)
            self._set_params(params)

    def _init_params(self, nb_users, nb_movies, nb_latent_f, nb_d2v_features, nb_hidden_neurons, scale=0.001):
        P = np.random.normal(scale=scale, size=(nb_users, nb_latent_f)) # user latent factor matrix
        Q = np.random.normal(scale=scale, size=(nb_movies, nb_latent_f)) # item latent factor matrix
        b_i = np.random.normal(scale=scale, size=(nb_movies, 1)) # item bias vector
        nb_input_dim = 2*nb_latent_f + nb_d2v_features + 1
        nn_w1 = np.random.rand(nb_input_dim, nb_hidden_neurons) / np.sqrt(nb_input_dim) # W'
        nn_w2 = np.random.rand(nb_hidden_neurons, 1) / np.sqrt(nb_hidden_neurons) # w

        params = {'P': P, 'Q': Q, 'b_i': b_i, 'nn_w1': nn_w1, 'nn_w2': nn_w2}
        return params

    def _get_params(self):
        return {'P': self.user_factors, 'Q': self.item_factors, 'b_i': self.item_bias, 'avg_train_rating': self.global_bias,
                'movie_to_imdb': self.movie_to_imdb}

    def _set_params(self, params):
        self.user_factors = params['P']
        self.item_factors = params['Q']
        self.item_bias = params['b_i']
        self.nn_w1 = params['nn_w1']
        self.nn_w2 = params['nn_w2']
        self.global_bias = params['avg_train_rating'] if 'avg_train_rating' in params else None

        if self.movie_to_imdb is None and 'movie_to_imdb' in params:
            self.movie_to_imdb = params['movie_to_imdb']

        if self.d2v_model is None and 'd2v_model' in self.config:
            self.d2v_model = Doc2Vec.load(self.config['d2v_model'])

    def test(self, data):
        test_errors = []
        for row in data.itertuples():
            user_id, movie_id, rating = row[1], row[2], row[3]

            r_predict = self.predict(user_id, movie_id)

            test_error = rating - r_predict
            test_errors.append(test_error)
        return test_errors

    def predict(self, user_id, item_id):
        u = self.users[user_id]
        i = self.items[item_id]
        imdb_id = self.movie_to_imdb[item_id]

        rating_mf = self.global_bias + self.item_bias[i] + np.dot(self.user_factors[u, :],
                                                                  self.item_factors[i, :].T)

        # nn_ui
        input_vec = np.concatenate(
            (self.item_factors[i, :],
             self.user_factors[u, :],
             self.d2v_model.docvecs['{}.txt'.format(imdb_id)],
             [1])).reshape((-1, 1))

        z = np.dot(input_vec.T, self.nn_w1)
        # a = 1 / (1 + np.exp(-z)) # sigmoid activation function
        a = np.maximum(0, z)  # relu activation function
        nn_ui = np.dot(a, self.nn_w2)

        r_predict = rating_mf + nn_ui

        return float(r_predict)

    def fit(self, train, val=None, test=None, zero_sampler=None, verbose=1):

        config = self.config
        lr = config['lr']
        reg_lambda = config['reg_lambda']
        nn_reg_lambda = config['nn_reg_lambda']

        if config['use_avg_rating']:
            self.global_bias = train['rating'].mean()
        else:
            self.global_bias = 0

        verbose = 'verbose' in config and config['verbose'] > 0

        # AdaGrad
        if 'adagrad' in config and config['adagrad']:
            adac_global_bias = 0
            adac_item_bias = np.zeros_like(self.item_bias)
            adac_user_factors = np.zeros_like(self.user_factors)
            adac_item_factors = np.zeros_like(self.item_factors)
            adac_nn_w1 = np.zeros_like(self.nn_w1)
            adac_nn_w2 = np.zeros_like(self.nn_w2)
            ada_eps = config['ada_eps']

        train_rmse = []
        val_rmse = []
        feature_loss = []

        if verbose:
            print "Start training ..."
        for epoch in range(config['nb_epochs']):
            if verbose:
                print "epoch {}, lr {}".format(epoch, lr)

            if zero_sampler and 'zero_samples_total' in config:
                if verbose:
                    print "sampling zeros ..."
                zero_samples = zero_sampler.sample(config['zero_samples_total'], verbose=verbose)
                train_for_epoch = train.append(zero_samples).reset_index(drop=True)
            else:
                train_for_epoch = train

            # shuffle train
            train_for_epoch = train_for_epoch.reindex(np.random.permutation(train_for_epoch.index))

            rating_errors = []

            if verbose:
                total = len(train_for_epoch)
                bar = progressbar.ProgressBar(max_value=total)
                bar.start()
                progress = 0

            # train / update model
            for row in train_for_epoch.itertuples():
                user_id, movie_id, rating = row[1], row[2], row[3]

                u = self.users[user_id]
                i = self.items[movie_id]
                imdb_id = self.movie_to_imdb[movie_id]

                # predict rating and calc rating error
                rating_mf = self.global_bias + self.item_bias[i] + np.dot(self.user_factors[u, :],
                                                                          self.item_factors[i, :].T)

                # nn_ui
                input_vec = np.concatenate(
                    (self.item_factors[i, :],
                     self.user_factors[u, :],
                     self.d2v_model.docvecs['{}.txt'.format(imdb_id)],
                     [1])).reshape((-1, 1))

                z = np.dot(input_vec.T, self.nn_w1)
                # a = 1 / (1 + np.exp(-z)) # sigmoid activation function
                a = np.maximum(0, z) # relu activation function
                nn_ui = np.dot(a, self.nn_w2)

                rating_predict = float(rating_mf + nn_ui)

                rating_error = rating - rating_predict
                rating_errors.append(float(rating_error))

                # calc gradients
                nn_d_out = -1 * rating_error
                nn_d_w2 = nn_d_out * a
                nn_d_w2 = np.reshape(nn_d_w2, (-1, 1))
                #deriv = np.vectorize(lambda x: x * (1 - x)) # if sigmoid
                deriv = np.vectorize(lambda x: 1 if x > 0 else 0) # if relu (max)
                nn_d_hidden = np.dot(nn_d_out, self.nn_w2.T) * deriv(z)
                nn_d_w1 = np.dot(input_vec, nn_d_hidden)
                nn_d_input = np.dot(nn_d_hidden, self.nn_w1.T).flatten()

                nn_d_item_f = nn_d_input[:self.nb_latent_f]
                nn_d_user_f = nn_d_input[self.nb_latent_f:2*self.nb_latent_f]

                # add reg to nn
                nn_d_w2 += nn_reg_lambda * self.nn_w2
                nn_d_w1 += nn_reg_lambda * self.nn_w1

                d_global_bias = rating_error
                d_item_bias = rating_error - reg_lambda * self.item_bias[i]
                d_user_factors = rating_error * self.item_factors[i, :] - reg_lambda * self.user_factors[u, :] - nn_d_user_f
                d_item_factors = rating_error * self.user_factors[u, :] - reg_lambda * self.item_factors[i, :] - nn_d_item_f
                d_nn_w1 = -1 * nn_d_w1
                d_nn_w2 = -1 * nn_d_w2

                # update AdaGrad caches and parameters
                if 'adagrad' in config and config['adagrad']:
                    adac_global_bias += d_global_bias ** 2
                    adac_item_bias[i] += d_item_bias ** 2
                    adac_user_factors[u, :] += d_user_factors ** 2
                    adac_item_factors[i, :] += d_item_factors ** 2
                    adac_nn_w1 += d_nn_w1 ** 2
                    adac_nn_w2 += d_nn_w2 ** 2

                    # update parameters
                    self.global_bias += lr * d_global_bias / (np.sqrt(adac_global_bias) + ada_eps)
                    self.item_bias[i] += lr * d_item_bias / (np.sqrt(adac_item_bias[i]) + ada_eps)
                    self.user_factors[u, :] += lr * d_user_factors / (np.sqrt(adac_user_factors[u, :]) + ada_eps)
                    self.item_factors[i, :] += lr * d_item_factors / (np.sqrt(adac_item_factors[i, :]) + ada_eps)
                    self.nn_w1 += lr * d_nn_w1 / np.sqrt(adac_nn_w1 + ada_eps)
                    self.nn_w2 += lr * d_nn_w2 / np.sqrt(adac_nn_w2 + ada_eps)
                else:
                    self.global_bias += lr * d_global_bias
                    self.item_bias[i] += lr * d_item_bias
                    self.user_factors[u, :] += lr * d_user_factors
                    self.item_factors[i, :] += lr * d_item_factors
                    self.nn_w1 += lr * d_nn_w1
                    self.nn_w2 += lr * d_nn_w2

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
                lr = config['lr'] / pow(epoch+1, config['lr_power_t'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            if verbose:
                print "Train RMSE:", current_rmse

            # validation
            if val is not None and 'val' in config and config['val']:
                val_errors = self.test(val)

                # report error
                current_val_rmse = np.sqrt(np.mean(np.square(val_errors)))
                val_rmse.append(current_val_rmse)
                if verbose:
                    print "Validation RMSE:", current_val_rmse

        # report error on test set
        test_rmse = []
        if test is not None and 'test' in config and config['test']:
            test_errors = self.test(test)

            # report error
            test_rmse = np.sqrt(np.mean(np.square(test_errors)))
            if verbose:
                print "Test RMSE:", test_rmse

        # history
        history = {'train_rmse': train_rmse, 'val_rmse': val_rmse, 'feature_loss': feature_loss, 'test_rmse': test_rmse}
        return history
