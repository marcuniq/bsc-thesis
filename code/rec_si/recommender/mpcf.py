import numpy as np
import progressbar

from base_recommender import BaseRecommender


class MPCFModel(BaseRecommender):

    def __init__(self, users=None, items=None, config=None):
        BaseRecommender.__init__(self, users, items, config)

        self.user_factors = None
        self.item_factors = None
        self.user_interest_factors = None
        self.user_interest_bias = None
        self.item_bias = None
        self.global_bias = None

        self.si_user_model = None
        self.si_item_model = None

        if config and self.users and self.items:
            nb_users = len(self.users)
            nb_movies = len(self.items)
            nb_latent_f = config['nb_latent_f']
            nb_user_pref = config['nb_user_pref']
            scale = config['init_params_scale'] if 'init_params_scale' in config else 0.001
            params = self._init_params(nb_users, nb_movies, nb_latent_f, nb_user_pref, scale=scale)
            self._set_params(params)

    def _init_params(self, nb_users, nb_movies, nb_latent_f, nb_user_pref, scale=0.001):
        user_factors = np.random.normal(scale=scale, size=(nb_users, nb_latent_f)) # user latent factor matrix
        item_factors = np.random.normal(scale=scale, size=(nb_movies, nb_latent_f)) # item latent factor matrix
        user_interest_factors = np.random.normal(scale=scale, size=(nb_users, nb_user_pref, nb_latent_f)) # user latent factor tensor
        item_bias = np.zeros((nb_movies, 1)) # item bias vector
        user_interest_bias = np.zeros((nb_users, nb_user_pref)) # user-interest bias matrix

        params = {'P': user_factors, 'Q': item_factors, 'W': user_interest_factors, 'b_i': item_bias, 'B_u': user_interest_bias}
        return params

    def _get_params(self):
        params = {'P': self.user_factors, 'Q': self.item_factors, 'W': self.user_interest_factors, 'b_i': self.item_bias,
                  'B_u': self.user_interest_bias, 'avg_train_rating': self.global_bias}
        if self.si_item_model is not None:
            params['si_item_nn_params'] = self.si_item_model.param_values
        return params

    def _set_params(self, params):
        self.user_factors = params['P']
        self.item_factors = params['Q']
        self.user_interest_factors = params['W']
        self.item_bias = params['b_i']
        self.user_interest_bias = params['B_u']
        self.global_bias = params['avg_train_rating'] if 'avg_train_rating' in params else None

    def _get_local_pref(self, u, i):
        max_score = False
        local_pref = 0
        for t in range(self.config['nb_user_pref']):
            score = self.user_interest_bias[u, t] + np.dot(self.user_interest_factors[u, t, :], self.item_factors[i, :].T)
            if not max_score or score > max_score:
                max_score = score
                local_pref = t
        return local_pref, max_score

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
        local_pref, local_pref_score = self._get_local_pref(u, i)
        r_predict = self.global_bias + self.item_bias[i] + np.dot(self.user_factors[u, :], self.item_factors[i, :].T) + local_pref_score
        return r_predict

    def fit(self, train, val=None, test=None, zero_sampler=None):

        config = self.config
        lr = config['lr']
        reg_lambda = config['reg_lambda']

        si_item_lr = config['si_item_lr'] if 'si_item_lr' in config else None
        si_item_lambda_d_item_f = config['si_item_lambda_d_item_f'] if 'si_item_lambda_d_item_f' in config else None

        si_user_lr = config['si_user_lr'] if 'si_user_lr' in config else None
        si_user_lambda_d_user_f = config['si_user_lambda_d_user_f'] if 'si_user_lambda_d_user_f' in config else None

        if config['use_avg_rating']:
            self.global_bias = train['rating'].mean()
        else:
            self.global_bias = 0

        verbose = 'verbose' in config and config['verbose'] > 0

        train_rmse = []
        val_rmse = []
        item_feature_loss = []
        user_feature_loss = []

        # AdaGrad caches
        if 'adagrad' in config and config['adagrad']:
            adac_global_bias = 0
            adac_item_bias = np.zeros_like(self.item_bias)
            adac_user_interest_bias = np.zeros_like(self.user_interest_bias)
            adac_user_factors = np.zeros_like(self.user_factors)
            adac_item_factors = np.zeros_like(self.item_factors)
            adac_user_interest_factors = np.zeros_like(self.user_interest_factors)
            ada_eps = config['ada_eps']

        if verbose:
            print "Start training ..."

        for epoch in range(config['nb_epochs']):
            if verbose:
                print "epoch {}, lr {}, si_item_lr {}".format(epoch, lr, si_item_lr)

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
            current_item_feature_losses = []
            current_user_feature_losses = []

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

                local_pref, local_pref_score = self._get_local_pref(u, i)

                # main model - predict rating and calc rating error
                rating_predict = self.global_bias + \
                                 self.item_bias[i] +\
                                 np.dot(self.user_factors[u, :], self.item_factors[i, :].T) +\
                                 local_pref_score

                rating_error = rating - rating_predict
                rating_errors.append(float(rating_error))

                # calc gradients
                d_global_bias = rating_error
                d_item_bias = rating_error - reg_lambda * self.item_bias[i]
                d_user_interest_bias = rating_error - reg_lambda * self.user_interest_bias[u, local_pref]
                d_user_factors = rating_error * self.item_factors[i, :] - reg_lambda * self.user_factors[u, :]
                d_user_interest_factors = rating_error * self.item_factors[i, :] - \
                                          reg_lambda * self.user_interest_factors[u, local_pref, :]
                d_item_factors = rating_error * (self.user_factors[u, :] + self.user_interest_factors[u, local_pref, :]) - \
                                 reg_lambda * self.item_factors[i, :]

                # side information model - predict feature vector, calculate feature vector error
                if self.si_item_model is not None:
                    item_factors_reshaped = np.reshape(self.item_factors[i, :], (1, -1))
                    feature_loss, si_d_item_factors = self.si_item_model.step(item_factors_reshaped, movie_id, si_item_lr)
                    current_item_feature_losses.append(float(feature_loss))

                    # update parameters
                    d_item_factors -= si_item_lambda_d_item_f * si_d_item_factors.flatten()

                # side information model - predict feature vector, calculate feature vector error
                if self.si_user_model is not None:
                    user_factors_reshaped = np.reshape(self.user_factors[u, :], (1, -1))
                    feature_loss, si_d_user_factors = self.si_user_model.step(user_factors_reshaped, user_id, si_user_lr)
                    current_user_feature_losses.append(float(feature_loss))

                    # update parameters
                    d_user_factors -= si_user_lambda_d_user_f * si_d_user_factors.flatten()

                # update AdaGrad caches and parameters
                if 'adagrad' in config and config['adagrad']:
                    adac_global_bias += d_global_bias ** 2
                    adac_item_bias[i] += d_item_bias ** 2
                    adac_user_interest_bias[u, local_pref] += d_user_interest_bias ** 2
                    adac_user_factors[u, :] += d_user_factors ** 2
                    adac_item_factors[i, :] += d_item_factors ** 2
                    adac_user_interest_factors[u, local_pref, :] += d_user_interest_factors ** 2

                    # update parameters
                    self.global_bias += lr * d_global_bias / (np.sqrt(adac_global_bias) + ada_eps)
                    self.item_bias[i] += lr * d_item_bias / (np.sqrt(adac_item_bias[i]) + ada_eps)
                    self.user_interest_bias[u, local_pref] += lr * d_user_interest_bias / (np.sqrt(adac_user_interest_bias[u, local_pref]) + ada_eps)
                    self.user_factors[u, :] += lr * d_user_factors / (np.sqrt(adac_user_factors[u, :]) + ada_eps)
                    self.user_interest_factors[u, local_pref, :] += lr * d_user_interest_factors / (np.sqrt(adac_user_interest_factors[u, local_pref, :]) + ada_eps)
                    self.item_factors[i, :] += lr * d_item_factors / (np.sqrt(adac_item_factors[i, :]) + ada_eps)
                else:
                    self.global_bias += lr * d_global_bias
                    self.item_bias[i] += lr * d_item_bias
                    self.user_interest_bias[u, local_pref] += lr * d_user_interest_bias
                    self.user_factors[u, :] += lr * d_user_factors
                    self.user_interest_factors[u, local_pref, :] += lr * d_user_interest_factors
                    self.item_factors[i, :] += lr * d_item_factors

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

            if 'si_item_lr_decay' in config:
                si_item_lr *= (1.0 - config['si_item_lr_decay'])
            if 'si_user_lr_decay' in config:
                si_user_lr *= (1.0 - config['si_user_lr_decay'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            if verbose:
                print "Train RMSE:", current_rmse

            if len(current_item_feature_losses) > 0:
                current_avg_feature_loss = np.sqrt(np.mean(current_item_feature_losses))
                item_feature_loss.append(current_avg_feature_loss)
                if verbose:
                    print "Item Avg Feature Loss:", current_avg_feature_loss
            if len(current_user_feature_losses) > 0:
                current_avg_feature_loss = np.sqrt(np.mean(current_user_feature_losses))
                user_feature_loss.append(current_avg_feature_loss)
                if verbose:
                    print "User Avg Feature Loss:", current_avg_feature_loss

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
        history = {'train_rmse': train_rmse,
                   'val_rmse': val_rmse,
                   'item_feature_loss': item_feature_loss,
                   'user_feature_loss': user_feature_loss,
                   'test_rmse': test_rmse}
        return history

    def fit_alternate(self, train, val=None, test=None, zero_sampler=None):
        config = self.config
        lr = config['lr']
        reg_lambda = config['reg_lambda']

        si_item_lr = config['si_item_lr'] if 'si_item_lr' in config else None
        si_item_l_d_item_f = config['si_item_l_d_item_f'] if 'si_item_l_d_item_f' in config else None

        si_user_lr = config['si_user_lr'] if 'si_user_lr' in config else None
        si_user_l_d_user_f = config['si_user_l_d_user_f'] if 'si_user_l_d_user_f' in config else None

        if config['use_avg_rating']:
            self.global_bias = train['rating'].mean()
        else:
            self.global_bias = 0

        verbose = 'verbose' in config and config['verbose'] > 0

        train_rmse = []
        val_rmse = []
        item_feature_rmse = []
        user_feature_rmse = []

        # AdaGrad caches
        if 'adagrad' in config and config['adagrad']:
            adac_item_bias = np.zeros_like(self.item_bias)
            adac_user_interest_bias = np.zeros_like(self.user_interest_bias)
            adac_user_factors = np.zeros_like(self.user_factors)
            adac_item_factors = np.zeros_like(self.item_factors)
            adac_user_interest_factors = np.zeros_like(self.user_interest_factors)
            ada_eps = config['ada_eps']

        if verbose:
            print "Start training ..."

        for epoch in range(config['nb_epochs']):
            if verbose:
                print "epoch {}, lr {}, si_item_lr {}".format(epoch, lr, si_item_lr)

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
            item_feature_losses = []
            user_feature_losses = []

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

                local_pref, local_pref_score = self._get_local_pref(u, i)

                # main model - predict rating and calc rating error
                rating_predict = self.global_bias + \
                                 self.item_bias[i] + \
                                 np.dot(self.user_factors[u, :], self.item_factors[i, :].T) + \
                                 local_pref_score

                rating_error = rating - rating_predict
                rating_errors.append(float(rating_error))

                # calc gradients
                d_item_bias = rating_error - reg_lambda * self.item_bias[i]
                d_user_interest_bias = rating_error - reg_lambda * self.user_interest_bias[u, local_pref]
                d_user_factors = rating_error * self.item_factors[i, :] - reg_lambda * self.user_factors[u, :]
                d_user_interest_factors = rating_error * self.item_factors[i, :] - \
                                          reg_lambda * self.user_interest_factors[u, local_pref, :]
                d_item_factors = rating_error * (self.user_factors[u, :] + self.user_interest_factors[u, local_pref, :]) - \
                                 reg_lambda * self.item_factors[i, :]

                # update AdaGrad caches and parameters
                if 'adagrad' in config and config['adagrad']:
                    adac_item_bias[i] += d_item_bias ** 2
                    adac_user_interest_bias[u, local_pref] += d_user_interest_bias ** 2
                    adac_user_factors[u, :] += d_user_factors ** 2
                    adac_item_factors[i, :] += d_item_factors ** 2
                    adac_user_interest_factors[u, local_pref, :] += d_user_interest_factors ** 2

                    # update parameters
                    self.item_bias[i] += lr * d_item_bias / (np.sqrt(adac_item_bias[i]) + ada_eps)
                    self.user_interest_bias[u, local_pref] += lr * d_user_interest_bias / (
                        np.sqrt(adac_user_interest_bias[u, local_pref]) + ada_eps)
                    self.user_factors[u, :] += lr * d_user_factors / (np.sqrt(adac_user_factors[u, :]) + ada_eps)
                    self.user_interest_factors[u, local_pref, :] += lr * d_user_interest_factors / (
                        np.sqrt(adac_user_interest_factors[u, local_pref, :]) + ada_eps)
                    self.item_factors[i, :] += lr * d_item_factors / (np.sqrt(adac_item_factors[i, :]) + ada_eps)
                else:
                    self.item_bias[i] += lr * d_item_bias
                    self.user_interest_bias[u, local_pref] += lr * d_user_interest_bias
                    self.user_factors[u, :] += lr * d_user_factors
                    self.user_interest_factors[u, local_pref, :] += lr * d_user_interest_factors
                    self.item_factors[i, :] += lr * d_item_factors

                # update progess bar
                if verbose:
                    progress += 1
                    bar.update(progress)

            if verbose:
                bar.finish()

            # side information model - predict feature vector, calculate feature vector error
            if self.si_item_model is not None:
                movie_ids = train_for_epoch['movie_id'].unique()
                if verbose:
                    total = len(movie_ids)
                    bar = progressbar.ProgressBar(max_value=total)
                    bar.start()
                    progress = 0

                for movie_id in movie_ids:
                    i = self.items[movie_id]
                    item_factors_reshaped = np.reshape(self.item_factors[i, :], (1, -1))
                    feature_loss, si_d_item_factors = self.si_item_model.step(item_factors_reshaped, movie_id, si_item_lr)
                    item_feature_losses.append(float(feature_loss))

                    # update parameters
                    d_item_factors = si_item_l_d_item_f * si_d_item_factors.flatten()
                    if 'adagrad' in config and config['adagrad']:
                        adac_item_factors[i, :] += d_item_factors ** 2
                        self.item_factors[i, :] += lr * d_item_factors / (np.sqrt(adac_item_factors[i, :]) + ada_eps)
                    else:
                        self.item_factors[i, :] += lr * d_item_factors

                    # update progess bar
                    if verbose:
                        progress += 1
                        bar.update(progress)
                if verbose:
                    bar.finish()

            # side information model - predict feature vector, calculate feature vector error
            if self.si_user_model is not None:
                user_ids = train_for_epoch['user_id'].unique()

                if verbose:
                    total = len(user_ids)
                    bar = progressbar.ProgressBar(max_value=total)
                    bar.start()
                    progress = 0

                for user_id in user_ids:
                    user_factors_reshaped = np.reshape(self.user_factors[u, :], (1, -1))
                    feature_loss, si_d_user_factors = self.si_user_model.step(user_factors_reshaped, user_id, si_user_lr)
                    user_feature_losses.append(float(feature_loss))

                    # update parameters
                    d_user_factors = si_user_l_d_user_f * si_d_user_factors.flatten()

                    if 'adagrad' in config and config['adagrad']:
                        adac_user_factors[u, :] += d_user_factors ** 2
                        self.user_factors[u, :] += lr * d_user_factors / (np.sqrt(adac_user_factors[u, :]) + ada_eps)
                    else:
                        self.user_factors[u, :] += lr * d_user_factors

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

            if 'si_item_lr_decay' in config:
                si_item_lr *= (1.0 - config['si_item_lr_decay'])
            if 'si_user_lr_decay' in config:
                si_user_lr *= (1.0 - config['si_user_lr_decay'])

            # report error
            current_rmse = np.sqrt(np.mean(np.square(rating_errors)))
            train_rmse.append(current_rmse)
            if verbose:
                print "Train RMSE:", current_rmse

            if len(item_feature_losses) > 0:
                current_feature_rmse = np.sqrt(np.mean(item_feature_losses))
                item_feature_rmse.append(current_feature_rmse)
                if verbose:
                    print "Item Feature RMSE:", current_feature_rmse
            if len(user_feature_losses) > 0:
                current_feature_rmse = np.sqrt(np.mean(user_feature_losses))
                user_feature_rmse.append(current_feature_rmse)
                if verbose:
                    print "User Feature RMSE:", current_feature_rmse

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
        history = {'train_rmse': train_rmse,
                   'val_rmse': val_rmse,
                   'item_feature_rmse': item_feature_rmse,
                   'user_feature_rmse': user_feature_rmse,
                   'test_rmse': test_rmse}
        return history
