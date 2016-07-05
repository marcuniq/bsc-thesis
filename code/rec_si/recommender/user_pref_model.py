import lasagne
import theano
import theano.tensor as T
import numpy as np


class UserPrefModel(object):
    def __init__(self, config):
        item_factors = T.matrix('item_factors', dtype=theano.config.floatX)
        user_factors = T.matrix('user_factors', dtype=theano.config.floatX)
        item_feature_vec = T.matrix('item_feature_vec', dtype=theano.config.floatX)

        rating_target = T.scalar('rating_target', dtype=theano.config.floatX)
        rating_mf = T.scalar('rating_mf', dtype=theano.config.floatX)

        input_layers = []
        input_layers.append(lasagne.layers.InputLayer(shape=(None, config['nb_latent_f']), input_var=item_factors))
        input_layers.append(lasagne.layers.InputLayer(shape=(None, config['nb_latent_f']), input_var=user_factors))
        if 'user_pref_input_movie_d2v' in config and config['user_pref_input_movie_d2v']:
            input_layers.append(lasagne.layers.InputLayer(shape=(None, config['nb_d2v_features']), input_var=item_feature_vec))

        network = lasagne.layers.ConcatLayer(incomings=input_layers)
        for i in range(0, len(config['user_pref_hidden_dim'])):
            n_out = config['user_pref_hidden_dim'][i]
            network = lasagne.layers.DenseLayer(network, num_units=n_out)
        network = lasagne.layers.DenseLayer(network, num_units=1,
                                            nonlinearity=lasagne.nonlinearities.identity)

        nn_ui = lasagne.layers.get_output(network).sum()
        rating_predict = rating_mf + nn_ui

        loss = 0.5 * (rating_target - rating_predict) ** 2
        loss += 0.5 * config['user_pref_reg_lambda'] * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

        d_item_factors = theano.grad(loss, item_factors)
        d_user_factors = theano.grad(loss, user_factors)
        lr = T.scalar('lr', dtype=theano.config.floatX)

        # create parameter update expressions
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, lr)

        self.param_values = lasagne.layers.get_all_param_values(network)
        self.network = network

        self.predict = theano.function([item_factors, user_factors, item_feature_vec],
                                       nn_ui,
                                       allow_input_downcast=True, on_unused_input='ignore')
        self.gradient_step = theano.function([item_factors, user_factors, item_feature_vec, rating_mf, rating_target, lr],
                                             outputs=[nn_ui, loss, d_item_factors, d_user_factors],
                                             updates=updates, allow_input_downcast=True, on_unused_input='ignore')

    def set_params(self, param_values):
        params_float32 = map(lambda arr: arr.astype(np.float32), param_values)
        lasagne.layers.set_all_param_values(self.network, params_float32)
