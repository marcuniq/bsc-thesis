import lasagne
import theano
import theano.tensor as T


class UserPrefModel(object):
    def __init__(self, config):
        Qi = T.matrix('Qi', dtype=theano.config.floatX)
        Pu = T.matrix('Pu', dtype=theano.config.floatX)
        movie_d2v = T.matrix('movie_d2v', dtype=theano.config.floatX)

        r = T.scalar('r', dtype=theano.config.floatX)

        input_layers = []
        input_layers.append(lasagne.layers.InputLayer(shape=(None, config['nb_latent_f']), input_var=Qi))
        input_layers.append(lasagne.layers.InputLayer(shape=(None, config['nb_latent_f']), input_var=Pu))
        if 'user_pref_movie_d2v' in config and config['user_pref_movie_d2v']:
            input_layers.append(lasagne.layers.InputLayer(shape=(None, config['nb_d2v_features']), input_var=movie_d2v))

        network = lasagne.layers.ConcatLayer(incomings=input_layers)
        for i in range(0, len(config['user_pref_hidden_dim'])-1):
            n_out = config['user_pref_hidden_dim'][i]
            network = lasagne.layers.DenseLayer(network, num_units=n_out, nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(network, num_units=config['user_pref_hidden_dim'][-1])
        self.network = network

        r_predict = lasagne.layers.get_output(self.network)

        loss = lasagne.objectives.squared_error(r, r_predict).sum()
        loss += config['user_pref_reg_lambda'] * lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)

        dQi = theano.grad(loss, Qi)
        dPu = theano.grad(loss, Pu)
        lr = T.scalar('lr', dtype=theano.config.floatX)

        # create parameter update expressions
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.sgd(loss, params, lr)

        self.param_values = lasagne.layers.get_all_param_values(self.network)

        self.predict = theano.function([Qi, Pu, movie_d2v], r_predict, allow_input_downcast=True)
        self.gradient_step = theano.function([Qi, Pu, movie_d2v, r, lr], outputs=[r_predict, loss, dQi, dPu],
                                        updates=updates, allow_input_downcast=True)

    def set_params(self, param_values):
        lasagne.layers.set_all_param_values(self.network, param_values)
