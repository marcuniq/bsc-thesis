import lasagne
import theano
import theano.tensor as T


class SideInfoModel(object):
    def __init__(self, dim, reg_lambda):
        Qi = T.matrix('Qi', dtype=theano.config.floatX)
        y = T.vector('y', dtype=theano.config.floatX)

        network = lasagne.layers.InputLayer(shape=(None, dim[0]), input_var=Qi)
        for i in range(1, len(dim)-1):
            n_out = dim[i]
            network = lasagne.layers.DenseLayer(network, num_units=n_out, nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(network, num_units=dim[-1])
        y_predict = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(y, y_predict).sum()
        loss += reg_lambda * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

        dQi = theano.grad(loss, Qi)
        lr = T.scalar('lr', dtype=theano.config.floatX)

        # create parameter update expressions
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, lr)

        self.gradient_step = theano.function([Qi, y, lr], outputs=[loss, dQi], updates=updates, allow_input_downcast=True)
