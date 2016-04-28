import lasagne
import theano
import theano.tensor as T


def build_si_model(dim, reg_lambda):
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
    updates = lasagne.updates.sgd(loss, params, lr)

    #forward_prop = T.function([Qi], y_predict, allow_input_downcast=True)
    calc_loss = theano.function([Qi, y], loss, allow_input_downcast=True)
    get_qi_grad = theano.function([Qi, y], dQi, allow_input_downcast=True)
    gradient_step = theano.function([Qi, y, lr], updates=updates, allow_input_downcast=True)

    return calc_loss, get_qi_grad, gradient_step
