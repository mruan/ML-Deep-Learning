""" Convolutional Restricted Boltzman Machine """

import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

class CRBM(object):
    """ Convolutional Restricted Boltzman Machine"""
    def __init__(self, input=None, filter_shape, image_shape, poolsize, \
                     W=None, hbias=None, vbias=None, numpy_rng=None,
                     theano_rng=None):

        """
        Convolutional RBM constructor.

        :param input: None for standalone RBMs or symbolic variable if RBM is part of a larger graph

        :type filter_shape: tuple or list of length 4
        :param filter_shape: [# of filters, # input feature maps, filter height, filter width];

        :type image_shape: tuple or list of length 4
        :param image_shape: [batch_size, # input feature maps, image height, image width];

        :type poolsize: tuple or list of length 2
        :param poolsize: down sample factor

        :type W: theano tensor
        :param W: None for standalone CRBM, otherwise it is a symbolic variable pointing to a
                  shared Weight matrix in a Conv-Pool layer in CNN

        :type hbias: theano tensor
        :param hbias: None for standalone CRBM, otherwise it is a symbolic variable pointing to a
                      shared hidden units bias vector.

        :type vbias: theano tensor
        :param vbias: None for standalone CRBMs or a symbolic variable pointing to a shared
                      visible units bias.
        """

        assert image_shape[1] == filter_shape[1]
        
        self.input = input
        if not input:
            self.input = T.matrix('input')

        if W is None:
            # there are "# input feature maps * filter height * filter width"
            # inputs into this layer unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "# output feature maps * filter height * filter width"
            W_bound= numpy.sqrt(6. / fan_in + fan_out)
            init_w = numpy.asarray(numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                   dtype = theano.config.floatX)

            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros((filter_shape[0],), 
                                                    dtype=theano.config.floatX), 
                                  name='hbias', borrow= True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=numpy.zeros((filter_shape[1],),
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow= True)

        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.FS = FS
        self.IS = IS

        ## some parameters for sampling h->v
        # Output now becomes input, input channels are outputs
        self.FSinv = (FS[1], FS[0], FS[2], FS[3])

        # Same batch size, # input=# output filter maps, 
        # image size = ImageHight-FilterHeight+1, ImageWidth-FilterWidth+1
        self.ISinv = (IS[0], FS[0], IS[2]-FS[2]+1, IS[3]-FS[3]+1)

        self.theano_rng = theano_rng # why do I need it here?--Minghao

### TODO: Figure out a way to put this
        conv_v2h = conv.conv2d(input=input, filters=self.W, 
                               filter_shape=filter_shape, image_shape=image_shape)

#        conv_h2v = conv.conv2d(

### end of TODO

        self.params = [self.W, self.hbias, self.vhias]

    # Convolve input visible samples up the the hidden units
    def conv_upward(self, v_sample):
        pre_sigm_h = conv.conv2d(v_sample, filters=self.W, 
                                 filter_shape=self.FS, 
                                 image_shape=self.IS) + self.hbias.dimshuffle('x', 0, 'x', 'x')
        return pre_sigm_h

    # Convolve output hidden down to the visible units (although it's technically not convolution)
    def conv_downward(self, h_sample):
        Wp = self.W.dimshuffle(1,0,2,3) # change input<->output
        # each filter has to be fliped horizontally and vertically
        pre_sigm_v = conv.conv2d(h_sample, filters=W[:,:,::-1,::-1],
                                 filter_shape= self.FSinv,
                                 image_shape = self.ISinv) + self.vbias.dimshuffle('x',0,'x','x')

    def free_energy(self, v_sample):
        ''' Compute the free energy '''
        conv_out = conv.conv2d(v_sample, filters=self.W,
                               filter_shape=self.FS, image_shape= self.IS)

        # reshape hbias to shape (1, # filters, 1, 1) which will be broadcasted over other dims        
        wx_b = conv_out + self.hbias.dimshuffle('x', 0, 'x', 'x')
        
        hidden_term = T.sum(T.log(1+T.exp(wx_b))) # Question me if think this makes no sense--Minghao

        # reshape vbias to shape (1, # input channels, 1, 1) which will be broadcasted over other dims
        vbias_term = T.sum(v_sample * self.vbias.dimshuffle('x', 0, 'x', 'x'))

        return -hidden_term - vbias_term

    def propup(self, vis):
        ''' This function propagates the visible units activation upwards to the hidden units
        The pre-sigmoid activation of the layer is also computed
        '''
        pre_sigm_h = conv_upward(vis)
        return [pre_sigm_h, T.nnet.sigmoid(pre_sigm_h)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # cimpute the activation of the hidden units given a sample of the visible
        pre_sigm_h1, h1_mean = self.propup(v0_sample)
        # batch Gibbs Sampling
        h1_sample = self.theano_rng.binomial(size=hi_mean.shape, n=1, p=h1_mean,
                                             dtype = theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        ''' This function propagates the hidden units downwards to visible units
        '''
        pre_sigm_v = conv_downward(hid)
        return [pre_sigm_v, T.nnet.sigmoid(pre_sigm_v)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigm_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigm_v1, v1_mean, v1_sample]

    ## TODO: add energy function and sampling method for Gaussian
    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigm_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigm_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigm_v1, v1_mean, v1_sample, 
                pre_sigm_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
         ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigm_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigm_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigm_h1, h1_mean, h1_sample, 
                pre_sigm_v1, v1_mean, v1_sample]

    
# the rest not so sure...
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                    dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))

        return cross_entropy



