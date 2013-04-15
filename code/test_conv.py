import numpy
import scipy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv 

rng = numpy.random.RandomState(23455)
input = T.tensor4(name='input')

w_shape = (2, 3, 9, 9)
w_bound = numpy.sqrt(3*9*9)

W = theano.shared( numpy.asarray(rng.uniform(low=-1.0/w_bound,
                                             high=1.0/w_bound,
                                             size=w_shape),
                                 dtype=input.dtype), name='W')

Wp = W[:,:,::-1,::-1] #.dimshuffle(1,0,2,3)

b_shape = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low=-.5, high=.5, size=b_shape),
                                dtype = input.dtype), name='b')

conv_out = conv.conv2d(input, Wp)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))

f = theano.function([input], output)

print 'Done compling'