
import time
import numpy
import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

from conv_rbm import CRBM

from utils import tile_raster_images
from utils import load_data

learn_rate=0.01
train_epochs = 15
dataset='../data/mnist.pkl.gz'
batch_size = 50
n_kerns=20
n_chains = 20
n_samples=10
             
# Prepare the data set
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x,  test_set_y  = datasets[2]

# compute number of mini-batches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size
n_test_batches /= batch_size

# allocate symbolic variables for the data
index = TT.lscalar()  # index to a [mini]batch
x = TT.matrix('x')    # the data is presented as rasterized images
y = TT.ivector('y')   # the labels are presented as 1D vector of [int] labels

# Reshape the rasterized image of shape (batch_size, 28x28)
# to a 4D tensor, compatible with our crbm
rbm_input = x.reshape((batch_size, 1, 28, 28))

# also initialize the random stream generator
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# construct the CRBM class
# for this test, the parameters are not shared with another CNN
crbm = CRBM(input=rbm_input, 
            IS=(batch_size, 1, 28, 28),
            FS=(n_kerns, 1, 5, 5),
            numpy_rng = rng, theano_rng = theano_rng)

# initialize storage for the persistent chain (state = hidden layer or chain)
#persistent_chain = theano.shared(numpy.zeros(crbm.ISinv,
#                                             dtype=theano.config.floatX),
#                                 borrow = True)
                                             
# get the cost and the gradient corresponding to on step of CD-15
print 'Build computing graph'
cost, updates = crbm.get_cost_updates(lr= learn_rate, k=15)

##########################
# Training starts
##########################
# TODO: set up output image folder

train_crbm = theano.function([index], cost, updates=updates,
                             givens={x: train_set_x[index*batch_size: (index+1)*batch_size]},
                             name = 'train_crbm')

start_time = time.clock()
print 'Training Starts...'
# go through training epochs
for epoch in xrange(train_epochs):

    # go through the training set
    mean_cost = []
    for batch_index in xrange(n_train_batches):
        print 'On batch %d of %d' % (batch_index, n_train_batches)
        mean_cost += [train_crbm(batch_index)]

    print "Training epoch %d, cost is " % epoch, numpy.mean(mean_cost)
    
end_time = time.clock()

pretrain_time = (end_time - start_time)

print "Training took %f minutes" % (pretrain_time /60.)

# visualize the filters:
fflaten = theano.function([], TT.reshape(crbm.W, (FS[0]*FS[1], FS[2]*FS[3])), name='fflaten')
image_data = tile_raster_images(X=fflaten(), img_shape=(FS[2],FS[3]),
                                tile_shape = (FS[0],FS[1]), tile_spacing=(1,1))

image = PIL.Image.fromarray(image_data)
image.save('filter.png')

'''
#########################
# Sampling from the RBM #
#########################
# find out the number of test samples
num_test_samples = test_set_x.get_value(borrow=True).shape[0]

# pick random sample to initialize the persistent chain
test_idx = rng.randint(num_test_samples - n_chains)
pvis_train = theano.shared(numpy.asarray(
                    test_set_x.get_value(borrow=True)[test_idx:test_idx+n_chains], 
                    dtype = theano.config.floatX))

plot_every = 1000
# define one step of Gibbs sampling (mf = mean-field) define a 
# function that does 'plot_every' steps before returning the 
# sample for plotting
[presig_hs, h_mfs, h_samples, presig_vs, v_mfs, v_samples], updates \
    = theano.scan(crbm.gibbs_vhv,
                  outputs_info=[None, None, None, None, None, pvis_train],
                  n_steps = plot_every)
    
# add to updates the shared variable that takes care of our persistent chain:
updates.update({pvis_train: v_samples[-1]})

# construct the function that implements our persistent chain.
# we generate the "mean field" activations for plotting and the actual
# samples for reinitializing the state of our persistent chain
sample_fn = theano.function([], [v_mfs[-1], v_samples[-1]],
                            updates = updates, name = 'sample_fn')

# create a space to store the image for plotting
# (we need to leave room for the tile_spacing as well)
# TODO: finish the sampling code!!!
''' 