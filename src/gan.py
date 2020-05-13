import os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# G(z)
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)
    # 1st hidden layer
    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.leaky_relu(tf.matmul(x, w0) + b0)

    ### Code:ToDo (Change the architecture as CW2 Guidance required)

    w2 = tf.get_variable('G_w2', [h0.get_shape()[1], 512], initializer=w_init)
    b2 = tf.get_variable('G_b2', [512], initializer=b_init)
    h2 = tf.nn.leaky_relu(tf.matmul(h0, w2) + b2)

    w3 = tf.get_variable('G_w3', [h2.get_shape()[1], 1024], initializer=w_init)
    b3 = tf.get_variable('G_b3', [1024], initializer=b_init)
    h3 = tf.nn.leaky_relu(tf.matmul(h2, w3) + b3)
    
    # output hidden layer
    w1 = tf.get_variable('G_w1', [h3.get_shape()[1], 784], initializer=w_init)
    b1 = tf.get_variable('G_b1', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h3, w1) + b1)
    return o


# D(x)
def discriminator(x, drop_out):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)
    ###  Code: ToDO( Change the architecture as CW2 Guidance required)
    # 1st hidden layer

    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.get_variable('D_b0', [1024], initializer=b_init)
    h0 = tf.nn.leaky_relu(tf.matmul(x, w0) + b0)
    drop0 = tf.nn.dropout(h0, rate=drop_out)

    w2 = tf.get_variable('D_w2', [drop0.get_shape()[1], 512], initializer=w_init)
    b2 = tf.get_variable('D_b2', [512], initializer=b_init)
    h2 = tf.nn.leaky_relu(tf.matmul(drop0, w2) + b2)
    drop2 = tf.nn.dropout(h2, rate=drop_out)

    w3 = tf.get_variable('D_w3', [drop2.get_shape()[1], 256], initializer=w_init)
    b3 = tf.get_variable('D_b3', [256], initializer=b_init)
    h3 = tf.nn.leaky_relu(tf.matmul(drop2, w3) + b3)
    drop3 = tf.nn.dropout(h3, rate=drop_out)

    # output layer
    w1 = tf.get_variable('D_w1', [drop3.get_shape()[1], 1], initializer=w_init)
    b1 = tf.get_variable('D_b1', [1], initializer=b_init)
    h1 = tf.sigmoid(tf.matmul(drop3, w1) + b1)
    return h1


def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    z_ = np.random.normal(0, 1, (25, 100))    # z_ is the input of generator, every epochs will random produce input
    ##Code:ToDo complete the rest of part
    gen_images = sess.run(G_z, {z: z_})
   
    gen_images = gen_images.reshape(25,28,28)
    
    figsize = (50,50)
    
    plt.figure(figsize=figsize)
    for i in range(gen_images.shape[0]):
        plt.subplot(5,5,i+1)
        plt.imshow(gen_images[i])

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 100

tf.reset_default_graph()

# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

# networks : generator
with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, 100))
    G_z = generator(z)
# networks : discriminator
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.placeholder(tf.float32, shape=(None, 784))
    D_real = discriminator(x, drop_out)
    scope.reuse_variables()
    D_fake = discriminator(G_z, drop_out)

# loss for each network
eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network
t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# results save folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/results'):
    os.mkdir('MNIST_GAN_results/results')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(train_set.shape[0] // batch_size):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

    ### Code: TODO Code complete show_result function)
    p = 'MNIST_GAN_results/results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=p, show=False)

    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')
sess.close()



# batch size 100
# [100/100] - ptime: 41.75 loss_d: 1.221, loss_g: 0.901
# Avg per epoch ptime: 50.13, total 100 epochs ptime: 5405.01

# batch size 256
# [100/100] - ptime: 41.08 loss_d: 0.958, loss_g: 1.355
# Avg per epoch ptime: 33.81, total 100 epochs ptime: 3728.94