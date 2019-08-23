######################################################
# Simple MNIST example
# Mark Harvey
# April 2019
######################################################
import os
import sys
import shutil
import numpy as np
import tensorflow as tf


#####################################################
# Set up directories
#####################################################

SCRIPT_DIR = os.getcwd()

TRAIN_GRAPH = 'training_graph.pb'
CHKPT_FILE = 'float_model.ckpt'

CHKPT_DIR = os.path.join(SCRIPT_DIR, 'chkpts')
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
CHKPT_PATH = os.path.join(CHKPT_DIR, CHKPT_FILE)
MNIST_DIR = os.path.join(SCRIPT_DIR, 'mnist_dir')


# create a directory for the MNIST dataset if it doesn't already exist
if not (os.path.exists(MNIST_DIR)):
    os.makedirs(MNIST_DIR)
    print("Directory " , MNIST_DIR ,  "created ") 


# create a directory for the TensorBoard data if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 


# create a directory for the checkpoint if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(CHKPT_DIR)):
    shutil.rmtree(CHKPT_DIR)
os.makedirs(CHKPT_DIR)
print("Directory " , CHKPT_DIR ,  "created ")



#####################################################
# Hyperparameters
#####################################################
LEARN_RATE = 0.0001
BATCHSIZE = 50
EPOCHS = 3


#####################################################
# Dataset preparation
#####################################################
# MNIST dataset has 60k images. Training set is 60k, test set is 10k.
# Each image is 28x28x8bits
mnist_dataset = tf.keras.datasets.mnist.load_data('mnist_data')
(x_train, y_train), (x_test, y_test) = mnist_dataset

# scale pixel values from 0:255 to 0:1
# Also converts uint8 values to float
x_train = (x_train/255.0)  
x_test = (x_test/255.0)

# reshape train & test images to [None, 28, 28, 1]
x_train = np.reshape(x_train, [-1, 28, 28, 1])
x_test = np.reshape(x_test, [-1, 28, 28, 1])

# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# take 5000 images from train set to make a dataset for prediction
x_valid = x_train[55000:]
y_valid = y_train[55000:]

# reduce train dataset to 55000 images
y_train = y_train[:55000]
x_train = x_train[:55000]

# calculate total number of batches
total_batches = int(len(x_train)/BATCHSIZE)



#####################################################
# Create the Computational graph
#####################################################

# define placeholders for the input data & labels
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images_in')
y = tf.placeholder(tf.float32, [None, 10], name='labels_in')



# define the CNN
def cnn(x):
  '''
  Build the convolution neural network
  arguments:
    inputs: the input tensor - shape must be [None,28,28,1]
  '''
  net = tf.layers.conv2d(x, 16, [3, 3], activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
  net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.relu)
  net = tf.layers.flatten(net)
  net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
  logits = tf.layers.dense(net, units=10, activation=None)
  return logits


# build the network, input comes from the 'x' placeholder
logits = cnn(x)

# softmax cross entropy loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

# Adaptive Momentum optimizer - minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE, name='Adam').minimize(loss)


# Check to see if the prediction matches the label
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

 # Calculate accuracy as mean of the correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorBoard data collection
tf.summary.scalar('cross_entropy_loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('input_images', x)


# set up saver object
saver = tf.train.Saver()



#####################################################
# Run the graph in a Session
#####################################################
# Launch the graph
with tf.Session() as sess:

    sess.run(tf.initializers.global_variables())
    
    # TensorBoard writer
    writer = tf.summary.FileWriter(TB_LOG_DIR, sess.graph)
    tb_summary = tf.summary.merge_all()

    # Training phase with training data
    print ('-------------------------------------------------------------')
    print ('TRAINING PHASE')
    print ('-------------------------------------------------------------')
    for epoch in range(EPOCHS):
        print ("Epoch", epoch+1, "/", EPOCHS)

        # process all batches
        for i in range(total_batches):
            
            # fetch a batch from training dataset
            batch_x, batch_y = x_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE], y_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE]

            # Run graph for optimization, loss, accuracy - i.e. do the training
            _, s = sess.run([optimizer, tb_summary], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(s, (epoch*total_batches + i))
            # Display accuracy per 100 batches
            if i % 100 == 0:
              acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
              print (' Step: {:4d}  Training accuracy: {:1.4f}'.format(i,acc))


    print ('-------------------------------------------------------------')
    print ('FINISHED TRAINING')
    print('Run `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % TB_LOG_DIR)
    print ('-------------------------------------------------------------')
    writer.flush()
    writer.close()


    # Evaluation phase with test dataset
    print ('EVALUATION PHASE:')
    print ("Final Accuracy with validation set:", sess.run(accuracy, feed_dict={x: x_valid, y: y_valid}))
    print ('-------------------------------------------------------------')

    # save post-training checkpoint & graph
    print ('SAVING:')
    save_path = saver.save(sess, os.path.join(CHKPT_DIR, CHKPT_FILE))
    print('Saved checkpoint to %s' % os.path.join(CHKPT_DIR,CHKPT_FILE))
    tf.train.write_graph(sess.graph_def, CHKPT_DIR, TRAIN_GRAPH, as_text=False)
    print('Saved binary graphDef to %s' % os.path.join(CHKPT_DIR,TRAIN_GRAPH))
    print ('-------------------------------------------------------------')


#####  SESSION ENDS HERE #############


