#####################################################
# Evaluation of frozen/quantized graph
# using MNIST dataset downloaded via Keras
#####################################################


import os
import sys
import argparse
import shutil
import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile



def graph_eval(input_graph_def, input_node, output_node):

    # MNIST dataset    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = (x_test/255.0).astype(np.float32)
    x_test = np.reshape(x_test, [-1, 28, 28, 1])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    tf.import_graph_def(input_graph_def,name = '')

    # Get input placeholders & tensors
    images_in = tf.get_default_graph().get_tensor_by_name(input_node+':0')
    labels = tf.placeholder(tf.int32,shape = [None,10])

    # get output tensors
    logits = tf.get_default_graph().get_tensor_by_name(output_node+':0')

    # top 5 and top 1 accuracy
    in_top5 = tf.nn.in_top_k(predictions=logits, targets=tf.argmax(labels, 1), k=5)
    in_top1 = tf.nn.in_top_k(predictions=logits, targets=tf.argmax(labels, 1), k=1)
    top5_acc = tf.reduce_mean(tf.cast(in_top5, tf.float32))
    top1_acc = tf.reduce_mean(tf.cast(in_top1, tf.float32))
    
    # Create the Computational graph
    with tf.Session() as sess:
        
        sess.run(tf.initializers.global_variables())
      
        feed_dict={images_in: x_test, labels: y_test}
        t5_acc,t1_acc = sess.run([top5_acc,top1_acc], feed_dict)
    
        print (' Top 1 accuracy with validation set: {:1.4f}'.format(t1_acc))
        print (' Top 5 accuracy with validation set: {:1.4f}'.format(t5_acc))

    print ('FINISHED!')
    return



def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.gfile.FastGFile(FLAGS.graph, "rb").read())
    graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str,
                        default='./quantize_results/quantize_eval_model.pb',
                        help='graph file (.pb) to be evaluated.')
    parser.add_argument('--input_node', type=str,
                        default='images_in',
                        help='input node.')
    parser.add_argument('--output_node', type=str,
                        default='dense_1/BiasAdd',
                        help='output node.')
    parser.add_argument('--class_num', type=int,
                        default=10,
                        help='number of classes.') 
    parser.add_argument('--gpu', type=str,
                        default='0',
                        help='gpu device id.')   
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


