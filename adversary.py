import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

EPSILON = 0.01
SAMPLE_SIZE = 10
META_GRAPH = './model/deepnn.meta'
CKPT = './model'

# write_jpeg Source: https://stackoverflow.com/a/40322153/4989649
def write_jpeg(data, filepath):
  g = tf.Graph()
  with g.as_default():
    data_t = tf.placeholder(tf.uint8)
    op = tf.image.encode_jpeg(data_t, format='grayscale', quality=100)

  with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    data_np = sess.run(op, feed_dict={data_t: data})

  with open(filepath, 'w') as fd:
    fd.write(data_np)


def get_data(true_label,data_dir):
	mnist = input_data.read_data_sets(data_dir)
	x_test = mnist.test.images
	y_test = mnist.test.labels

	indexes = (y_test==true_label)
	x_test_sample = np.reshape(x_test[indexes][0:SAMPLE_SIZE],(SAMPLE_SIZE,784))
	return x_test_sample


def build_image(x_feed):
	output = []
	x_original = get_data(arguments['input_class'],arguments['data_dir'])

	for i in range(SAMPLE_SIZE):

		adversarial = np.array(x_feed[i]).reshape(28, 28, 1)
		original = np.array(x_original[i]).reshape(28, 28, 1)
		delta = np.abs(np.subtract(x_feed[i],x_original[i])).reshape(
					28, 28, 1)
		
	    #Image along each row: 28x3
		out = np.concatenate([original, delta, adversarial], axis=1)
		out = np.array(out).reshape(28, 84, 1)
		out = np.multiply(out,255)
		output.append(out)

	#10 rows as sample size	
	out = np.array(output).reshape(28*SAMPLE_SIZE,84,1)
	write_jpeg(out, 'image.jpg')	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--input_class',help='The class of the image to be misclassified',required=True,type=int)
	parser.add_argument('-t','--target_class',help='The target class of the misclassified image',required=True,type=int)
  	parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')	

	args = parser.parse_args()
	#Convert args to dictionary
	arguments = args.__dict__

	x_feed = get_data(arguments['input_class'],arguments['data_dir'])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		graph = tf.get_default_graph()

		#Restore the model
		saver = tf.train.import_meta_graph(META_GRAPH)
		saver.restore(sess, tf.train.latest_checkpoint(CKPT))	

		#Restore placeholders and tensors
		x = graph.get_tensor_by_name('x:0')
		y = graph.get_tensor_by_name('y_:0')
		keep_prob = graph.get_tensor_by_name('dropout/keep_prob:0')
		cse_loss = graph.get_tensor_by_name('cross-entropy:0')		
		acc = graph.get_tensor_by_name('acc:0')
		grads = tf.gradients(cse_loss, x, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)	

		for j in range(SAMPLE_SIZE):
			print "Creating adversarial image for image {}".format(j)
			feed_dict = {x:np.reshape(x_feed[j],(1,784)),y:np.reshape(arguments['target_class'],(1,)),keep_prob:1.0}

			i=0
			#Run the loop till the prediction is the target class
			while sess.run(acc,feed_dict={x:np.reshape(x_feed[j],(1,784)),y:np.reshape(arguments['target_class'],(1,)),keep_prob:1.0})!=1.0:
				print "\tStep: {}".format(i)

				#Get the gradients from J_{\theta}(x,y_target)
				cse=sess.run(cse_loss,feed_dict=feed_dict)
				grads_ = sess.run(grads,feed_dict=feed_dict)
				
				#Reshape gradients to match the input shape and update the image
				grads_ = np.asarray(grads_)
				grads_ = np.reshape(grads_,(1,784))
				x_feed[j] = x_feed[j]-(EPSILON*(np.sign(grads_)))		
				x_feed[j] = np.clip(x_feed[j], 0, 1.)

				i+=1

	build_image(x_feed)