#following the guide on https://www.tensorflow.org/get_started/get_started

import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
#print(node1, node2)

#evaluate nodes by running the computational graph in a session
sess = tf.Session()
#print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
#print("node3:", node3)
#print("sess.run(node3):", sess.run(node3))

#graph can be parametrized to accept external input, known as a placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

#print(sess.run(adder_node, {a: 3, b: 4.5}))
#print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

add_and_triple = adder_node * 3.
#print(sess.run(add_and_triple, {a: 3, b: 4.5}))

#variables allow us to add trainable parameters to a graph, constructed with type and init value
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b

#must call the following to initalize variables
init = tf.global_variables_initializer()
sess.run(init)
#print(sess.run(linear_model, {x:[1,2,3,4]}))

#printing the loss value of the original model with a loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#compare the functions values with the expected values to calculate loss
print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))


#fix the values of w and b to fit the model perfectly to see the resuling loss value
fixw = tf.assign(w, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixw, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0, -1, -2, -3]}))
