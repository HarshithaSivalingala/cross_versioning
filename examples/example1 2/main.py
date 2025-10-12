import tensorflow as tf
import numpy as np

# TensorFlow 1.x session-based code
sess = tf.Session()

# Create placeholders
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# Simple neural network using tf.layers
hidden = tf.layers.dense(x, 4, activation=tf.nn.relu)
output = tf.layers.dense(hidden, 1)

# Loss and optimizer
loss = tf.reduce_mean(tf.square(output - y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Initialize variables
sess.run(tf.global_variables_initializer())

# Sample data
X_data = np.random.random((100, 2))
y_data = np.random.random((100, 1))

# Training loop
for i in range(100):
    _, loss_val = sess.run([optimizer, loss], 
                          feed_dict={x: X_data, y: y_data})
    if i % 20 == 0:
        print(f"Step {i}, Loss: {loss_val}")

# Clean up
sess.close()
