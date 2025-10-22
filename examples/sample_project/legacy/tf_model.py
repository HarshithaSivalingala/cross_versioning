import tensorflow as tf


def build_graph(hidden_units: int = 8):
    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(tf.float32, shape=[None, 4], name="features")
        targets = tf.placeholder(tf.float32, shape=[None, 1], name="targets")

        dense = tf.layers.dense(features, hidden_units, activation=tf.nn.relu, name="hidden")
        predictions = tf.layers.dense(dense, 1, name="predictions")

        loss = tf.reduce_mean(tf.square(predictions - targets), name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01, name="optimizer")
        train_op = optimizer.minimize(loss, name="train_op")

        init = tf.global_variables_initializer()

    return graph, {
        "features": features,
        "targets": targets,
        "train_op": train_op,
        "loss": loss,
        "init": init,
    }
