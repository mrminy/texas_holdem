import tensorflow as tf

from mikkel.mikkel_ai2 import Poker_data

# Hyperparameters
nb_inputs = 4
nb_classes = 5
H = 200
H2 = 200
batch_size = 246
learning_rate = 0.0001
epochs = 50
# validation_split = 0.15

"""
TODO:
- Add softmax
- Add dropout?
"""


def train_tf_model(save=False):
    print("Loading data...")
    data_train = Poker_data()
    data_train.load_data('data/train/')
    data_test = Poker_data()
    data_test.load_data('data/test/')

    # Build model
    print("Building model...")
    layer_sizes = [nb_inputs, H, H2, nb_classes]

    x = tf.placeholder(tf.float32, [None, layer_sizes[0]])
    y_ = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]])

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[0], layer_sizes[1]])),
                      'biases': tf.Variable(tf.random_normal([layer_sizes[1]]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[1], layer_sizes[2]])),
                      'biases': tf.Variable(tf.random_normal([layer_sizes[2]]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[2], layer_sizes[3]])),
                    'biases': tf.Variable(tf.random_normal([layer_sizes[3]]))}

    l1 = tf.add(tf.matmul(tf.cast(x, tf.float32), hidden_1_layer['weights']),
                hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver()

    # Train and test model
    print("Training and testing model...")
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_cost = 0
            for _ in range(int(len(data_train.data_states) / batch_size)):
                epoch_x, epoch_y = data_train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y_: epoch_y})
                epoch_cost += c
            print('Epoch', epoch+1, 'completed of', epochs, 'loss', epoch_cost)

        if save:
            # Save model
            save_path = saver.save(sess, 'first_tf_model_for_dqn.ckpt')
            print('Model saved in file: %s' % save_path)

        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = sess.run(accuracy, feed_dict={x: data_test.data_states, y_: data_test.data_actions})
        print('Accuracy:', acc)


if __name__ == '__main__':
    train_tf_model(save=False)
