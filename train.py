import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.exposure import rescale_intensity

# Start
print("Running on TensorFlow " + str(tf.__version__))

# Get list of images and labels for training
images = []
labels = []
for dir in [(0,'red'), (1,'yellow'), (2,'green'), (4,'unknown')]:

    # Read, convert, resize
    filelist = sorted(glob.glob('ros/src/tl_detector/' + dir[1] + '/*.png' ))
    images_in = [cv2.imread(filename) for filename in filelist]
    images_in = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_in]
    images_in = [cv2.resize(image, (320, 320)) for image in images_in]

    # Add to lists
    images += images_in
    labels_in = [dir[0]] * len(images_in)
    labels += labels_in
      
# Save a sample image
cv2.imwrite("sample.png", images[0])

# Numpify
images = np.array(images)

# Log
print("Number of images: " + str(len(images)))
print("Number of labels: " + str(len(labels)))

# Shuffle
images, labels = shuffle(images, labels)

# Split
X_train, X_validation, y_train, y_validation = train_test_split(images, labels, test_size=0.2, random_state=0)

# Count
n_classes = len(set(labels))
n_train = X_train.shape[0]
n_validation = X_validation.shape[0]
print("Number of training images: " + str(n_train))
print("Number of validation images: " + str(n_validation))

# Define the model
def Model(x):
    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 320x240x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Relu activation
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Relu activation
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = tf.contrib.layers.flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Relu Activation
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Relu Activation
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    # That's it
    return logits

# Hype those parameters
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Inputs
x = tf.placeholder(tf.float32, (None, 320, 320, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# TF Operations
logits = Model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Let's save when done
saver = tf.train.Saver()

# Evaluate the model on a batch
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train
def train(X_train, y_train):
    with tf.Session() as sess:
    
        # Init
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        # Train
        print("Training...")
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            # Evaluate
            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {}, validation accuracy = {:.2f}".format(i+1, validation_accuracy))

        # Save
        saver.save(sess, './model')
        print("Model saved")

# Train
train(X_train, y_train)

# Test the test set
with tf.Session() as sess:
    # Load model
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    # Evaluate test data
#    test_accuracy = evaluate(X_test, y_test)
#    print("Test Accuracy = {:.3f}".format(test_accuracy))
    

