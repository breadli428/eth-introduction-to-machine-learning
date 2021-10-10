import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


IMG_WIDTH = 224
IMG_HEIGHT = 224


def train_val_split():
    samples = 'train_triplets.txt'
    with open(samples, 'r') as file:
        triplets = [line for line in file.readlines()]
    train_samples, val_samples = train_test_split(triplets, test_size=0.1)
    with open('val_samples.txt', 'w') as file:
        for item in val_samples:
            file.write(item)
    with open('train_samples.txt', 'w') as file:
        for item in train_samples:
            file.write(item)
    return len(train_samples)


def load_triplets(triplet, training):
    ids = tf.strings.split(triplet)
    image_triplet = []
    for i in range(3):
        image = tf.io.read_file('food/' + ids[i] + '.jpg')
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        image_triplet.append(image)
    if training:
        return tf.stack(image_triplet, axis=0), 1
    else:
        return tf.stack(image_triplet, axis=0)


def load_dataset(dataset_filename, training=True):
    dataset = tf.data.TextLineDataset(dataset_filename)
    dataset = dataset.map(lambda triplet: load_triplets(triplet, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def compute_distances(outputs):
    distance_pos = tf.reduce_sum(tf.square(outputs[..., 0] - outputs[..., 1]), axis=1)
    distance_neg = tf.reduce_sum(tf.square(outputs[..., 0] - outputs[..., 2]), axis=1)
    return distance_pos, distance_neg


def triplet_loss(_, outputs):
    distance_pos, distance_neg = compute_distances(outputs)
    return tf.reduce_mean(tf.math.softplus(distance_pos - distance_neg))


def accuracy(_, outputs):
    distance_pos, distance_neg = compute_distances(outputs)
    return tf.reduce_mean(tf.cast(tf.greater_equal(distance_neg, distance_pos), tf.float32))


epochs = 3
train_batch_size = 32
inference_batch_size = 64
train_sample_size = train_val_split()
test_sample_size = 59544

train_dataset = load_dataset('train_samples.txt')
val_dataset = load_dataset('val_samples.txt')
test_dataset = load_dataset('test_triplets.txt', training=False).batch(inference_batch_size).prefetch(2)

pretrained = tf.keras.applications.MobileNetV3Large(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
pretrained.trainable = False
custom_layers = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=None),
    tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))
    ]) 
inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
output_triplet = []
for i in range(3):
    output_triplet.append(custom_layers(pretrained(inputs[:, i, ...])))
outputs = tf.stack(output_triplet, axis=-1)
triplet_model = tf.keras.Model(inputs=inputs, outputs=outputs)
triplet_model.summary()
triplet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=triplet_loss, metrics=[accuracy])
train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True).repeat().batch(train_batch_size)
val_dataset = val_dataset.batch(train_batch_size)
triplet_model.fit(train_dataset, steps_per_epoch=int(np.ceil(train_sample_size / train_batch_size)),
    epochs=epochs, validation_data=val_dataset, validation_steps=10)

distance_positive, distance_negative = compute_distances(triplet_model.output)
predictions = tf.cast(tf.greater_equal(distance_negative, distance_positive), tf.int8)
inference_model = tf.keras.Model(inputs=triplet_model.inputs, outputs=predictions)

predictions = inference_model.predict(test_dataset, steps=int(np.ceil(test_sample_size / inference_batch_size)), verbose=1)

np.savetxt('sub.txt', predictions, fmt='%i')
