import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


IMG_WIDTH = 224
IMG_HEIGHT = 224


def load_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    return img


def load_triplets(triplet, training):
    ids = tf.strings.split(triplet)
    anchor = load_image(tf.io.read_file('food/' + ids[0] + '.jpg'))
    truthy = load_image(tf.io.read_file('food/' + ids[1] + '.jpg'))
    falsy = load_image(tf.io.read_file('food/' + ids[2] + '.jpg'))
    if training:
        return tf.stack([anchor, truthy, falsy], axis=0), 1
    else:
        return tf.stack([anchor, truthy, falsy], axis=0)


def create_model(freeze=True):
    inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
    encoder = tf.keras.applications.MobileNetV3Large(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    encoder.trainable = not freeze
    decoder = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=None),
        tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))
        ])
    anchor, truthy, falsy = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    anchor_features = decoder(encoder(anchor))
    truthy_features = decoder(encoder(truthy))
    falsy_features = decoder(encoder(falsy))
    embeddings = tf.stack([anchor_features, truthy_features, falsy_features], axis=-1)
    triple_siamese = tf.keras.Model(inputs=inputs, outputs=embeddings)
    triple_siamese.summary()
    return triple_siamese


def create_inference_model(model):
    distance_truthy, distance_falsy = compute_distances_from_embeddings(model.output)
    predictions = tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.int8)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)


def compute_distances_from_embeddings(embeddings):
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    return distance_truthy, distance_falsy


def make_training_labels():
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


def make_dataset(dataset_filename, training=True):
    dataset = tf.data.TextLineDataset(dataset_filename)
    dataset = dataset.map(lambda triplet: load_triplets(triplet, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def triplet_loss(_, embeddings):
    distance_truthy, distance_falsy = compute_distances_from_embeddings(embeddings)
    return tf.reduce_mean(tf.math.softplus(distance_truthy - distance_falsy))


def accuracy(_, embeddings):
    distance_truthy, distance_falsy = compute_distances_from_embeddings(embeddings)
    return tf.reduce_mean(tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.float32))


def main():
    epochs = 3
    train_batch_size = 32
    inference_batch_size = 64
    num_train_samples = make_training_labels()
    num_test_samples = 59544

    train_dataset = make_dataset('train_samples.txt')
    val_dataset = make_dataset('val_samples.txt')
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=triplet_loss,
                  metrics=[accuracy])
    train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True).repeat().batch(train_batch_size)
    val_dataset = val_dataset.batch(train_batch_size)
    model.fit(
        train_dataset,
        steps_per_epoch=int(np.ceil(num_train_samples / train_batch_size)),
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=10
        )
    test_dataset = make_dataset('test_triplets.txt', training=False).batch(inference_batch_size).prefetch(2)
    inference_model = create_inference_model(model)
    predictions = inference_model.predict(
        test_dataset,
        steps=int(np.ceil(num_test_samples / inference_batch_size)),
        verbose=1
        )
    np.savetxt('predictions.txt', predictions, fmt='%i')


if __name__ == '__main__':
    main()