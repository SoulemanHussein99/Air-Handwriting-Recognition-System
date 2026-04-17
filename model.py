import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
# import EMNIST
ds_train, ds_test = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True
)

# preprocessing for images
def preprocess(image, label):
    
    image = tf.cast(image, tf.float32) / 255.0

    # rotate issue
    image = tf.transpose(image, perm=[1, 0, 2])
    # flip issue
    image = tf.image.flip_left_right(image)
    
   

    return image, label

ds_train = ds_train.map(preprocess).shuffle(10000).batch(64).prefetch(AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(64).prefetch(AUTOTUNE)
 #  Augmentation 
data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
# model
model = Sequential([
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),   # reduce overfitting
    Dense(62, activation='softmax')  
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)

model.fit(ds_train,
        validation_data=ds_test,
        epochs=50,
        callbacks=[callback])

#loss, accuracy = model.evaluate(ds_test)

model.save('model.h5')