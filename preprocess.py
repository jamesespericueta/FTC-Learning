from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image


img_width, img_height = 1920, 1080

training__data_dir = 'data/train'
validation_data_dir = 'data/validation'
training_samples = 1500
validation_samples = 100
epochs = 50
batch_size = 30

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_width)

else:
    input_shape = (img_width, img_height, 3)

train_data_gen = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=.2,
    vertical_flip=True
)

test_data_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_data_gen.flow_from_directory(
    training__data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)

model.save_weights('first_attempt.h5')

img_predict = image.load_img('data/validation/left/1203.jpg', target_size= (1920, 1080))
img_predict = image.img_to_array(img_predict)
img_predict = np.expand_dims(img_predict, axis = 0)
