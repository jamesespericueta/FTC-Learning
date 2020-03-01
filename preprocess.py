import numpy as np
import glob
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

train_labels = []
train_samples = []

right_files = glob.glob('train/right/*.jpg')
left_files = glob.glob('train/left/*.jpg')
print('starting loop')
for file in right_files:
    image = cv2.imread(file)
    train_samples.append(image)
    train_labels.append(0)
print(train_samples[0])
for file in left_files:
    image = cv2.imread(file)
    train_samples.append(image)
    train_labels.append(1)
idg = ImageDataGenerator(brightness_range=[0.2, 1.0])
idg.fit(train_samples)
it = idg.flow(np.array(train_samples), np.array([1, 1, ]), batch_size=1)
