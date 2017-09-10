import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense,  Lambda, Activation,  Convolution2D, Cropping2D
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.utils import shuffle
import os

lines = []
with open('../../bag/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines,test_size =0.2)


def generator(lines,batch_size = 32):
    num_samples = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0,num_samples,batch_size):
            batch_samples =lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../../bag/IMG' +batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train =np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)

train_generator = generator(train_samples,batch_size=32)

validation_generator = generator(validation_samples,batch_size=32)

ch,row,col = 3,160,320
model = Sequential()
#model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3))) #normalize
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=( row, col, ch),
        output_shape=(row, col, ch)))
print("Number of training examples =", model.summary())
#model.add(Lambda(lambda x: x/127.5 -1.,input_shape=(row,col,ch),output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
print("Number of training examples =", model.summary())
model.compile(loss='mse', optimizer='adam')
#model.fit(train_generator, validation_generator, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
# images = []
# measurements = []
# for line in lines:
#     for i in range(3):
#         source_path = line[i]
#         correction = 0.2
#        # filename = source_path.split('/')[-1]
#       #  current_path = '../../bag/IMG' + filename 
#         image = cv2.imread(source_path)
#         images.append(image)
#         if i ==0:
#             measurement = float(line[3])
#         elif i ==1:
#             measurement = float(line[3]) + correction
#         else:
#             measurement = float(line[3]) - correction
#         measurements.append(measurement)

# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images,measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image,1))
#     augmented_measurements.append(measurement*-1.0)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3))) #normalize
# model.add(Cropping2D(cropping=((75,25), (0,0))))
# model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

# model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

# model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(64,3,3,activation="relu"))
# #model.add(Convolution2D(64,3,3,activation="relu"))

# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')
