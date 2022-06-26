import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, Activation, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD
import os
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import time
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import itertools
import csv
import pandas as pd


BASE_PATH = "../static/results/TRAINING/"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "dataset"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations.csv"])

BASE_OUTPUT = "../static/models/"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "models.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

print("[INFO] loading dataset...")
# rows = open(ANNOTS_PATH).read().strip().split("\n")

# rows = open('data.csv')

data = []
targets = []
filenames = []

with open(ANNOTS_PATH, 'r', newline='') as csv_file:
    reader = csv.reader(line.replace('  ', ',') for line in csv_file)
    rows = list(reader)
    # rows = csv_file.readlines()
# print(rows[0])
for i in range(len(rows)):
    # for j in rows[i]:
	(filename, startX, startY, endX, endY) = rows[i]
	pathfile = BASE_PATH + filename
	imagePath = os.path.sep.join([pathfile])
	# print(imagePath)
	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]
	# print((h,w))
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	startX = float(startX) / w
	startY = float(startY) / h
	endX = float(endX) / w
	endY = float(endY) / h

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)

	data.append(image)
	# np.append(data, image)
	# np.append(targets,(startX, startY, endX, endY))
	targets.append((startX, startY, endX, endY))
	filenames.append(pathfile)

	# convert the data and targets to NumPy arrays, scaling the input
	# pixel intensities from the range [0, 255] to [0, 1]
print(len(data))
print(len(targets))
data = (np.array(data, dtype="float32") / 255.0)
# print(filenames)
targets = (np.array(targets, dtype="float32"))
# print(targets)
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing

split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)
# # unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
# plot the model training history
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)