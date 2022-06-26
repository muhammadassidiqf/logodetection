import pandas as pd
import os, sys, re
import numpy as np
from PIL import Image
import cv2

path = "static/results/bri_1652689676/"
path1 = "static/results/bri_1652689676/50x50/bri/"
PATH = "static/results/bri_1652689676/50x50/"


# items = os.listdir(path)

# for img in items:
#     if "png" in img or "gif" in img or "jpg" in img or "jpeg" in img:

#         if "png" in img and ".png" not in img:
#             img = img + ".png"

#         if "jpg" in img and ".jpg" not in img:
#             img = img + ".jpg"

#         if "asp" not in img and "php" not in img:
#             try:
#                 if not os.path.exists(path1 + "/" + img):
#                     img1 = Image.open(path + "/" + img)
#                     img1 = img1.resize((50, 50), Image.ANTIALIAS)

#                     if not os.path.exists(path1):
#                         os.makedirs(path1)

#                     img1.save(str(path1 + "/" + img))
#             except OSError:
#                 pass


def load_images_from_folder(PATH):
    print(os.listdir(PATH))
    count = 0
    # images = []
    Y_Data = []
    for target_name in os.listdir(PATH):
        for filename in os.listdir(PATH + target_name):
            # Y_Data.append(target_name)
            # print(Y_Data)
            im_path = PATH + target_name + "/" + filename
            img = cv2.imread(os.path.join(PATH, target_name, filename))
            if img is not None:
                if count == 0:
                    images = img
                    Y_Data.append(target_name)
                    # print(img.shape)
                else:
                    images = np.vstack((images, img))
                    Y_Data.append(target_name)
                    # print(img.shape)
                count = count + 1
    return images, Y_Data


train, target = load_images_from_folder(PATH=PATH)
print(train.shape)
print(len(target))

train = train.reshape(len(target), 50, 50, 3)

train.shape
np.save("static/models/train.npy", train)

target_original = np.asarray(target)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(target)
target_labels = le.transform(target)
# print(target_original)
# print(target_labels)


target = pd.DataFrame(list(target_original), list(target_labels))
target = target.drop_duplicates()
target = target.reset_index()
target.columns = ["Encoded_Label", "Original_Label"]

print(target.head())

np.save("static/models/target", target_labels)
target.to_csv("static/models/targets.csv", index=False)
train = np.load("static/models/train.npy")
target_original = np.load("static/models/target.npy")

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

target = utils.to_categorical(target_original).reshape(target_original.shape[0], -1)
print(target_original.shape)
print(target.shape)

model = Sequential()
model.add(Conv2D(32, (2, 2), activation="relu", input_shape=(50, 50, 3)))

model.add(Conv2D(20, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(500, activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(train, target, batch_size=32, epochs=1500)
model.save("static/models/model_5k550_with_insofe.h5")

