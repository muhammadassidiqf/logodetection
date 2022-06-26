import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, Activation, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import os
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import time
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

def get_model():
    base_model = tf.keras.applications.MobileNet(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")
    model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.2),
                                #  tf.keras.layers.Dense(1, activation="sigmoid")                                     
                                 tf.keras.layers.Dense(2, activation="softmax")
                                ])
    optimizer = keras.optimizers.Adam(learning_rate= 0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

image_size = 224
base_dir = 'static/results/'

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.2,zoom_range=0.2, fill_mode="nearest")
valid_datagen = ImageDataGenerator(rescale=1./255)
batch_size=16
# train_generator = train_datagen.flow_from_directory(os.path.join(base_dir,"bca"), target_size=(image_size, image_size), batch_size=(batch_size), class_mode='binary')
# valid_generator = valid_datagen.flow_from_directory(os.path.join(base_dir,"bca_valid"), target_size=(image_size, image_size), batch_size=(batch_size), class_mode='binary')
train_generator = train_datagen.flow_from_directory(os.path.join(base_dir,"TRAINING"), target_size=(image_size, image_size), batch_size=(batch_size), class_mode='categorical',
            classes = ['bri'])
valid_generator = valid_datagen.flow_from_directory(os.path.join(base_dir,"TESTING"), target_size=(image_size, image_size), batch_size=(batch_size), class_mode='categorical',
            classes = ['bri'])
kategori = train_generator.class_indices
print(kategori)

lis = list(kategori.keys())
print(lis)

jumlah_class = len(kategori)
print(jumlah_class)

print(train_generator.image_shape)
print(valid_generator.image_shape)

now = datetime.datetime.now
t = now()
checkpoint_path = "static/checkpoint3/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
callbacks = [ ModelCheckpoint(filepath=checkpoint_path, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')]

step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
model = get_model()
# model.save_weights(checkpoint_path.format(epoch=0))
history = model.fit_generator(train_generator, validation_data=valid_generator, epochs=10, steps_per_epoch=step_size_train, validation_steps=step_size_valid, verbose=1, callbacks=callbacks)
model.save('static/models/my_model3') 

print('Training time: %s' % (now() - t))



# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(loss))

# fig = plt.figure(figsize=(10,6))
# plt.plot(epochs,loss,c="red",label="Training Loss")
# plt.plot(epochs,val_loss,c="blue",label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# epochs = range(len(acc))

# fig = plt.figure(figsize=(10,6))
# plt.plot(epochs,acc,c="red",label="Training Acc")
# plt.plot(epochs,val_acc,c="blue",label="Validation Acc")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
test_num = valid_generator.samples

label_test = []
for i in range((test_num // valid_generator.batch_size)+1):
    X,y = valid_generator.next()
    label_test.append(y)
        
label_test = np.argmax(np.vstack(label_test), axis=1)
label_test.shape

ineference_model = get_model()
ineference_model.set_weights(model.get_weights())

pred_test = np.argmax(ineference_model.predict(valid_generator), axis=1)

print('ACCURACY:', accuracy_score(label_test, pred_test))

cnf_matrix = confusion_matrix(label_test, pred_test)

plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=['BRI'])
plt.show()