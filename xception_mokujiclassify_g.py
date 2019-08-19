import os
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
import shutil
import random
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
BASE_DIR ='dataset'
TRAIN_DIR = 'traindata'
TEST_DIR = 'valdata'
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
K.set_session(sess)

batch_size = 20
nb_classes = 2

img_rows, img_cols = 300, 200
channels = 3

nb_train_samples = 18300
nb_val_samples = 400
nb_epoch = 20
#class_weight = {0 : 0.8, 1 : 0.2}
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)
    for d in os.listdir(BASE_DIR):
        files = os.listdir(os.path.join(BASE_DIR, d))
        random.shuffle(files)
        if not os.path.exists(os.path.join(TEST_DIR,d)):
            os.mkdir(os.path.join(TEST_DIR,d))
        if not os.path.exists(os.path.join(TRAIN_DIR,d)):
            os.mkdir(os.path.join(TRAIN_DIR,d))
        for f in files[:200]:
            source = os.path.join(BASE_DIR, d, f)
            dest = os.path.join(TEST_DIR, d, f)
            print(source+"->"+dest)
            shutil.copy(source, dest)
        for f in files[200:]:
            source = os.path.join(BASE_DIR, d, f)
            dest = os.path.join(TRAIN_DIR, d, f)
            print(source+"->"+dest)
            shutil.copy(source,dest)

if __name__ == '__main__':
    np.random.seed(777)
    base_model = Xception(weights=None,include_top=False,input_shape=(img_rows,img_cols,1))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='sigmoid')(x)
    model = Model(base_model.input, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1.0 / 255,width_shift_range=0.15,height_shift_range=0.15,rotation_range=5)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_rows, img_cols),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(img_rows, img_cols),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size)

    checkpoint=ModelCheckpoint(filepath = './checkpoints_xcp/cnn_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto',save_weights_only=True)

    # モデル訓練
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples/batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples/batch_size,
        callbacks=[checkpoint])
        #class_weight=class_weight)
    with open('history.pickle', 'wb') as f:
        pickle.dump(history, f)

