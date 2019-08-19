import os
import glob
import shutil
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense,Convolution2D,MaxPooling2D,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
import shutil
import random
import csv
import argparse


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
batch_size = 50
nb_classes = 2

img_rows, img_cols = 300, 200
channels = 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    path_test_prefix=args.input_dir
    inputs = []
    filenames=[]
    cnt=0
    resdir=args.output_dir
    if not os.path.exists(resdir):
        os.mkdir(resdir)
        os.mkdir(os.path.join(resdir,"TOC"))
        os.mkdir(os.path.join(resdir,"notTOC"))

    for img_path in glob.glob(os.path.join(path_test_prefix,"*")):
        #print(img_path)
        img = image.load_img(img_path, grayscale=True,  target_size=(img_rows, img_cols))
        img = image.img_to_array(img)
        img/=255.0
        inputs.append(img.copy())
        filenames.append(os.path.basename(img_path))
        cnt+=1
        if cnt%100==0:
            print(cnt)
    inputs = np.array(inputs)
    # CNNを構築
    np.random.seed(777)
    base_model = Xception(weights=None,include_top=False,input_shape=(img_rows,img_cols,1))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='sigmoid')(x)
    model = Model(base_model.input, x)
    model.load_weights("sampleweights.hdf5")
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    pred_prob=model.predict(inputs,batch_size)
    pred=np.argmax(pred_prob, axis = 1)
    print(pred)
    wf = open('resdata.csv','w')
    csvWriter = csv.writer(wf)
    for index,fname in enumerate(filenames):
        dirname=""
        if pred[index]==0:
            dirname=os.path.join(resdir,"TOC")
        else:
            dirname=os.path.join(resdir,"notTOC")
        root, ext = os.path.splitext(fname)
        prob=('%.3f' % pred_prob[index][pred[index]])
        root2,ext=os.path.splitext(root)
        pid,num=root2.split("_")
        csvWriter.writerow([fname,pid,num,1-pred[index],prob])
        shutil.copyfile(path_test_prefix+fname,os.path.join(dirname,fname))
    wf.close()

