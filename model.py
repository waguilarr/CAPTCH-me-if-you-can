import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.python.framework import graph_io
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import sys
print(tf.__version__)

train = pd.read_csv(r'C:\Users\William\Desktop\catch_me_if_you_can/emnist-mnist-train.csv')

classes = train.iloc[:,0].nunique()
x = np.array(train.iloc[:,1:].values)
y = np.array(train.iloc[:,0].values)
x = x/255

train_shape = train.shape[0]
train_height = 28
train_width = 28
train_size = train_height*train_width

x = x.reshape(train_shape, train_height, train_width, 1)
y = to_categorical(y, classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)

stop = EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=0, restore_best_weights=True, patience=3,
                      mode='max')
reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, mil_lr=0.0001)

history = model.fit(
    x_train, y_train, batch_size=64, epochs = 20, validation_data = (x_test, y_test),
    verbose = 1, steps_per_epoch=x_train.shape[0] // 64, callbacks=[reduce,stop])

os.makedirs('C:\Users\William\Desktop\catch_me_if_you_can\model', exist_ok=True)
model.save('C:\Users\William\Desktop\catch_me_if_you_can\model\keras_model.h5')
model = load_model(r'C:\Users\William\Desktop\catch_me_if_you_can\model\keras_model.h5')

tf.saved_model.save(model, r'C:\Users\William\Desktop\catch_me_if_you_can\model')
with tf.compat.v1.gfile.FastGFile(r"C:\Users\William\Desktop\catch_me_if_you_can\model\saved_model.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        proto_b = f.read()
        graph_def = tf.compat.v1.GraphDef()
        text_format.Merge(proto_b, graph_def) 
        _ = tf.graph_util.import_graph_def(graph_def, name='')
