import keras
# from tensorflow.keras.models import Sequential,Input,Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
# from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

batch_size = 50
epochs = 20
num_classes = 2

npy_path = "/home/nimai/workspace/fall-detection-wifi/Numpys/exp55_denoised_test.npy"
data = np.load(npy_path, allow_pickle=True)
labels = data[:, -1]
data = np.delete(data, -1, 1)
data = data.astype(np.float)
x_dataset = []
y_label = []
for idx, d in enumerate(data):
    dd = data[idx:idx+batch_size]
    label = labels[idx:idx+batch_size]
    label = label[np.argmax(label)]
    x, y = dd.shape
    if x == batch_size:
        x_dataset.append(dd)
        if label == 'nofall':
            y_label.append(0)
        else:
            y_label.append(1)

print(np.array(x_dataset).shape)
print(np.array(y_label).shape)

X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_label, test_size=0.2,random_state=109) # 70% training and 30% test
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
print('Training data shape: ', X_train.shape, y_train.shape)
print('Testing data shape: ', X_test.shape, y_test.shape)
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

X_train = X_train.reshape(-1, batch_size,53, 1)
X_test = X_test.reshape(-1, batch_size,53, 1)
print(X_train.shape, X_test.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.

train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)

X_train,valid_X,train_label,valid_label = train_test_split(X_train, train_Y_one_hot, test_size=0.2, random_state=13)

fall_model = Sequential()
fall_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(batch_size, 53, 1),padding='same'))
fall_model.add(LeakyReLU(alpha=0.1))
fall_model.add(MaxPooling2D((2, 2),padding='same'))
fall_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fall_model.add(LeakyReLU(alpha=0.1))
fall_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fall_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fall_model.add(LeakyReLU(alpha=0.1))                  
fall_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fall_model.add(Flatten())
fall_model.add(Dense(128, activation='linear'))
fall_model.add(LeakyReLU(alpha=0.1))                  
fall_model.add(Dense(num_classes, activation='softmax'))
fall_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizer_v2.adam.Adam(),metrics=['accuracy'])
fall_model.summary()
fall_train = fall_model.fit(X_train, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
fall_model.save("fall_train_dropout.h5py")

test_eval = fall_model.evaluate(X_test, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
target_names = ["Class {}".format(i) for i in range(num_classes)]
predicted_classes = fall_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(classification_report(y_test, predicted_classes, target_names=target_names))