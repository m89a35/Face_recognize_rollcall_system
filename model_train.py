import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.optimizers import Adam

# Ratio of Train and Test data
# 70% Data for training 30% Data for test
test_data_ratio = 0.3

# nn model dictionary
hdf5_filename = './nn_model/face_recog_special_weights.hdf5'
json_filename = './nn_model/face_recog_special_arch.json'
labeldict_filename = './nn_model/label_dict_special.pkl'

# face sample dictionary
authorized_person_image_dir = 'authorized_person/'
unknown_person_image_dir = 'unknown_person/'

# 神經網路的超參數設定
batch_size = 50
epochs = 1
loss = 'categorical_crossentropy'
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
optimizer = adam


def make_model(nb_class):
    print('Start building NN model..')
    model = Sequential()
    # First layer
    model.add(Dense(4096, input_dim=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Second Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Thrid Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    print('Finished building NN model..')
    return model

#模型訓練
def train_model(model, train_data, train_label, test_data, test_label, nb_epoch):
    #檢查點(會存下訓練出來最好的模型)
    checkpointer = ModelCheckpoint(filepath=hdf5_filename,
                                   verbose=1,
                                   save_best_only=True)

    dnn_json_model = model.to_json()
    #除存模型架構
    with open(json_filename, "w") as json_file:
        json_file.write(dnn_json_model)

    print("Saved NN architecture to disk..")

    print('Start training NN model..')
    #fit模型
    model.fit(train_data,
              train_label,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(test_data, test_label),
              callbacks=[checkpointer],
              shuffle=True,
              verbose=1)

    print('Training finished..')

    return model


x_train = []
x_test = []
y_train = []
y_test = []
#抓樣本的檔名(已知的點名同學資料)
authorized_person_list = os.listdir(authorized_person_image_dir)
# 加unknow person的lable
nb_class = len(authorized_person_list)+1

print('Building neural network architecture...')
dnn_model = make_model(nb_class)

print('Processing AUTHORIZED person data...')

label_dict = dict()
counter = 0
for person in authorized_person_list:
    print('Processing %s data.....' % (person))
    #填入檔名
    label_dict[counter] = person
    #讀入樣本(pkl)
    temp_data = joblib.load(authorized_person_image_dir +
                            person+'/face_descriptor.pkl')
    #填入label
    temp_label = np.repeat(counter, len(temp_data))
    #分割訓練資料與測試資料(訓練七成測試三成)
    temp_x_train, temp_x_validation, temp_y_train, temp_y_validation = train_test_split(
        temp_data, temp_label, test_size=test_data_ratio, random_state=42)
    print('Obtained %i train and %i test data' %
          (len(temp_x_train), len(temp_x_validation)))
    #加入訓練資料
    if len(x_train) == 0:
        x_train = temp_x_train
        x_test = temp_x_validation
        y_train = np.append(y_train, temp_y_train)
        y_test = np.append(y_test, temp_y_validation)
    else:
        x_train = np.append(x_train, temp_x_train, axis=0)
        x_test = np.append(x_test, temp_x_validation, axis=0)
        y_train = np.append(y_train, temp_y_train)
        y_test = np.append(y_test, temp_y_validation)

    counter += 1

print('Finished AUTHORIZED person data exctraction...')
#抓未知臉部的樣本
print('Processing UNKNOWN person data')
label_dict[counter] = 'UNKNOWN'
joblib.dump(label_dict, labeldict_filename)

neg_data = joblib.load(unknown_person_image_dir +
                       'preprocessed_data/unknown_person_face_descriptor.pkl')
temp_data = neg_data
#填入未知臉部樣本的label
temp_label = np.repeat(counter, len(temp_data))
#分割訓練資料與測試資料
temp_x_train, temp_x_validation, temp_y_train, temp_y_validation = train_test_split(
    temp_data, temp_label, test_size=test_data_ratio, random_state=42)
#把已知與未知的資料加在一起
x_train = np.append(x_train, temp_x_train, axis=0)
x_test = np.append(x_test, temp_x_validation, axis=0)
y_train = np.append(y_train, temp_y_train)
y_test = np.append(y_test, temp_y_validation)

print('Finished extracting data.....')
#訓練資料的格式轉換
y_train_cat = y_train.astype('int')
y_train_cat = np_utils.to_categorical(y_train_cat, nb_class)
y_test_cat = y_test.astype('int')
y_test_cat = np_utils.to_categorical(y_test_cat, nb_class)

#訓練模型
trained_dnn_model = train_model(
    dnn_model, x_train, y_train_cat, x_test, y_test_cat, epochs)
