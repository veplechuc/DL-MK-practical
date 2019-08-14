# image classification using CIFAR-10 dataset
# Data Source: https://www.cs.toronto.edu/~kriz/cifar.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam, rmsprop
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def show_img():
    W_grid = 4
    L_grid = 4
    fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
    axes = axes.ravel()
    n_training = len(X_train)
    
    for i in np.arange(0, L_grid * W_grid):
        index = np.random.randint(0, n_training) # pick a random number\n",
        axes[i].imshow(X_train[index])
        axes[i].set_title(y_train[index])
        axes[i].axis('off')
        
    plt.subplots_adjust(hspace = 0.4)
    plt.show()

def main():
    augmentation()    

def augmentation():
    from tensorflow.python.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

    datagen_train = ImageDataGenerator(rotation_range=90)
    datagen_train.fit(X_train)

def saving_model(model, augmented=False):
     # saving the model
    dir = os.path.join(os.getcwd(), 'saved_models')

    if not os.path.isdir(dir):
         os.mkdir(dir)
    
    name = 'keras_cf10_trained_augmented.h5' if augmented else 'keras_cf10_trained.h5'
    model_path = os.path.join(dir, name)
    model.save(model_path)

def model_definition():
    model =  Sequential() #the to create the model
    # now start adding layers
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape)) #1rst conv layer
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')) #2nd layer
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')) #3rst conv layer
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')) #4nd layer
    model.add(MaxPooling2D(2,2)) #sample filter going to be 2 by 2
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(units = 512, activation = 'relu')) #fully connected net
    #here could add a dropout 
    model.add(Dense(units = 512, activation = 'relu')) #fully connected net 

    model.add(Dense(units=10, activation = 'softmax')) #units in the output (10 classes) activation

    # relu = generating an output that is continus for regression
    # softmax = for classification (zeros or ones)

    model.compile(loss='categorical_crossentropy', optimizer = rmsprop(lr=0.001), metrics = ['accuracy'])

    return model


if __name__ == "__main__":

    # main()
    # exit()
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # show_img()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    num_categ = 10
    norm_value = 255

    y_train = to_categorical(y_train, num_categ)
    y_test = to_categorical(y_test, num_categ)
    # convert to values between 0 and 1
    X_train = X_train/norm_value
    X_test = X_test/norm_value

    input_shape = X_train.shape[1:]

    # print(input_shape)
    model = model_definition()
    # rmsprop = root mean square error

    hist = model.fit(X_train, y_train, batch_size = 32, epochs = 1, shuffle = True)
    #shuffle = changes the order 
    # batch_size = how many images at once
    
    # Evaluate the model
    eval = model.evaluate(X_test, y_test)

    print('Test accuracy: {}'.format(eval[1]))

    predict = model.predict_classes(X_test)

    # compare the prediction with the y_test
    y_test = y_test.argmax(1)

    # confision matrix
    
    cm = confusion_matrix (y_test, predict)
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot = True)
    plt.show()

    # saving_model(cnn_model)

    # augmentation, changing somthing from the dataset, lare, brightness, colors, etc
    datagen = ImageDataGenerator(rotation_range = 90,
                                width_shift_range = 0.1,
                                horizontal_flip =True,
                                vertical_flip =True)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train,
                            batch_size = 32), epochs = 3)

    score = model.evaluate(X_test, y_test)

    print('Test accuracy: {}'.format(score[1]))

    saving_model(model, True)