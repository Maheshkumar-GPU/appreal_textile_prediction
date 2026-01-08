import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D


train_df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/ml project datasets/fashion apprealdataset/fashion-mnist_train.csv")
test_df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/ml project datasets/fashion apprealdataset/fashion-mnist_test.csv")
print(train_df.head())
print(train_df.shape)
print(train_df.isnull().sum())
print(train_df.describe())

# splitting features
x_train = train_df.iloc[:,1:].values
y_train = train_df.iloc[:,0].values

x_test = train_df.iloc[:,1:].values
y_test = train_df.iloc[:,0].values

# reshaping the images
x_train = x_train.reshape(-1,28,28)
x_test = x_test.reshape(-1,28,28)

# normailzation
x_train = x_train /255.0
x_test = x_test /255.0

# allocating class names
class_names = ['t - shirt', 'trowser', 'pullover ', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# vizualization

plt.figure(figsize = (6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()

# adding channel dimension
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# one hot encoding label
y_train_cato = to_categorical(y_train,num_classes=10)
y_test_cato = to_categorical(y_test,num_classes=10)
print("y_train encoded shape : ",y_train_cato.shape)
print("y_test encoded shape : ",y_test_cato.shape)

# creating a validation data
x_train,x_val,y_train_cato, y_val_cato = train_test_split(x_train,y_train_cato,test_size=0.2,random_state=42)

#shaping  checking
print("data shapes ")
print("training data ",x_train.shape,y_train_cato.shape)
print("validation data  ",x_val.shape,y_val_cato.shape)
print("testing data ",x_test.shape,y_test_cato.shape)

#CNN model creation
model = Sequential()

# 1st convolutional layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

#1st pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#2nd convolution layer
model.add(Conv2D(64,(3,3),activation='relu'))

# 2nd pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#flattern layer
model.add(Flatten())

# fully connected layer
model.add(Dense(128,activation='relu'))

# dropout layer
model.add(Dropout(0.5))

#output layer
model.add(Dense(10,activation='softmax'))

model.summary()

#complie setup
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

