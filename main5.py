import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


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

# model training
history = model.fit(x_train,y_train_cato,epochs=10,batch_size = 64,validation_data=(x_val,y_val_cato))

# accuracy and loss graph
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label = "train accuracy")
plt.plot(history.history["val_accuracy"],label = "val accuracy")
plt.legend()
plt.title("model accuracy")

# loss  graph
plt.subplot(1,2,2)
plt.plot(history.history["loss"],label = "train loss")
plt.plot(history.history["val_loss"],label = "val loss")
plt.legend()
plt.title("model_loss")
plt.show()

# test data evaluation
test_loss, test_acc = model.evaluate(x_test,y_test_cato)
print("test loss : ",test_loss)
print("test accuracy : ",test_acc)

# final prediction sample
index = 5
img = x_test[index].reshape(1,28,28,1)
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
print("predicted = ",class_names[predicted_class])
print("actual = ",class_names[y_test[index]])

# predict all images
y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred,axis=1)

# confusion matrix
cm = confusion_matrix(y_test,y_pred_class)

# ploting
plt.figure(figsize = (12,12))
sns.heatmap(cm,annot=True,fmt="d",xticklabels=class_names,yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# classification report
print(classification_report(y_test,y_pred_class,target_names=class_names))

# model saving

model.save("appreal_textile.h5")
print("model saved successfully ")


