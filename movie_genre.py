#import stuff
import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import optimizers
from keras import models,layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


#load the dataset into training and validation
train_data = pd.read_csv('train.csv')
train_images = []
for i in tqdm(range(train_data.shape[0])):
    img = image.load_img('../multi_label_dataset/Images/'+train_data['ImageId'][i], target_size = (112,112,3))
    img = image.img_to_array(img)
    img = img/255
    train_images.append(img)
X = np.array(train_images)
y = np.array(train_data.drop(['ImageId'],axis = 1))
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42, test_size = 0.2)

#define the model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (112,112,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation = 'relu'))
model.add(layers.Dense(46,activation = 'sigmoid'))

#compile the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#define callbacks
callbacks_list = [EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1), ModelCheckpoint(filepath = 'movie_genre_weights.h5',monitor = 'val_loss', verbose = 1, save_best_only = True)]

#train the model
history = model.fit(X_train, y_train,epochs =10, validation_data = (X_test,y_test), batch_size = 256, callbacks = callbacks_list)

#save weights and model
model.save('movie_genre_weights.h5')
model_json = model.to_json()
with open('movie_genre_model.json','w') as json_file:
    json_file.write(model_json)

#plot loss curves during training
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'ro',label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and validation accur'test
plt.legend()
plt.show()

#testing an image
img = image.load_img('test.jpg',target_size=(112,112,3))
img = image.img_to_array(img)
img = img/255

# print top 5 labels predicted by the model
labels = np.array(train_data.columns[2:])
predictions = model.predict(img.reshape(1,112,112,3))
top_5 = np.argsort(predictions[0])[:-6:-1]
for i in range(5):
    print("{}".format(labels[top_5[i]])+" ({:.3})".format(predictions[0][top_5[i]]))
plt.imshow(img)
