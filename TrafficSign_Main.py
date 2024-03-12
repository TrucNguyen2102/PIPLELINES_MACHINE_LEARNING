import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
cur_path = os.getcwd()
data_path = os.path.join(cur_path, 'train')

# Kiểm tra sự tồn tại của thư mục chứa dữ liệu
if not os.path.exists(data_path):
    print(f"Error: The directory {data_path} does not exist.")
    exit()

# Kiểm tra và thu thập dữ liệu
for i in range(classes):
    class_path = os.path.join(data_path, str(i))

    # Kiểm tra sự tồn tại của thư mục con (thư mục của mỗi nhãn)
    if not os.path.exists(class_path):
        print(f"Error: The directory {class_path} does not exist.")
        continue

    images = os.listdir(class_path)

    for image_name in images:
        image_path = os.path.join(class_path, image_name)

        # Kiểm tra sự tồn tại của tệp ảnh
        if not os.path.exists(image_path):
            print(f"Error: The image file {image_path} does not exist.")
            continue

        # Kiểm tra định dạng ảnh
        valid_image_formats = ['.jpg', '.jpeg', '.png']
        if not any(image_path.lower().endswith(ext) for ext in valid_image_formats):
            print(f"Error: Invalid image format for {image_path}.")
            continue

        try:
            image = Image.open(image_path)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except FileNotFoundError:
            print(f"Error: The image file {image_path} does not exist.")
        except PIL.UnidentifiedImageError:
            print(f"Error: Unable to identify image file {image_path}.")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Converting the labels into one-hot encoding
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
eps = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=eps, validation_data=(X_val, y_val))

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Testing accuracy on test dataset
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')
test_labels = y_test["ClassId"].values
test_imgs = y_test["Path"].values
test_data = []

for img_path in test_imgs:
    img = Image.open(img_path)
    img = img.resize((30, 30))
    test_data.append(np.array(img))

X_test = np.array(test_data)
pred = model.predict_classes(X_test)

# Accuracy with the test data
print(accuracy_score(test_labels, pred))

# Save the model
model.save('traffic_classifier.h5')
