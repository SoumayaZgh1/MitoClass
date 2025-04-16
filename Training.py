# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:26:43 2025

@author: souma
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Initialize the number of epochs to train for, initial learning rate, batch size, and image dimensions
EPOCHS = 200 # Reduced number of epochs
INIT_LR = 1e-5  # Lower learning rate
BS = 32
IMAGE_DIMS = (255, 255, 3)

# Define paths to save the model and label binarizer
MODEL_PATH = "C:/Codes/Classifiction/test/clean/model.h5"
LABELBIN_PATH = "C:/Codes/Classifiction/test/clean/labelbin.pkl"
PLOT_PATH = "C:/Codes/Classifiction/test/clean/plot_fnal.png"

# Load and preprocess the images
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("C:/Codes/Classifiction/test/clean/Dataset/Elong99")))

random.seed(42)
random.shuffle(imagePaths)

# Data augmentation settings
aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.05,
                          height_shift_range=0.05, shear_range=0.1,
                          zoom_range=0.1, horizontal_flip=True,
                          fill_mode="nearest")

data = []
labels = []

# Loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # Extract the class label
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# Convert labels to binary format
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)

# Split the dataset
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                                  test_size=0.2, random_state=42)

# Load a simpler base model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, 
                          input_shape=(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]))

# Add custom layers on top of the MobileNetV2 base model
head_model = base_model.output
head_model = GlobalAveragePooling2D()(head_model)  # Replace Flatten with GlobalAveragePooling2D
head_model = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(head_model)  # Further reduce number of units
head_model = Dropout(0.4)(head_model)  # Increase dropout rate
head_model = Dense(len(mlb.classes_), activation="softmax")(head_model)

# Combine base model and custom layers into a new model
model = Model(inputs=base_model.input, outputs=head_model)

# Optionally unfreeze some of the layers in the base model for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile the model with a lower learning rate (without decay)
opt = Adam(learning_rate=INIT_LR)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Convert y_true from int to float32
        y_true = tf.cast(y_true, dtype=tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed


model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Implement learning rate scheduler and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
# Train the model
H = model.fit(
    x=aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, 
    verbose=1
    # callbacks=[early_stopping, reduce_lr]  # Uncomment if needed
)

# Save the model to a .h5 file
print("[INFO] Saving model...")
model.save(MODEL_PATH)

# Save the label binarizer to disk
print("[INFO] Saving label binarizer...")
with open(LABELBIN_PATH, "wb") as f:
    pickle.dump(mlb, f)

# Import necessary libraries
import matplotlib
matplotlib.use("TkAgg")  # Ensure GUI support (only if needed)
import matplotlib.pyplot as plt
import numpy as np

# Reset settings to avoid conflicts
plt.rcParams.update(plt.rcParamsDefault)

# Use a valid style (adjust for deprecated styles)
plt.style.use("seaborn-v0_8-whitegrid")

# Create figure
plt.figure(figsize=(12, 8))

N = len(H.history["loss"])

# Plot training and validation loss/accuracy
plt.plot(np.arange(0, N), H.history["loss"], label="Training Loss", color='blue', linestyle='-', linewidth=2)
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation Loss", color='red', linestyle='dashed', linewidth=2)
plt.plot(np.arange(0, N), H.history["accuracy"], label="Training Accuracy", color='green', linestyle='-', linewidth=2)
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Validation Accuracy", color='orange', linestyle='dashed', linewidth=2)

# Labels and title
plt.title("Training and Validation Loss/Accuracy", fontsize=20)
plt.xlabel("Epoch #", fontsize=20)
plt.ylabel("Loss/Accuracy", fontsize=20)
plt.legend(loc="upper right", fontsize=18)
plt.grid(False)  # Disable the grid

# ✅ Make Axes Lines Thicker and Black
ax = plt.gca()  # Get current axis
ax.spines['bottom'].set_linewidth(2)  # Thicken x-axis
ax.spines['left'].set_linewidth(2)  # Thicken y-axis
ax.spines['top'].set_linewidth(0)  # Hide top spine
ax.spines['right'].set_linewidth(0)  # Hide right spine
ax.spines['bottom'].set_color('black')  # X-axis color
ax.spines['left'].set_color('black')  # Y-axis color
ax.tick_params(axis='both', labelsize=16)  
# ✅ Save the plot first
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")

# ✅ Show plot only if in GUI environment
try:
    plt.show()
except:
    print("[INFO] Plot saved but not displayed (non-GUI environment).")

