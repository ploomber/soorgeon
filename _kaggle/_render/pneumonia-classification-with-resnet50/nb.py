# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    Dropout,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [markdown]
# ## first

# %%
# Save the training data directory in a variable
XRay_Directory = "/kaggle/input/pneumonia-clasification/Dataset/Dataset/"

# Check the folders in the directory
os.listdir(XRay_Directory)

# %%
# Create tensor images, normalize them and create a validation set
image_generator = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# %%
# Create the training generator
train_generator = image_generator.flow_from_directory(
    batch_size=40,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode="categorical",
    subset="training",
)

# %%
# Create the validation generator
validation_generator = image_generator.flow_from_directory(
    batch_size=40,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode="categorical",
    subset="validation",
)

# %% [markdown]
# ## generate

# %%
# Generate a batch with 40 images
train_images, train_labels = next(train_generator)
train_images.shape, train_labels.shape

# %%
# Dictionary with the categories
label_names = {
    0: "Covid-19",
    1: "Normal",
    2: "Viral Pneumonia",
    3: "Bacterial Pneumonia",
}

# %%
# Create a 6x6 matrix to show an example of the images
L = 6
W = 6

fig, axes = plt.subplots(L, W, figsize=(15, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(train_images[i])
    axes[i].set_title(label_names[np.argmax(train_labels[i])])
    axes[i].axis("off")

plt.subplots_adjust(wspace=0.3)

# %% [markdown]
# ## transfer-learning

# %%
# Using the ResNet50 imagenet weights for transfer learning
basemodel = ResNet50(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(256, 256, 3))
)
basemodel.summary()

# %%
# Freezing the model but not the last layers

for layer in basemodel.layers[:-10]:
    layers.trainable = False

# %%
# Create the last layers for our model
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(4, 4))(headmodel)
headmodel = Flatten(name="flatten")(headmodel)
headmodel = Dense(256, activation="relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation="relu")(headmodel)
headmodel = Dropout(0.2)(headmodel)
headmodel = Dense(4, activation="softmax")(headmodel)

model = Model(inputs=basemodel.input, outputs=headmodel)

# %%
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=0.0001, decay=0.000001),
    metrics=["accuracy"],
)

# %%
# Using earlystopping to avoid overfitting
earlystopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

# %%
# Saving the best model weights
checkpoint = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

# %%
train_generator = image_generator.flow_from_directory(
    batch_size=4,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode="categorical",
    subset="training",
)
val_generator = image_generator.flow_from_directory(
    batch_size=4,
    directory=XRay_Directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode="categorical",
    subset="validation",
)

# %%
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // 4,
    epochs=25,
    validation_data=val_generator,
    validation_steps=val_generator.n // 4,
    callbacks=[checkpoint, earlystopping],
)

# %% [markdown]
# ## plotting

# %%
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])

plt.title("Model Loss and Accuracy Progress During Training")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy and Loss")
plt.legend(["Training Accuracy", "Training Loss"])
plt.show()

# %%
plt.plot(history.history["val_loss"])
plt.title("Model Loss During Cross-Validation")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend(["Validation Loss"])
plt.show()

# %%
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy Progress During Cross-Validation")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend(["Validation Accuracy"])
plt.show()

# %% [markdown]
# ## testing

# %%
test_directory = "/kaggle/input/pneumonia-clasification/Test/Test/"

# %%
test_gen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_gen.flow_from_directory(
    batch_size=40,
    directory=test_directory,
    shuffle=True,
    target_size=(256, 256),
    class_mode="categorical",
)

evaluate = model.evaluate_generator(
    test_generator, steps=test_generator.n // 4, verbose=1
)

print("Accuracy Test : {}".format(evaluate[1]))

# %%
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

prediction = []
original = []
image = []

for i in range(len(os.listdir(test_directory))):
    for item in os.listdir(os.path.join(test_directory, str(i))):
        img = cv2.imread(os.path.join(test_directory, str(i), item))
        img = cv2.resize(img, (256, 256))
        image.append(img)
        img = img / 255
        img = img.reshape(-1, 256, 256, 3)
        predict = model.predict(img)
        predict = np.argmax(predict)
        prediction.append(predict)
        original.append(i)

# %%
score = accuracy_score(original, prediction)
print("Test Accuracy : {}".format(score))

# %%
L = 5
W = 5

# %% [markdown]
# ## predict

fig, axes = plt.subplots(L, W, figsize=(15, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(image[i])
    axes[i].set_title(
        "Predict= {}\nReal= {}".format(
            str(label_names[prediction[i]]), str(label_names[original[i]])
        )
    )
    axes[i].axis("off")

plt.subplots_adjust(wspace=1.2)

# %%
print(classification_report(np.asarray(original), np.asarray(prediction)))

# %%
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap="Blues")

ax.set_xlabel("Predicted")
ax.set_ylabel("Original")
ax.set_title("Confusion_matrix")
plt.show()

# %%
