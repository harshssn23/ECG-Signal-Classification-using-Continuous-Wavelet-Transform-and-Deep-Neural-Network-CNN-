import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set dataset path
dataset_path = 'C:\\Users\\Harsh\\OneDrive\\Desktop\\Sleep Pattern Recog\\cnn\\ecgdataset'

# Data generators
datagen = ImageDataGenerator(validation_split=0.2)  # Assuming 20% of data for validation

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(227, 227),
    batch_size=20,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(227, 227),
    batch_size=20,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained AlexNet model + higher level layers
base_model = AlexNet(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Adjust to the number of classes

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers which you don't want to train
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=8
)

# Save the trained model
model.save('trained_alexnet.h5')

# Evaluate the model
Y_pred = model.predict(validation_generator, validation_generator.samples // validation_generator.batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

# Calculate accuracy
accuracy = np.sum(y_pred == y_true) / len(y_true)
print('Validation accuracy:', accuracy)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=validation_generator.class_indices.keys())
cmd.plot()
plt.show()
