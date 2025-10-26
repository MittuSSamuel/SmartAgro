
# train_mobilenet.py
# This script shows how to fine-tune MobileNetV2 on PlantVillage-like dataset.
# It prepares data, does transfer learning, and saves a SavedModel in ./model/
# NOTE: Run this on your local machine or a notebook with GPU. This is a template.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Paths - modify to point to your dataset folders
train_dir = './data/train'   # structured as train/class_name/*.jpg
val_dir = './data/val'

img_size = (224,224)
batch_size = 32
epochs = 8

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True, zoom_range=0.2)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_flow = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

base = tf.keras.applications.MobileNetV2(input_shape=img_size+(3,), include_top=False, weights='imagenet', pooling='avg')
base.trainable = False

x = base.output
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
preds = layers.Dense(train_flow.num_classes, activation='softmax')(x)

model = models.Model(inputs=base.input, outputs=preds)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train head
model.fit(train_flow, validation_data=val_flow, epochs=epochs)

# Unfreeze some layers and fine-tune
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=5)

# Save SavedModel
os.makedirs('./model', exist_ok=True)
model.save('./model')
print('Saved model to ./model')