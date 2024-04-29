import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Function to create a model with a specified input shape and number of classes
def create_cnn_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:100]:
        layer.trainable = True

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

class DogBreedModel:
    def __init__(self, model, class_labels):
        self.model = model
        self.class_labels = class_labels

    def fine_tune_model(self, train_dir, test_dir, epochs, steps_per_epoch, test_steps=None):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical'
        )

        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        history = self.model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

        # Evaluate the model on the train dataset
        train_accuracy = history.history['accuracy'][-1]
        print(f"Overall Training Accuracy: {train_accuracy * 100:.2f}%")

        # Evaluate the model on the test dataset
        test_results = self.model.evaluate(test_generator, steps=test_steps)
        print(f"Overall Testing Accuracy: {test_results[1] * 100:.2f}%")

    def visualize_training_data(self, data_dir):
        sample_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        # Create a single frame to hold all the images
        all_images_frame = Frame(self.image_frame, padx=10, pady=10)
        all_images_frame.pack()

        for i in range(len(sample_generator.filenames)):
            img_array, label = sample_generator.next()
            label_index = np.argmax(label)
            label = self.class_labels[label_index]

            # Convert palette images to RGBA
            if img_array.shape[3] == 1:
                img_array = np.concatenate([img_array] * 3, axis=3)

            img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
            img = img.resize((150, 150))

            tk_img = ImageTk.PhotoImage(img)

            # Create a frame for each row
            row_frame = Frame(all_images_frame, padx=10, pady=10)
            row_frame.pack()

            label_widget = Label(row_frame, text=f"Class: {label}")
            label_widget.pack(side=LEFT)

            img_widget = Label(row_frame, image=tk_img)
            img_widget.image = tk_img
            img_widget.pack(side=LEFT)

    def validate_model(self, val_dir, save_path='fine_tuned_model_mobilenetv2.keras'):
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical'
        )

        if val_generator.samples == 0:
            print("No images found in the validation dataset.")
            return

        results = self.model.evaluate(val_generator)
        print(f"Overall Validation Accuracy: {results[1] * 100:.2f}%")

        # Save the fine-tuned model for future use
        self.model.save(save_path)
        print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    class_labels = ["Bulldog", "Chihuahua", "German Shepherd", "Golden Retriever", "Husky"]

    # Create a new instance of the DogBreedModel class
    pretrained_model_mobilenetv2 = create_cnn_model((224, 224, 3), num_classes=5)
    dog_breed_model = DogBreedModel(pretrained_model_mobilenetv2, class_labels)

    # Directories for training, validation, and test datasets
    train_data_dir = 'C:\\Users\\RYZEN\\Desktop\\Dog Breed Identifier\\dataset\\train'
    val_data_dir = 'C:\\Users\\RYZEN\\Desktop\\Dog Breed Identifier\\dataset\\val'
    test_data_dir = 'C:\\Users\\RYZEN\\Desktop\\Dog Breed Identifier\\dataset\\test'

    # Fine-tune the model with the training and validation datasets
    dog_breed_model.fine_tune_model(train_data_dir, test_data_dir, epochs=5, steps_per_epoch=10, test_steps=10)

    # Validate the model and print the overall accuracy on validation
    dog_breed_model.validate_model(val_data_dir)
