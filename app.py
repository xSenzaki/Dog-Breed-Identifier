import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from model import create_cnn_model

class DogBreedIdentifierApp:
    def __init__(self, root, model, class_labels):
        self.root = root
        self.root.title("Dog Breed Identifier")
        self.model = model
        self.class_labels = class_labels
        self.image_path = None

        self.root.geometry("600x600+0+0")
        self.root.configure(background='#f5f5f5')

        self.header_label = Label(self.root, text="Dog Breed Identifier", font=("Inter", 24), bg='#4285f4', fg='white', pady=10)
        self.header_label.pack(fill=X)

        self.image_frame = Frame(self.root, padx=10, pady=10, bg='#f5f5f5')
        self.image_frame.pack()

        self.upload_button = Button(self.root, text="Upload Image", command=self.upload_image, font=("Inter", 12), bg='#1877f2', fg='white')
        self.upload_button.pack(side=TOP, pady=(10, 5))

        self.predict_button = Button(self.root, text="Predict", command=self.predict, font=("Inter", 12), bg='#ed4242', fg='white')
        self.predict_button.pack(side=TOP, pady=(5, 10))

        self.current_image_label = Label(self.image_frame, bg='#f5f5f5')
        self.current_image_label.pack()

        self.result_label = Label(self.image_frame, font=("Inter", 16), bg='#f5f5f5')
        self.result_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
            try:
                image = Image.open(file_path)
                image = image.resize((224, 224))
                image = image.convert('RGB')

                tk_image = ImageTk.PhotoImage(image)
                self.current_image_label.configure(image=tk_image)
                self.current_image_label.image = tk_image

                self.result_label.configure(text="")  # Clear previous result
                self.image_path = file_path  # Save the file path for prediction

                print("Image Uploaded Successfully")

            except FileNotFoundError:
                self.result_label.configure(text="Error: Model file not found", fg="red")

            except Exception as e:
                self.result_label.configure(text=f"Error: {str(e)}", fg="red")

    def predict(self):
        try:
            if self.image_path:
                image = Image.open(self.image_path)
                image = image.resize((224, 224))
                image = image.convert('RGB')

                tk_image = ImageTk.PhotoImage(image)
                self.current_image_label.configure(image=tk_image)
                self.current_image_label.image = tk_image

                img_array = img_to_array(image)
                img_array = preprocess_input(img_array)
                img_array = tf.expand_dims(img_array, 0)
                predictions = self.model.predict(img_array)
                predicted_class_index = tf.argmax(predictions, axis=1)[0].numpy()
                accuracy = predictions[0][predicted_class_index]

                predicted_class_label = self.class_labels[predicted_class_index]

                if accuracy < 0.5:  # Adjust the threshold as needed
                    self.result_label.configure(text="Uploaded image does not appear to be a dog.", fg="red")
                    print("Uploaded image does not appear to be a dog.")
                else:
                    self.result_label.configure(
                        text=f"Predicted Dog Breed is: {predicted_class_label}\nAccuracy: {accuracy:.2%}")
                    print(f"Predicted Dog Breed is {predicted_class_label} with a confidence of {accuracy:.2f}%")

        except FileNotFoundError:
            self.result_label.configure(text="Error: Model file not found", fg="red")

        except Exception as e:
            self.result_label.configure(text=f"Error: {str(e)}", fg="red")


if __name__ == "__main__":
    class_labels = ["Bulldog", "Chihuahua", "German Shepherd", "Golden Retriever", "Husky"]

    pretrained_model_mobilenetv2 = create_cnn_model((224, 224, 3), num_classes=5)
    app = DogBreedIdentifierApp(Tk(), pretrained_model_mobilenetv2, class_labels)

    fine_tuned_model_path = 'fine_tuned_model_mobilenetv2.keras'
    if os.path.exists(fine_tuned_model_path):
        app.model = load_model(fine_tuned_model_path)
        print("Fine-tuned model loaded successfully.")
    else:
        print("Fine-tuned model not found. Please run the fine-tuning process first.")

    app.root.mainloop()
