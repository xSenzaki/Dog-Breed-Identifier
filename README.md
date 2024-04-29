# Dog Breed Identifier
This Dog Breed Identifier is a project that leverages the power of deep learning, specifically Convolutional Neural Networks (CNNs), to accurately identify the breed of a dog from an uploaded image. The system is trained on a diverse dataset containing images of five popular dog breeds: Bulldog, Chihuahua, German Shepherd, Golden Retriever, and Husky
    
# REQUIREMENTS
    The Dog Breed Identifier requires the following Python libraries:
      TensorFlow
      Keras
      Numpy
      Pillow
      Scipy
      Tkinter
      MobileNetV2 (Pre-trained Model)

# OVERVIEW
    The project consists of two main components:
        Model Training ('train_model.py)'
            Utilizes TensorFlow and Keras to create a CNN model based on the MobileNetV2 architecture.
            Fine-tunes the model using a dataset containing images of the five dog breeds for training and validation.
        User Interface ('app.py')
            Uses Tkinter for the graphical user interface (GUI).
            Allows users to upload an image of a dog.
            Predicts the breed of the dog using the trained CNN model.
            Displays the predicted breed and its confidence level.

# DATASET
    The dataset used for training and testing the model includes:
      Training Images: 800 images per breed (total 4,000 images).
      Testing Images: 20 images per breed (total 100 images).
      Validation Images: 20 images per breed (total 100 images).

# MODEL PERFORMANCE
    After training and evaluation, the model achieved the following accuracies:
      Training Accuracy: 85.62%
      Testing Accuracy: 88.75%
      Validation Accuracy: 89%

# OUTPUT
After completing the training, testing, and validation process for the model, the fine-tuned model file named 'fine_tuned_model_mobilenetv2.keras' will be saved in the main root directory of the folder. Run the 'app.py' to run the program.

![output_1](https://github.com/xSenzaki/Dog-Breed-Identifier/assets/105161390/b4188df5-1c34-4ada-81bf-5463c5695a2f)
![output_2](https://github.com/xSenzaki/Dog-Breed-Identifier/assets/105161390/c9fbf63a-60ca-4c1e-9ea5-fd7756b630d3)
