# Plant Disease Detection Using Deep Learning

This project leverages a Convolutional Neural Network (CNN) model, specifically ResNet-18, to classify diseases affecting crops such as tomato, potato, and bell pepper. The model predicts whether a plant is healthy or affected by a specific disease based on image data.

## Table of Contents

- Overview
- Requirements
- Dataset
- Model Architecture
- Training the Model
- Application Structure
- Running the Application
- Explanation of Code
- Results and Evaluation
- References

## Overview

This project aims to assist in the early detection of plant diseases, which is crucial for agricultural productivity. By using a CNN model trained on labeled images, the application classifies images of tomato, potato, and bell pepper leaves to determine if they are healthy or suffering from diseases like Bacterial Spot, Early Blight, Late Blight, and more.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Flask
- PIL (Pillow)
- scikit-learn
- matplotlib
- tqdm
- pandas

### Install the required packages using:

```

pip install -r requirements.txt

```

## Dataset
The dataset is structured in the following format under the PlantVillage directory:

```
PlantVillage_Balanced/
    pepper_bell_bacterial_spot/
    pepper_bell_healthy/
    potato_early_blight/
    ...
    tomato_spider_mites/
    tomato_healthy/
```
This dataset should be split into training, validation, and testing sets during data loading.

## Model Architecture
The model is based on ResNet-18, a pre-trained CNN model available in PyTorch. The final fully connected layer is modified to output 15 classes, each representing a plant health status or disease. Pre-trained weights on the ImageNet dataset (```IMAGENET1K_V1```) are used, which helps in better feature extraction and faster convergence.

## Training the Model

The training script ```reduced_complexity.ipynb``` performs the following tasks:

- **Data Loading**: Loads and splits the dataset into training, validation, and test sets.
- **Transformations**: Includes resizing, cropping, normalization, and data augmentation to improve model generalization.
- **Training**: The model is trained for a set number of epochs. Metrics such as accuracy and loss are logged.
- **Validation**: Performance is validated after each epoch. Early stopping and learning rate scheduling are applied to improve convergence.
- **Evaluation and Saving**: The final model is saved in .pth format. Additional scripts generate and display the confusion matrix and ROC curves.


## Application Structure
The web application uses Flask as the backend framework. It consists of the following key components:

- **app.py**: Main application file that loads the trained model and handles image uploads and predictions.
- **templates/index.html**: HTML file for the user interface.

## Running the Application

- **Train the Model (optional)**: If you would like to train the model from scratch, run modeltrain.ipynb in Jupyter Notebook or convert it into a script.

- **Run the Application**: Launch the Flask web application with:
  ```

  python app.py

  ```
- **Upload and Predict**: Visit ```http://127.0.0.1:5000``` in your web browser. Upload a plant image to classify its disease status.

## Example Usage

- **Clone the repository**:
  ```
  git clone <repository_url>

  ```

- **Navigate to the project folder**
  ```

  cd plant-disease-detection

  ```

- Install dependencies and run the application.


## Explanation of Code

### app.py
- **Model Loading**: The model is loaded with a state dictionary saved as ```crop_disease_simple_undersample2.pth``` The model structure is modified to output predictions for 15 classes.
- **Transformation Pipeline**: Images are resized, cropped, and normalized to match the pre-trained ResNet-18 input requirements.
- **Prediction**: After uploading an image, it’s processed through the model, and the predicted class is displayed on the web interface.

### reduced_complexity.ipynb

- **Data Transformation and Augmentation**: Random cropping, resizing, and horizontal flipping augment the dataset to enhance generalization.
- **Training and Validation**: Loss and accuracy for each epoch are logged, and a scheduler adjusts the learning rate based on validation loss.
- **Evaluation**: Confusion matrix and ROC curve are generated to assess classification performance. A CSV file logs training metrics for review.


## References
- PyTorch ResNet Documentation
- PlantVillage Dataset
 
