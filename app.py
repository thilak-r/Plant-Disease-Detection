from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision import models

# Initialize the Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Define the model architecture (ResNet18 in this case)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 15)  # 15 classes

# Load the state dictionary
model.load_state_dict(torch.load('crop_disease.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image is uploaded
        file = request.files["file"]
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Preprocess the image for prediction
            image = Image.open(file_path)
            image = transform(image).unsqueeze(0)

            # Predict the class
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]

            return render_template("index.html", prediction=predicted_class)

    return render_template("index.html")

import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
