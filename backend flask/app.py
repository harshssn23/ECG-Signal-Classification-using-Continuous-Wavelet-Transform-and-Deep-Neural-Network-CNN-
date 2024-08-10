# Import necessary libraries
from flask import Flask, render_template, request
import torch
from PIL import Image
from torchvision import transforms
from torchvision import transforms, models
from heart_beat import hb
import cv2
import numpy as np
from scipy.signal import find_peaks

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained AlexNet model
# Load the saved model
model = models.alexnet(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 3)  # Assuming 3 classes
# model.load_state_dict(torch.load(r'C:\Users\Harsh\OneDrive\Desktop\Sleep Pattern Recog\cnn\trained_alexnet5.pth', map_location=torch.device('cpu')))
model.load_state_dict(torch.load(r'project\trained_alexnet5.pth', map_location=torch.device('cpu')))
model.eval()  

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to check image quality
def is_image_blurry(image, threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("value:",laplacian_var)
    return laplacian_var < threshold

# Define route for home page
@app.route('/')
def home():
    return render_template('index3.html')

# Define route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index3.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index3.html', message='No selected file')
    if file:
        # Read and preprocess the uploaded image
        img1 = Image.open(file.stream)
        img_np = np.array(img1)
        # Convert RGB to BGR
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Edges', img_np_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Check if the image is blurry
        if is_image_blurry(img_np_bgr):
            message='Upload a Higher Quality Image'
            # return render_template('index2.html', message='Upload a higher quality image')
            return message
        else:
            img = transform(img1)
            img = img.unsqueeze(0)  # Add batch dimension

            # Perform prediction using the model
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
            a=hb(img_np_bgr)
            # Return the prediction result to the user
            class_names = ['arr', 'chf', 'nsr']  # Replace with your class names
            result = class_names[prediction]

            if result == 'arr':
                message = f'Detected ARR: Arrhythmia <br> Additional monitoring may be required'
            elif result == 'chf':
                message = f'Detected CHF: Congestive Heart Failure <br> Consult a healthcare professional'
            elif result == 'nsr':
                # message = f'Detected NSR: Normal Sinus Rhythm <br> No immediate action required <br> Heart Rate:{a} bpm'
                message = f'Detected NSR: Normal Sinus Rhythm <br> No immediate action required'

            else:
                message = 'Unknown class detected.'

            print('Prediction:', result)
            return message  # Return custom message based on predicted class

if __name__ == '__main__':
    app.run(debug=True)
