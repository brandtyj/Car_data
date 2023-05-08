import base64
import io
import json
from PIL import Image
from flask import Flask, jsonify, request
import torch
from torch import nn
import torchvision.transforms as tt
import torchvision.models as models
import numpy as np
from io import BytesIO

app = Flask(__name__)

class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Load pretrained resnet50
        self.network = models.resnet50(pretrained=True)
        
        #Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 196)
    
    def forward(self, xb):
        return self.network(xb)

# Load the trained model
loading = torch.load('/Users/app/model/car__.pth')
model = Resnet50()
model.load_state_dict(loading)
model.eval()

# Define API endpoints
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

@app.route('/infer', methods=['POST'])
def infer():

    #with open("test_image.jpg", "rb") as image_file:
    #    image = Image.open(BytesIO(base64.b64decode(image_file)))
    #    image = Image.open(image_file)
        
    # Get the image from the POST request
    image_b64 = request.files['image'].read()
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    image = Image.open(image_b64)
    
    # Preprocess the image
    preprocess = tt.Compose([
        tt.Resize(224),
        tt.CenterCrop(224),
        tt.ToTensor(),
        tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = preprocess(image).unsqueeze(0)
    
    # Make a prediction
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
        class_names = ['Make', 'Model', 'Year']
        pred_class = class_names[pred]
    
    # Return the prediction in JSON format
    return jsonify({'class': pred_class})

@app.route('/', methods=['GET'])
def index():
    return 'hi'


if __name__ == '__main__':
    app.run(port=3000, debug=True)
