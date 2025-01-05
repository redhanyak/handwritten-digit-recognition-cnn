from flask import Flask, request, jsonify,  make_response
from flask import render_template


app = Flask(__name__)



@app.route("/")
def hello_world():
    return render_template('index.html')


import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np




# import model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)  # Output after pooling: 64x14x14
        self.dropout1 = nn.Dropout(0.25)  # Dropout after convolutional layers
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.dropout2 = nn.Dropout(0.5)  # Dropout after first fully connected layer
        self.fc2 = nn.Linear(250, 10)
        # self.fc3 = nn.Linear(250, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 14 * 14) 
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened input to 128 units
        self.fc2 = nn.Linear(128, 10)  # Output for 10 classes (digits 0-9)

    def forward(self, x):
        # Convolution -> ReLU -> Pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Convolution -> ReLU -> Pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten the output
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Instantiate the model
model1 = SimpleCNN()
model2 = CNN_1()
model3 = CNN_1()




state_dict = torch.load('scripts/simple_cnn_1.pth')  # Load the state dictionary
model1.load_state_dict(state_dict)  # Load the weights into the model
model1.eval()  # Set the model to evaluation mode

state_dict = torch.load('scripts/model_2.pth') 
model2.load_state_dict(state_dict) 
model2.eval() 

state_dict = torch.load('scripts/aug.pth') 
model3.load_state_dict(state_dict) 
model3.eval() 


import numpy as np
import PIL
from PIL import Image

def transform(canvas_pixels):
    canvas_pixels = np.array(canvas_pixels)
    image_array = canvas_pixels.reshape(200, 200)

    # Step 2: Resize to 28x28 using PIL
    pil_image = Image.fromarray(image_array)  # Convert to a PIL image
    pil_image = pil_image.resize((28, 28), Image.LANCZOS)  # Resize to 28x28

    input_tensor = torch.tensor(np.array(pil_image), dtype=torch.float32)  # Convert to tensor
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    input_tensor /= 255.0  # Scale pixel values to [0, 1]
    return input_tensor

    

@app.route("/print_pixels", methods=["POST"])
def print_pixels():
    try:
        # Get the pixel data from the request
        pixel_data = request.get_json()
        pixel_data = transform(pixel_data)
      
      
        # Make a prediction
        with torch.no_grad():
           
            # average of 3 models
            predictions = (model1(pixel_data) + model2(pixel_data) + 0.6*model3(pixel_data)) / 3
            predicted_digit = predictions.argmax(1).item()

            arr = predictions
            arr = torch.tensor(predictions)
            predictions = torch.nn.functional.softmax(arr, dim=1)

        return jsonify({"arr": arr.tolist(),"predictions": predictions.tolist(), "digit":predicted_digit}), 200
    
    except Exception as e:

        print("Error processing pixel data:", e)
        return jsonify({"error": "Failed to process pixel data"}), 500
    



if __name__=='__main__':
    app.run(debug=True)
