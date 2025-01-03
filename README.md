# handwritten-digit-recognition-cnn
A handwritten recognition app using deep learning

This is a simple Flask web application which accepts a handwritten digit and predicts the digit. The prediction is done using a basic Convolutional Neural Network. This cnn model is trained using MNIST dataset using pytorch. 

## Dataset
The MNIST dataset comprises:

Training Set: 60,000 images
Testing Set: 10,000 images 
Each image is 28x28 pixels, representing a single digit from 0 to 9.

## Model Architecture
The recognition system uses a Convolutional Neural Network (CNN) with the following architecture:

Convolutional Layers: Extract spatial features from the images.
Pooling Layers: Reduce dimensionality while preserving important features.
Fully Connected Layers: Map the extracted features to digit classes.
Dropout Layers: Prevent overfitting by randomly deactivating neurons during training.

- Ensemble Learning
There are three cnn models trained using the same dataset.The final prediction is the average of their predictions during inference.

```
# Example: Averaging predictions from three different models
predictions = (model1(input) + model2(input) + model3(input)) / 3

```



'''
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
model1 = CNN_1()

# Print the model architecture
print(model1)
'''