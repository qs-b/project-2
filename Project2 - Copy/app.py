from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define class labels
class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Function to preprocess the input image
def preprocess_image(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img)  # Convert to NumPy array
    img = img.reshape((1, 28, 28, 1))  # Reshape to match model input shape
    img = img / 255.0  # Normalize pixel values between 0 and 1
    return img

# Function to make predictions on the input image
def predict_image(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    prediction_indices = np.argsort(predictions[0])[::-1]
    labels_with_percentages = []
    for index in prediction_indices:
        label = class_labels[index]
        percentage = predictions[0][index] * 100
        labels_with_percentages.append((label, percentage))
    return labels_with_percentages

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']
        # Read the image file
        image = Image.open(image_file)
        # Make prediction
        labels_with_percentages = predict_image(image)
        return render_template('index.html', predictions=labels_with_percentages)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)