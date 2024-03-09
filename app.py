# from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import cv2, io
from PIL import Image
import numpy as np

app = Flask(__name__)

app.config['DEBUG'] = True
app.config['ENV'] = 'development'

dic = {0: 'Healthy', 1: 'Diseased'}

# Load the model
model = load_model('model.h5')
model.make_predict_function()  # Ensure that the model is thread-safe

# List to store images
image_list = []

def predict_label(img):
    image = np.array(img)  # Convert PIL Image to NumPy array
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (OpenCV uses BGR by default)
    height, width, _ = image.shape
    new_width = int(width * 0.52)
    new_height = int(height * 0.68)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    # Resize the cropped image to a fixed size
    target_size = (224, 224)
    resized_image = cv2.resize(cropped_image, target_size)
    image_list.append(resized_image)
    # Perform prediction
    return None

# Route for serving the index page
@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

# Route for handling image classification
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return jsonify({'error': 'No file part'})

        img = request.files['my_image']
        if img.filename == '':
            return jsonify({'error': 'No selected file'})

        img_bytes = img.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Predict and store result
        predict_label(img)
        
        # Convert image list to NumPy array
        image_array = np.array(image_list)
        
        # Predict result for the numpy array
        result = model.predict(image_array)
        
        # Round off the result
        rounded_result = np.round(result)
        
        # Map the rounded values to labels using the dictionary
        labels = [dic[val] for val in rounded_result.flatten().tolist()]
        
        # Clear image list for next prediction
        image_list.clear()
        
        return jsonify({'result': labels})
    
    return jsonify({'res': "NONE"})

if __name__ == '__main__':
    app.run(debug=True)
