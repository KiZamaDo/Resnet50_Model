from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import cv2

app = Flask(__name__)

dic = {0: 'Healthy', 1: 'Diseased'}

# Load the model
model = load_model('model.h5')
model.make_predict_function()  # Ensure that the model is thread-safe

def predict_label(img_path):
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    new_width = int(width * 0.52)
    new_height = int(height * 0.68)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    # Resize the cropped image to a fixed size
    target_size = (224, 224)
    resized_image = cv2.resize(cropped_image, target_size)
    # Preprocess the image (if required) before prediction
    # Perform prediction
    p = model.predict_classes(resized_image)
    return dic[p[0]]

# Route for serving the index page
@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return render_template("index.html", error="No file uploaded")
        img = request.files['my_image']
        if img.filename == '':
            return render_template("index.html", error="No selected file")

        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return render_template("index.html", prediction=p, img_path=img_path)

    return render_template("index.html")

# Route for handling image classification
@app.route("/predict", methods=['POST'])
def predict():
    print(request.files)
    if 'my_image' not in request.files:
        return jsonify({'error': 'No file part'})

    img = request.files['my_image']
    if img.filename == '':
        return jsonify({'error': 'No selected file'})

    img_path = "static/" + img.filename
    img.save(img_path)
    p = predict_label(img_path)
    return jsonify({'prediction': p, 'img_path': img_path})

if __name__ == '__main__':
    app.run(debug=True)
