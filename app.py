import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('plant_disease_model.h5')  # Update with your model name

# Class labels
class_labels = ['Healthy', 'Early Blight', 'Late Blight']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No file selected")

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the image for prediction
            img = image.load_img(file_path, target_size=(128, 128))  # Resize based on your model input
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)  # Get class index
            confidence = round(np.max(predictions) * 100, 2)  # Get accuracy

            return render_template("index.html", filename=file.filename, prediction=class_labels[predicted_class], accuracy=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
