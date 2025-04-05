import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from classifier import create_model, load_checkpoint
from helper import predict_image
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load class names from validation folder
val_dir = config.VAL_DIR  # e.g., r"C:\Users\Sandhya\Deep_Learning\DL_Practice\Flower_Dataset\val"
class_names = sorted(os.listdir(val_dir))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(len(class_names))
model = load_checkpoint(model, config.MODEL_SAVE_PATH)
model = model.to(device)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get list of sample images from the 'static/samples' directory
    sample_images = os.listdir('static/samples')

    if request.method == 'POST':
        if 'file' in request.files:  # File uploaded by the user
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Predict the class for the uploaded image
                predicted_class, confidence = predict_image(model, filepath, class_names)

                # Pass the filename (of uploaded image) to results.html
                return render_template('results.html',
                                       filename=file.filename,  # This is the filename of the uploaded image
                                       predicted_class=predicted_class,
                                       confidence=confidence,
                                       image_path=url_for('static', filename='uploads/' + file.filename))  # Pass the full path to display the image

        elif 'sample_image' in request.form:  # A sample image was clicked
            sample_image = request.form['sample_image']
            sample_image_path = os.path.join('static/samples', sample_image)

            # Predict the class for the clicked sample image
            predicted_class, confidence = predict_image(model, sample_image_path, class_names)

            # Pass the sample image filename to results.html
            return render_template('results.html',
                                   filename=sample_image,  # This is the filename of the clicked sample image
                                   predicted_class=predicted_class,
                                   confidence=confidence,
                                   image_path=url_for('static', filename='samples/' + sample_image))  # Pass the full path to display the sample image

    return render_template('index.html', sample_images=sample_images)

if __name__ == '__main__':
    app.run(debug=True)
