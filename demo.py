import os
import subprocess
from flask import Flask, request, send_from_directory, jsonify, redirect, url_for
import preprocess
from recommendation import recommend_fashion_items_cnn
import pickle
app = Flask(__name__)

UPLOAD_FOLDER_clothes = 'dataset/outerwear'
UPLOAD_FOLDER_person = 'dataset/women_top_reference_person_test'

app.config['UPLOAD_FOLDER_clothes'] = UPLOAD_FOLDER_clothes
app.config['UPLOAD_FOLDER_person'] = UPLOAD_FOLDER_person


def read_image_paths_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            image_paths = file.readlines()
            image_paths = [path.strip() for path in image_paths]
            return image_paths
    except Exception as e:
        print(f"Error reading image paths from file: {e}")
        return []


@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Input</title>
    <!-- Your CSS styles -->
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            margin-bottom:10px;
        }
        .heading-container {
            text-align: center;
        }
        #upload-container {
            display: flex;
            flex-direction: row;
            align-items: center;
            text-align:center;
        }
        #upload-container input[type="file"] {
            margin-right: 10px;
            text-align:center;            
        }
        #preview-container {
            display: flex;
            flex-direction: row;
            align-items: center;
            text-align:center;
        }
        #cloth-preview, #person-preview {
            max-width: 240px;
            max-height: 240px;
            margin-right: 40px;
            text-align:center;
        }
        #uploaded-images {
            margin-top: 20px;
            text-align:center;
        }
        #uploaded-images img {
            max-width: 250px;
            max-height: 250px;
            margin-right: 10px;
        }
        #simi-images, #comp-images {
            margin-top: 20px;
            text-align: center;
            display: block;
        }
        #simi-images img, #comp-images img {
            max-width: 150px;
            max-height: 150px;
            margin-right: 10px;
        }
        /* Custom button styles */
        .custom-button {
            display: none;
            margin: 20px auto;
            padding: 10px 20px;
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }
        .custom-button:hover {
            background-color: #45a049; /* Darker green */
        }
        /* Custom file input styles */
        .custom-file-input {
            display: inline-block;
            padding: 10px 20px;
            background-color: #3498db; /* Blue */
            border: none;
            color: white;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin-top: 20px;
            cursor: pointer;
            border-radius: 5px;
            margin-right:50px
        }
        .custom-file-input:hover {
            background-color: #2980b9; /* Darker blue */
        }
        /* Hide default file input button */
        input[type="file"] {
            display: none;
        }
    </style>
</head>
<body>
    <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">  
        <div>      
            <h1 class="heading-container">Upload Two Images</h1>
            <div id="upload-container">
                <!-- Custom file input buttons -->
                <label for="cloth-input" class="custom-file-input">Choose Cloth Image</label>
                <input type="file" name="cloth" id="cloth-input" accept="image/*">
                <label for="person-input" class="custom-file-input">Choose Person Image</label>
                <input type="file" name="person" id="person-input" accept="image/*">
                <button type="button" class="custom-file-input" id="capture-photo">Capture Photo</button>
                <label for="gender-select">Select Gender:</label>
                <select id="gender-select" name="gender">
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
            </div>
            <br>
            <div id="preview-container">
                <img id="cloth-preview" >
                <img id="person-preview">
            </div>
            <!-- Container for uploaded images -->
        </div>
        <input type="button" value="Generate" class="custom-button" id="generate-button">
        <div id="loading-spinner" style="display: none;">
            <img src="spinner.gif" alt="Loading...">
         </div>
        <div id="uploaded-images"></div>
        <div id="simi-images"></div>
        <div id="comp-images"></div>
    </form>
    <script>
    function showLoadingSpinner() {
        const spinner = document.getElementById('loading-spinner');
        spinner.style.display = 'block';
    }

    function hideLoadingSpinner() {
        const spinner = document.getElementById('loading-spinner');
        spinner.style.display = 'none';
    }

    function previewImage(input, previewId) {
        const preview = document.getElementById(previewId);
        const file = input.files[0];
        const reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }

    function checkImagesUploaded() {
        const clothInput = document.getElementById('cloth-input');
        const personInput = document.getElementById('person-input');
        const generateButton = document.getElementById('generate-button');

        if (clothInput.files.length > 0 && (personInput.files.length > 0 || document.getElementById('person-preview').src)) {
            generateButton.style.display = 'block';
        } else {
            generateButton.style.display = 'none';
        }
    }

    document.getElementById('cloth-input').addEventListener('change', function() {
        previewImage(this, 'cloth-preview');
        checkImagesUploaded();
    });

    document.getElementById('person-input').addEventListener('change', function() {
        previewImage(this, 'person-preview');
        checkImagesUploaded();
    });

    document.getElementById('generate-button').addEventListener('click', async function() {
        showLoadingSpinner(); // Show loading spinner

        const formData = new FormData(document.getElementById('upload-form')); // Get form data

        // Check if captured photo is available and add to formData if it is
        const capturedPhoto = document.getElementById('person-preview').src;
        if (capturedPhoto && !document.getElementById('person-input').files.length) {
            const blob = await fetch(capturedPhoto).then(res => res.blob());
            formData.append('captured_photo', blob, 'captured_photo.png');
        }

        const response = await fetch('/upload', { // Send form data to the server
            method: 'POST',
            body: formData
        });

        const result = await response.json(); // Parse JSON response

        hideLoadingSpinner(); // Hide loading spinner

        if (result.error) {
            alert(result.error); // Display error if any
        } else {
            // Display uploaded images without reloading the page
            const imagesContainer = document.getElementById('uploaded-images');
            imagesContainer.innerHTML = ''; // Clear previous images
            result.images.forEach(image => {
                const img = document.createElement('img');
                img.src = `${image}?timestamp=${new Date().getTime()}`; // Append timestamp to image URL
                imagesContainer.appendChild(img);
            });
            
            const simiContainer = document.getElementById('simi-images');
            simiContainer.innerHTML = ''; // Clear previous images
            const simiHeader = document.createElement('h3');
            simiHeader.textContent = 'Similar Images';
            simiContainer.appendChild(simiHeader);
            result.simi.forEach(image => {
                const img = document.createElement('img');
                img.src = `${image}?timestamp=${new Date().getTime()}`; // Append timestamp to image URL
                simiContainer.appendChild(img);
            });
            
            const compContainer = document.getElementById('comp-images');
            compContainer.innerHTML = ''; // Clear previous images
            const compHeader = document.createElement('h3');
            compHeader.textContent = 'Compatible Images';
            compContainer.appendChild(compHeader);
            result.comp.forEach(image => {
                const img = document.createElement('img');
                img.src = `${image}?timestamp=${new Date().getTime()}`; // Append timestamp to image URL
                compContainer.appendChild(img);
            });
        }
    });

    document.getElementById('capture-photo').addEventListener('click', function() {
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Get access to the camera
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                video.srcObject = stream;
                video.play();

                const modal = document.createElement('div');
                modal.style.position = 'fixed';
                modal.style.top = '0';
                modal.style.left = '0';
                modal.style.width = '100%';
                modal.style.height = '100%';
                modal.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
                modal.style.display = 'flex';
                modal.style.justifyContent = 'center';
                modal.style.alignItems = 'center';

                const captureButton = document.createElement('button');
                captureButton.textContent = 'Capture Photo';
                captureButton.style.position = 'absolute';
                captureButton.style.bottom = '20px';

                modal.appendChild(video);
                modal.appendChild(captureButton);
                document.body.appendChild(modal);

                captureButton.addEventListener('click', function() {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const dataURL = canvas.toDataURL('image/png');

                    previewCapturedImage(dataURL);
                    stopStream(stream);
                    document.body.removeChild(modal);
                });
            })
            .catch((error) => {
                console.error('Error accessing the camera: ', error);
            });

        function stopStream(stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        function previewCapturedImage(dataURL) {
            const preview = document.getElementById('person-preview');
            preview.src = dataURL;
            checkImagesUploaded();
        }
    });
    </script>

</body>
</html>

"""


@app.route('/upload_capture', methods=['POST'])
def upload_capture():
    if 'person' not in request.files:
        return jsonify({'error': 'No image uploaded.'})

    person = request.files['person']

    if person.filename == '':
        return jsonify({'error': 'No image selected for upload.'})

    if person:
        person.save(os.path.join(app.config['UPLOAD_FOLDER_person'], person.filename))
        person_path = os.path.join(app.config['UPLOAD_FOLDER_person'], person.filename).replace("\\", "/")

        return jsonify({'success': 'Image uploaded successfully.'})

    return jsonify({'error': 'An error occurred while uploading the image.'})


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle POST request for uploading images
        if 'cloth' not in request.files or (
                'person' not in request.files and 'captured_photo' not in request.files) or 'gender' not in request.form:
            return jsonify({'error': 'Please upload two images and select gender.'})

        cloth = request.files['cloth']
        person = request.files.get('person') or request.files.get('captured_photo')
        gender = request.form['gender']

        if cloth.filename == '' or (person and person.filename == ''):
            return jsonify({'error': 'Please select two images to upload.'})

        if cloth and person:
            # Save uploaded files to UPLOAD_FOLDER
            cloth.save(os.path.join(app.config['UPLOAD_FOLDER_clothes'], cloth.filename))
            person.save(os.path.join(app.config['UPLOAD_FOLDER_person'], person.filename))
            cloth_path = os.path.join(app.config['UPLOAD_FOLDER_clothes'], cloth.filename).replace("\\", "/")
            person_path = os.path.join(app.config['UPLOAD_FOLDER_person'], person.filename).replace("\\", "/")

            # 1. resize the images
            image_resized_path = preprocess.resize_single(person_path, app.config['UPLOAD_FOLDER_person'], 0)
            cloth_resized_path = preprocess.resize_single(cloth_path, app.config['UPLOAD_FOLDER_clothes'], 1)

            # 2. remove their backgrounds
            preprocess.remove_background_single('dataset/women_top_cloth_mask_test/', image_resized_path)
            preprocess.remove_background_single('dataset/women_top_cloth_mask_test/', cloth_resized_path)

            # 3. create the test pairs file
            preprocess.create_pairs_file_single(image_resized_path, cloth_resized_path, 'test_pairs.txt')
            # 4. create the cloth-mask
            preprocess.remove_background_single('dataset/women_top_cloth_mask_test/', cloth_resized_path, mask=True)

            # Run Python command to process images
            if gender == 'female':
                with open('featuresNew.pkl', 'rb') as f:
                    all_features, all_image_paths = pickle.load(f)
                recommend_fashion_items_cnn(cloth_path, all_features, all_image_paths, 'outerwear', top_n=6)
            else:
                with open('featuresMen.pkl', 'rb') as f:
                    all_features, all_image_paths = pickle.load(f)
                recommend_fashion_items_cnn(cloth_path, all_features, all_image_paths, 'men_outerwear', top_n=6)
            try:
                subprocess.run(
                    ["python", "test.py", "--name", "test_pairs", "--resize_or_crop", "scale_width", "--batchSize", "1",
                     "--gpu_ids", "0", "--hr", "--predmask"], check=True)
                image_paths = read_image_paths_from_file("static/output.txt")
                images = [f"/static/{path}" for path in image_paths]
                simi_paths = read_image_paths_from_file("recommended_simi_image_paths.txt")
                simi = [f"/static/{path}" for path in simi_paths]
                comp_paths = read_image_paths_from_file("recommended_comp_image_paths.txt")
                comp = [f"/static/{path}" for path in comp_paths]

                return jsonify({'images': images, 'simi': simi, 'comp': comp})
            except subprocess.CalledProcessError as e:
                return jsonify({'error': str(e)})

    elif request.method == 'GET':
        # Handle GET request for uploading images
        return redirect(url_for('index'))


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True)