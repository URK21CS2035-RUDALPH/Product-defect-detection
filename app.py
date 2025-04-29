from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import pandas as pd
import imagesize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
import cv2
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO


plt.style.use('ggplot')
app = Flask(__name__)

# Set up paths
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper class and functions
class prepare_annotations:
    def __init__(self, sourcedir='datasets/images/jarlids_annots.csv', imagesdir='datasets/images/', destdir='static/dest/', yamlfile='train.yaml', trainstart=1, trainend=125, valstart=200, valend=239, teststart=1, testend=6):
        self.sourcedir = sourcedir
        self.imagesdir = imagesdir
        self.destdir = destdir
        self.yamlfile = yamlfile
        self.trainstart = trainstart
        self.trainend = trainend
        self.valstart = valstart
        self.valend = valend
        self.teststart = teststart
        self.testend = testend

    def isize(self, filename):
        w, h = imagesize.get(os.path.join(self.imagesdir, filename))
        return w, h

    def load_dataframe(self):
        dat = pd.read_csv(self.sourcedir)
        tr = ['p'+str(s)+'.JPG' for s in range(self.trainstart, self.trainend)]
        va = ['p'+str(s)+'.JPG' for s in range(self.valstart, self.valend)]
        te = ['t'+str(s)+'.JPG' for s in range(self.teststart, self.testend)]
        
        for i, j in zip(['train', 'val', 'test'], [tr, va, te]):
            dat.loc[dat['filename'].isin(j), 'dataset'] = i
        dat['region_attributes'] = dat['region_attributes'].replace({'{}': 'None'})
        dat = dat[~dat['region_attributes'].isin(['None'])].reset_index(drop=True)
        
        dat['category_names'] = dat['region_attributes'].apply(lambda x: str(list(eval(x).values())[0]))
        dat['category_codes'] = dat[['category_names']].apply(lambda x: pd.Categorical(x).codes)
        dat['image_width'] = dat['filename'].apply(lambda x: self.isize(x)[0])
        dat['image_height'] = dat['filename'].apply(lambda x: self.isize(x)[1])
        dat['x_min'] = dat['region_shape_attributes'].apply(lambda x: eval(x)['x'])
        dat['y_min'] = dat['region_shape_attributes'].apply(lambda x: eval(x)['y'])
        dat['bb_width'] = dat['region_shape_attributes'].apply(lambda x: eval(x)['width'])
        dat['bb_height'] = dat['region_shape_attributes'].apply(lambda x: eval(x)['height'])
        dat['n_x_center'] = (((dat['x_min'] + dat['bb_width']) + dat['x_min']) / 2) / dat['image_width']
        dat['n_y_center'] = (((dat['y_min'] + dat['bb_height']) + dat['y_min']) / 2) / dat['image_height']
        dat['n_width'] = dat['bb_width'] / dat['image_width']
        dat['n_height'] = dat['bb_height'] / dat['image_height']
        dat['color_cat'] = dat['category_names'].replace({'intact': 'green', 'damaged': 'red'})
        return dat

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("/upload triggered")
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        print("There is right file in it")
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Process image and annotation
        img = Image.open(file_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("c:\WINDOWS\Fonts\CASTELAR.TTF", 16, encoding="unic")
        draw.text((8, 8), "Examples of jar lid damage", font=font, fill=(255, 255, 255))
        
        # Save the processed image with annotations in the output folder
        annotated_output_file = 'annotated_' + file.filename
        annotated_output_path = os.path.join(OUTPUT_FOLDER, annotated_output_file)
        img.save(annotated_output_path)
        learning_curve_img_path,conf_matrix_img_path = load_graph()
        print(learning_curve_img_path)
        # Perform annotation visualization
        df = prepare_annotations().load_dataframe()
        visualisation_annotations(df, 'datasets/images/', file.filename)
        return render_template('index.html', filename=file.filename, output_filename=annotated_output_file, graph=learning_curve_img_path,matrix=conf_matrix_img_path)

def visualisation_annotations(dat, filedir, fname):
    im = Image.open(os.path.join(filedir, fname))
    fig, ax = plt.subplots(figsize=(14, 20))
    ax.imshow(im)
    ndat = dat[dat['filename'] == fname].reset_index()
    for i in range(len(ndat)):    
        xmin = ndat['x_min'][i]
        ymin = ndat['y_min'][i]
        w = ndat['bb_width'][i]
        h = ndat['bb_height'][i]
        color = ndat['color_cat'][i]
        label = "Non-Defect" if color == 'green' else "Defect"

        # Draw the rectangle
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Draw the label with a background box for better visibility
        ax.text(xmin, ymin - 10, label, color='white', fontsize=16, weight='bold',
                bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.5'))

    # Save annotated image
    annotated_output_path = os.path.join(OUTPUT_FOLDER, 'annotated_' + fname)
    plt.savefig(annotated_output_path)
    plt.close()

def get_image_regions_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    regions = []
    for _, row in df.iterrows():
        filename = row['filename']
        region_shape = json.loads(row['region_shape_attributes'])
        region_label = json.loads(row['region_attributes'])['type']
        
        # Convert the region to coordinates
        x, y, width, height = region_shape['x'], region_shape['y'], region_shape['width'], region_shape['height']
        
        # Create a dict for each region's image
        regions.append({
            'filename': filename,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'label': region_label
        })
    return regions

# Function to load images based on the regions
def load_image_region(img_path, x, y, width, height):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path}")
    
    # Load the image in grayscale (1 channel)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at path: {img_path}")
    
    # Crop image based on region coordinates
    cropped_img = image[y:y+height, x:x+width]
    
    # Resize the cropped image to (300, 300)
    cropped_img_resized = cv2.resize(cropped_img, (300, 300))
    
    return cropped_img_resized

# Extract labels and images
csv_file = 'jarlids_annots.csv'
regions = get_image_regions_from_csv(csv_file)
def load_graph():
    images = []
    labels = []
    for region in regions:
        img_path = os.path.join('D:\yolov8-defectdetection\ProductDefectDetection\yolov7\datasets\images', region['filename'])
        cropped_img = load_image_region(img_path, region['x'], region['y'], region['width'], region['height'])
        images.append(cropped_img)
        labels.append(1 if region['label'] == 'damaged' else 0)  # Label: 1 for damaged, 0 for intact
    images = np.array(images)
    labels = np.array(labels)

    # Normalize images to [0, 1]
    images = images / 255.0

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Reshape images to (batch_size, height, width, channels)
    X_train = X_train.reshape(X_train.shape[0], 300, 300, 1)
    X_val = X_val.reshape(X_val.shape[0], 300, 300, 1)

    yolo_model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', strides=2, input_shape=(300, 300, 1)),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(64, 3, activation='relu', padding='same', strides=2),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Only 1 output (intact or damaged)
    ])

    # Compile model
    yolo_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # ModelCheckpoint to save the model at best epoch
    checkpoint_callback = ModelCheckpoint('CNN_Casting_Inspection.hdf5', save_best_only=True, monitor='Dataset Analysis', verbose=1)

    # Fit model
    epochs = 2
    
    
    if os.path.exists('CNN_Casting_Inspection.hdf5'):
        best_model = load_model('CNN_Casting_Inspection.hdf5')
        yolo_model = best_model
        print("Resuming from last checkpoint...")

    yolo_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[checkpoint_callback], verbose=1)

    # Plot learning curve from model history and save it as an image
    histo_dict = yolo_model.history.history
    histo_df = pd.DataFrame(histo_dict, index=range(1, epochs+1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in histo_df.columns:
        ax.plot(histo_df.index, m, data=histo_df)
    ax.set_xlabel('Epoch')
    ax.set_title('Yolov8 Chart', loc='left', weight='bold')
    ax.legend()

    # Save learning curve plot
    learning_curve_img_path = "learning_curve.png"
    file_path = os.path.join(OUTPUT_FOLDER, learning_curve_img_path)
    plt.savefig(file_path)
    plt.close()

    # Make predictions on validation data
    y_pred_prob = yolo_model.predict(X_val, verbose=1)
    y_pred = (y_pred_prob >= 0.5).reshape(-1,)

    # Visualize the confusion matrix and save it as an image
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, annot_kws={'size': 14, 'weight': 'bold'}, fmt='d', cbar=False, cmap='Blues', ax=ax)
    ax.set_xticklabels(['Intact', 'Damaged'])
    ax.set_yticklabels(['Intact', 'Damaged'], va='center')
    plt.tick_params(axis='both', labelsize=14, length=0)
    plt.ylabel('Actual', size=14, weight='bold')
    plt.xlabel('Predicted', size=14, weight='bold')

    # Save confusion matrix plot
    conf_matrix_img_path = "confusion_matrix.png"
    
    file_path2 = os.path.join(OUTPUT_FOLDER, conf_matrix_img_path)
    
    plt.savefig(file_path2)
    plt.close()
    return learning_curve_img_path,conf_matrix_img_path
@app.route('/output/<filename>')
def output(filename):
    return render_template('output.html', filename=filename)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)