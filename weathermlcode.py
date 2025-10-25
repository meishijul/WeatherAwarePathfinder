import csv
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFSegformerForSemanticSegmentation, SegformerFeatureExtractor

# Path to the CSV file
csv_file_path = 'OctoberHackathonData.csv'

# Lists to store the data
image_names = []
classifications = []

# Read the CSV file
try:
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        # Skip the header row if it exists
        header = next(csv_reader)
        
        # Extract the first two columns
        for row in csv_reader:
            image_names.append(row[0])  # First column: Image name
            classifications.append(row[1])  # Second column: Classification (dusty/clear)
    
    # Print the extracted data
    print("Image Names:", image_names)
    print("Classifications:", classifications)

except Exception as e:
    print(f"An error occurred: {e}")
    # Verify if the images exist in the 'images' folder

    images_folder_path = 'images'
    missing_images = []

    for image_name in image_names:
        image_path = os.path.join(images_folder_path, image_name)
        if not os.path.isfile(image_path):
            missing_images.append(image_name)

    if missing_images:
        print("The following images are missing in the 'images' folder:", missing_images)
    else:
        print("All images are present in the 'images' folder.")
        # Print each image name with its corresponding label
        for image_name, classification in zip(image_names, classifications):
            print(f"Image: {image_name}, Label: {classification}")
            # Define U-Net model
            def unet_model(input_size=(128, 128, 3)):
                inputs = Input(input_size)

                # Encoder
                conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
                conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
                pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

                conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
                conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
                pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

                # Bottleneck
                conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
                conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

                # Decoder
                up1 = UpSampling2D(size=(2, 2))(conv3)
                up1 = concatenate([up1, conv2], axis=-1)
                conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
                conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

                up2 = UpSampling2D(size=(2, 2))(conv4)
                up2 = concatenate([up2, conv1], axis=-1)
                conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
                conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

                outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

                model = Model(inputs=[inputs], outputs=[outputs])
                model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

                return model

            # Load and preprocess images
            def load_images(folder_path, image_names, labels, target_size=(128, 128)):
                images = []
                for image_name in image_names:
                    image_path = os.path.join(folder_path, image_name)
                    img = load_img(image_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                images = np.array(images)
                labels = np.array([1 if label == 'dusty' else 0 for label in labels])
                labels = to_categorical(labels, num_classes=2)
                return images, labels

            # Paths
            future_images_folder = 'futureimages'

            # Load images and labels
            X, y = load_images(future_images_folder, image_names, classifications)

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the model
            model = unet_model()
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

            # Save the model
            model.save('unet_model.h5')
            # Load a pre-trained Segformer model from Hugging Face
            feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

            # Preprocess images for the Segformer model
            def preprocess_images(image_paths, target_size=(512, 512)):
                images = []
                for image_path in image_paths:
                    img = load_img(image_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                return np.array(images)

            # Load and preprocess future images
            future_image_paths = [os.path.join(future_images_folder, img) for img in image_names]
            X_future = preprocess_images(future_image_paths)

            # Perform inference on future images
            outputs = model.predict(X_future)
            predictions = np.argmax(outputs.logits, axis=-1)

            # Print predictions for future images
            for image_name, prediction in zip(image_names, predictions):
                label = "dusty" if prediction == 1 else "clear"
                print(f"Image: {image_name}, Predicted Label: {label}")