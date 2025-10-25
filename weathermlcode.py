import csv
import os

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