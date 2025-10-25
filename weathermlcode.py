import csv

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

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except IndexError:
    print("Error: The CSV file does not have the expected number of columns.")
except Exception as e:
    print(f"An error occurred: {e}")