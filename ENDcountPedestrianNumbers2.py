import cv2
import os
import csv
from datetime import datetime

def count_pedestrians(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define the color range for pedestrian detection
    lower_color = (228, 120, 228)
    upper_color = (248, 140, 248)

    # Create a mask to isolate pixels within the color range
    mask = cv2.inRange(image, lower_color, upper_color)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of contours (each contour represents a pedestrian)
    num_pedestrians = len(contours)

    # Draw contours on a copy of the original image
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Draw all contours in green

    return num_pedestrians, image_with_contours

# Folder paths
input_folder = r"D:\simulationMM\python Proximity\2024_output_rot_new\merged"
output_folder = os.path.join(input_folder, "contour_images")

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Timestamp for CSV file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_file_path = os.path.join(input_folder, f"pedestrian_counts_{timestamp}.csv")

# Process each image in the input folder
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File Name', 'Pedestrian Count'])

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust file extensions as needed
            image_path = os.path.join(input_folder, filename)
            num_pedestrians, image_with_contours = count_pedestrians(image_path)
            csv_writer.writerow([filename, num_pedestrians])

            # Save the image with contours
            contour_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(contour_image_path, image_with_contours)

print("Processing complete. Results saved in:", csv_file_path)
