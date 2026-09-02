import os
import cv2
from zipfile import ZipFile
import tempfile


def merge_images(color_zip_path, silhouette_zip_path, output_zip_path):
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract color and silhouette images from the zip files
        with ZipFile(color_zip_path, 'r') as color_zip, ZipFile(silhouette_zip_path, 'r') as silhouette_zip:
            color_zip.extractall(temp_dir)
            silhouette_zip.extractall(temp_dir)

        # Create a zip file for the merged images
        with ZipFile(output_zip_path, 'w') as output_zip:
            # Iterate over files in the temporary directory
            for color_filename in os.listdir(temp_dir):
                if color_filename.endswith(".png"):
                    # Read the color and silhouette images
                    color_image = cv2.imread(os.path.join(temp_dir, color_filename))
                    silhouette_image = cv2.imread(os.path.join(temp_dir, color_filename.replace(".png", "_silhouette.png")), cv2.IMREAD_GRAYSCALE)

                    # Convert silhouette image to binary mask
                    _, silhouette_mask = cv2.threshold(silhouette_image, 10, 255, cv2.THRESH_BINARY)

                    # Create a three-channel silhouette mask
                    silhouette_mask_3ch = cv2.merge((silhouette_mask, silhouette_mask, silhouette_mask))

                    # Apply the silhouette mask to the color image
                    merged_image = cv2.bitwise_and(color_image, silhouette_mask_3ch)

                    # Write the merged image to the output zip file
                    output_filename = os.path.basename(color_filename)
                    output_zip.writestr(output_filename, cv2.imencode('.png', merged_image)[1].tobytes())


# Example usage:
color_zip_path = r"D:\\simulationMM\\python Proximity\\2024_output_rot_old\\color.zip"
silhouette_zip_path = r"D:\\simulationMM\\python Proximity\\2024_output_rot_old\\silhouette.zip"
output_zip_path = r"D:\\simulationMM\\python Proximity\\2024_output_rot_old\\output.zip"

merge_images(color_zip_path, silhouette_zip_path, output_zip_path)

