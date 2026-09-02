import os
import cv2


def merge_images(color_folder, silhouette_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files in the color folder
    color_files = os.listdir(color_folder)
    # print(color_files)

    # Iterate over each file in the color folder
    for color_file in color_files:
        # Check if the file is a PNG image
        if color_file.endswith(".png"):
            # Construct the paths for the color and silhouette images
            color_image_path = os.path.join(color_folder, color_file)
            silhouette_image_path = os.path.join(silhouette_folder, color_file)

            # Read the color image
            color_image = cv2.imread(color_image_path)

            # Read the silhouette image
            silhouette_image = cv2.imread(silhouette_image_path, cv2.IMREAD_GRAYSCALE)

            # Convert silhouette image to binary mask
            _, silhouette_mask = cv2.threshold(silhouette_image, 10, 255, cv2.THRESH_BINARY)

            # Create a three-channel silhouette mask
            silhouette_mask_3ch = cv2.merge((silhouette_mask, silhouette_mask, silhouette_mask))

            # Apply the silhouette mask to the color image
            merged_image = cv2.bitwise_and(color_image, silhouette_mask_3ch)

            # Save the merged image
            output_path = os.path.join(output_folder, "merged_" + color_file)
            cv2.imwrite(output_path, merged_image)


# Example usage:
# color_folder = r"D:\simulationMM\python Proximity\2024_output_rot_old\color_0_14268_all"
# silhouette_folder = r"D:\simulationMM\python Proximity\2024_output_rot_old\sillhouette_0_14268_all"

color_folder = r"D:\simulationMM\python Proximity\2024_output_rot_new\color"
silhouette_folder = r"D:\simulationMM\python Proximity\2024_output_rot_new\silhouette"

output_folder = r"D:\simulationMM\python Proximity\2024_output_rot_new\merged"

merge_images(color_folder, silhouette_folder, output_folder)
