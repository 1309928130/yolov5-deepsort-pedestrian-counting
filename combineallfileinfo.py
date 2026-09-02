import os
import csv

def process_csv_files(folder_path):
    # Create a new CSV file to store the processed information
    output_csv_path = os.path.join(folder_path, 'processed_info.csv')
    with open(output_csv_path, mode='w', newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)
        csv_writer.writerow(['File Name', '(Frame+IdentifiedTotal),(Up/left+Down/right+Total)'])

        # Iterate through all CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                csv_file_path = os.path.join(folder_path, filename)

                # Extract the modified file name
                modified_file_name = extract_modified_file_name(filename)

                # Read the last non-empty row of the CSV file
                last_row = read_last_non_empty_row(csv_file_path)

                # Write the information to the new CSV file
                csv_writer.writerow([modified_file_name, last_row])

# def extract_modified_file_name(original_file_name):
#     # Assuming the original file name format is either VID_...mp4_..._output or VID_...mp4_output.csv
#     parts = original_file_name.split('_')
#     modified_file_name = parts[-3] + '.mp4' if len(parts) > 3 else parts[-2] + '.mp4'
#     return modified_file_name

# def extract_modified_file_name(original_file_name):
#     # Assuming the original file name format is either VID_...mp4_..._output or VID_...mp4_output.csv
#     parts = original_file_name.split('_')
#     modified_file_name = "_".join(parts[:-1]).split('.mp4')[0] + '.mp4'
#     return modified_file_name

def extract_modified_file_name(original_file_name):
    # Assuming the original file name format is either VID_...mp4_..._output or VID_...mp4_output.csv

    modified_file_name = ""
    if '.mp4' in original_file_name:
        modified_file_name = original_file_name.split('.mp4')[0] + '.mp4'
    elif '.avi' in original_file_name:
        modified_file_name = original_file_name.split('.avi')[0] + '.avi'
    return modified_file_name


def read_last_non_empty_row(csv_file_path):
    with open(csv_file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():  # Check if the line is not empty after stripping whitespace
                return lines[i].strip()

if __name__ == '__main__':
    folder_path = r'E:\videos\allcountingresults'
    process_csv_files(folder_path)
