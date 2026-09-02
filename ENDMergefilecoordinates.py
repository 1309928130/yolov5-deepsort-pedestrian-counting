import os
import csv

folder_path = r'E:\videos\allvideo_coordinates\ped'
input_file_name= os.path.join(folder_path, 'merged_coordinates_ped_modified.csv')
new_video_list_path = os.path.join(folder_path, input_file_name)
output_csv_path = os.path.join(folder_path, 'merged_coordinates_ped.csv')

# Read the video names from 'new_video_list_car.csv'
video_names = []
with open(new_video_list_path, 'r') as new_video_list_file:
    reader = csv.DictReader(new_video_list_file)
    for row in reader:
        video_names.append(row['videoName'])

# Write the merged information to a new CSV file
with open(output_csv_path, 'w', newline='') as output_csv_file:
    fieldnames = ['videoName', 'Cx1', 'Cy1', 'Cx2', 'Cy2']
    writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through the video names and extract information from other CSV files
    for video_name in video_names:
        video_info = {'videoName': video_name, 'Cx1': '', 'Cy1': '', 'Cx2': '', 'Cy2': ''}

        # Iterate through other CSV files
        for file in os.listdir(folder_path):
            if file.endswith('.csv') and file != input_file_name:
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as input_csv_file:
                    reader = csv.DictReader(input_csv_file)
                    for row in reader:
                        if 'Cx1' in row and 'Cy1' in row and 'Cx2' in row and 'Cy2' in row:
                            if row['videoName'] == video_name:
                                video_info['Cx1'] = row['Cx1']
                                video_info['Cy1'] = row['Cy1']
                                video_info['Cx2'] = row['Cx2']
                                video_info['Cy2'] = row['Cy2']
                                break

        # Write the merged information for the current video name
        writer.writerow(video_info)

print(f"Merged coordinates saved to: {output_csv_path}")
