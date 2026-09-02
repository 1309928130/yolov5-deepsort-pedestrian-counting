import os
import csv
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

def get_video_metadata(file_path):
    parser = createParser(file_path)
    metadata = extractMetadata(parser)
    return metadata

folder_path = r'E:\minyu\cologne\automation_car'
csv_file_path = os.path.join(folder_path, 'all-processed_info.csv')

# Read existing CSV file
data = []
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Read the header
    data = [row for row in csv_reader]

# Update CSV file with video lengths
for row in data:
    if len(row) > 0:
        file_name = row[0]  # Assuming the file name is in the first column
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a video (you can customize the list of video extensions)
        if file_name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            # Get video metadata
            metadata = get_video_metadata(file_path)

            if metadata:
                # Add video length to the end of the row
                row.append(metadata.get('duration', 'Unknown'))
            else:
                row.append('Unknown')

# Save the updated CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header + ['Video Length'])  # Add 'Video Length' to the header
    csv_writer.writerows(data)

print(f'Video lengths have been added to: {csv_file_path}')
