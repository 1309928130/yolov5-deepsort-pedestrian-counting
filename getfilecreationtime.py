import os
import csv
from datetime import datetime
import win32file
import pywintypes

folder_path = r'E:\videos'
csv_file_path = r'E:\videos\file_list.csv'


def get_metadata_creation_time(file_path):
    # Use win32file to get the file creation time from metadata
    handle = win32file.CreateFile(
        file_path,
        win32file.GENERIC_READ,
        win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
        None,
        win32file.OPEN_EXISTING,
        0,
        None
    )
    creation_time = win32file.GetFileTime(handle)[0]
    win32file.CloseHandle(handle)
    return creation_time


# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Create a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header to the CSV file
    csv_writer.writerow(['File Name', 'Creation Time'])

    # Iterate through each file in the folder
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        # Get the creation time of the file from metadata
        creation_time = get_metadata_creation_time(file_path)

        # Convert pywintypes.datetime to datetime and then format it
        creation_time_formatted = datetime.utcfromtimestamp(creation_time.timestamp()).strftime('%Y-%m-%d %H:%M:%S')

        # Write file name and creation time to the CSV file
        csv_writer.writerow([file_name, creation_time_formatted])

print(f'File list has been saved to: {csv_file_path}')
