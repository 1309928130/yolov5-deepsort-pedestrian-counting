import os
import csv

def get_video_names(folder_path):
    video_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            video_names.append(file_name)
    return video_names

def write_video_names_to_csv(video_names, csv_file_path):
    # f = open(csv_file_path, 'w')
    # out = csv.writer(f, delimiter=",")
    # for video_name in video_names:
    #     out.writerow([video_name])
    # f.close()

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['videoName'])

        for video_name in video_names:
            csv_writer.writerow([video_name])

if __name__ == "__main__":
    folder_path = r'E:\videos\7'
    csv_file_path = r'E:\videos\7\automation_videos.csv'

    video_names = get_video_names(folder_path)
    write_video_names_to_csv(video_names, csv_file_path)
