import os
import cv2
import numpy as np
import csv
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_frames(video_path, num_frames=9):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []

    for index in frames_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def draw_lines_arrows(image, lines_data):
    for line_data in lines_data:
        Cx1, Cy1, Cx2, Cy2 = line_data
        if Cx1 is not None and Cy1 is not None and Cx2 is not None and Cy2 is not None:
            pt1 = (int(Cx1 * image.shape[1]), int(Cy1 * image.shape[0]))
            pt2 = (int(Cx2 * image.shape[1]), int(Cy2 * image.shape[0]))
            cv2.line(image, pt1, pt2, (0, 0, 255), 5, cv2.LINE_AA)

            # Draw arrowhead
            arrow_size = 100
            angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
            pt3 = (int(pt2[0] + arrow_size * np.cos(angle + np.pi / 6)),
                   int(pt2[1] + arrow_size * np.sin(angle + np.pi / 6)))
            pt4 = (int(pt2[0] + arrow_size * np.cos(angle - np.pi / 6)),
                   int(pt2[1] + arrow_size * np.sin(angle - np.pi / 6)))
            cv2.line(image, pt2, pt3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.line(image, pt2, pt4, (0, 0, 255), 5, cv2.LINE_AA)


def process_videos(input_folder, output_folder, coordinates_file):
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.avi', '.mp4'))]

    with open(coordinates_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            video_name = row['videoName']
            Cx1, Cy1, Cx2, Cy2 = row.get('Cx1'), row.get('Cy1'), row.get('Cx2'), row.get('Cy2')

            video_path = os.path.join(input_folder, video_name)

            if (video_name in video_files) and \
                    os.path.exists(video_path) and \
                    Cx1 != '' and Cy1 != '' and Cx2 != '' and Cy2 != '':

                frames = get_frames(video_path)
                draw_lines_arrows(frames[0], [(float(Cx1), float(Cy1), float(Cx2), float(Cy2))])

                output_path = os.path.join(output_folder, f"{os.path.splitext(video_name)[0]}__montage_with_grid_and_arrows.png")
                cv2.imwrite(output_path, frames[0])
                print(f"Image with arrows saved: {output_path}")
            else:
                print(f"Skipping video {video_name} as it does not exist in the folder or coordinates are missing.")


if __name__ == "__main__":
    input_folder = r'E:\videos\7'
    output_folder = r'E:\videos\7\drawline_forcheck'
    coordinates_file = r'E:\videos\allvideo_coordinates\ped\merged_coordinates_ped_modified.csv'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_videos(input_folder, output_folder, coordinates_file)
