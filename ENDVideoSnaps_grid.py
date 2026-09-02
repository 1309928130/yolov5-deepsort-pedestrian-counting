import os
import cv2
import numpy as np
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


def draw_grid_and_coordinates(image, rows=3, cols=3):
    height, width, _ = image.shape
    cell_width = width // 10
    cell_height = height // 10

    # Draw vertical lines
    for i in range(1, 10):
        x = i * cell_width
        cv2.line(image, (x, 0), (x, height), (0, 255, 255), 1)
        cv2.putText(image, f"{i / 10:.1f}", (x, int(height * 0.02)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)

    # Draw horizontal lines and write coordinates
    for j in range(1, 10):
        y = j * cell_height
        cv2.line(image, (0, y), (width, y), (0, 255, 255), 1)
        cv2.putText(image, f"{j / 10:.1f}", (int(width * 0.01), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)

        # Write coordinates at every intersection
        for i in range(1, 10):
            x_coord = i / 10.0
            y_coord = j / 10.0
            x_pos = int(i * cell_width - 0.5 * cell_width)
            y_pos = int(j * cell_height - 0.5 * cell_height)
            cv2.putText(image, f"({x_coord:.1f}, {y_coord:.1f})", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)


def create_montage_with_grid(frames, rows=3, cols=3):
    height, width, _ = frames[0].shape
    montage = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(frames):
                frame = frames[index].copy()
                draw_grid_and_coordinates(frame)
                montage[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = frame

    return montage


def process_videos(input_folder, output_folder):
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.avi', '.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        frames = get_frames(video_path)
        montage = create_montage_with_grid(frames)

        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_montage_with_grid.png")
        s = cv2.imwrite(output_path, montage)
        print(f"Montage with grid saved: {output_path}")


if __name__ == "__main__":
    input_folder = r'E:\videos\7'
    output_folder = r'E:\videos\7\extracted_frames'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_videos(input_folder, output_folder)
