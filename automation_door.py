
import yaml
import time
import datetime
import os
import cv2
import csv
import torch
import warnings
import argparse
import numpy as np
import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from person_detect_yolov5 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
# count
from collections import Counter
from collections import deque
import math
from PIL import Image, ImageDraw, ImageFont


# Add this function to check if a point is inside the polygon
def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def draw_polygon(img, polygon, color=(0, 255, 255), thickness=2):
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_crossing_edges(img, polygon, paths, color=(0, 0, 255), thickness=2):
    for path_id, points in paths.items():
        for i in range(len(points) - 1):
            pt1, pt2 = points[i], points[i + 1]
            if intersect(pt1, polygon) and intersect(pt2, polygon):
                cv2.line(img, pt1, pt2, color, thickness)

# Modify intersect function to use point_inside_polygon
def intersect(midpoint, polygon):
    x, y = midpoint
    return point_inside_polygon(x, y, polygon)

def read_automation_csv(csv_file):
    video_info = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_info.append(row)

    return video_info


def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
    return midpoint, midpoint[0]


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))


def get_size_with_pil(label,size=25):
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf
    return font.getsize(label)


#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式


colors = np.array([
    [1,0,1],
    [0,0,1],
    [0,1,1],
    [0,1,0],
    [1,1,0],
    [1,0,0]
    ]);

def get_color(c, x, max):
    ratio = (x / max) * 5;
    i = math.floor(ratio);
    j = math.ceil(ratio);
    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;

def compute_color_for_labels(class_id,class_total=80):
    offset = (class_id + 0) * 123457 % class_total;
    red = get_color(2, offset, class_total);
    green = get_color(1, offset, class_total);
    blue = get_color(0, offset, class_total);
    return (int(red*256),int(green*256),int(blue*256))

class yolo_reid():
    def __init__(self, cfg, args, path, line=None):
        self.logger = get_logger("root")
        self.args = args
        self.video_path = path
        self.line = line  # Placeholder for line coordinates
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.person_detect = Person_detect(self.args, self.video_path)
        imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        self.dataset = LoadImages(self.video_path, img_size=imgsz)
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)

        # Extract video file name and directory
        video_dir, video_name = os.path.split(path)
        video_name_without_extension, extension = os.path.splitext(video_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(video_dir, f'{video_name_without_extension + extension}_{timestamp}_output.csv')

        # CSV Logging
        self.csv_file = open(csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Frame+Total", 'line1', "(Up+Down+Total)"])

        # Open the input video file
        vid_cap = cv2.VideoCapture(args.video_path)

        # Check if the video file was opened successfully
        if not vid_cap.isOpened():
            raise Exception("Error: Could not open video file")

        # Video Output
        frame_width = int(vid_cap.get(3))  # Width of the frames in the video
        frame_height = int(vid_cap.get(4))  # Height of the frames in the video

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or XVID
        # self.output_video = cv2.VideoWriter(os.path.join(video_dir, f'{video_name_without_extension}_output_video.avi'), fourcc, 20.0, (imgsz, imgsz))
        self.output_video = cv2.VideoWriter(os.path.join(video_dir, f'{video_name_without_extension + extension}_{timestamp}_output_video.avi'), fourcc, 30, (frame_width, frame_height)) # int(frame_width/2), int(frame_height/2)

    def set_polygon(self, polygon_coordinates):
        self.polygon = polygon_coordinates

    def deep_sort(self):
        idx_frame = 0
        results = []
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        total_counter = 0
        up_count = 0
        down_count = 0

        entering_count = 0
        exiting_count = 0
        already_counted_direction = None

        class_counter = Counter()   # store counts of each detected class
        already_counted = deque(maxlen=50)   # temporary memory for storing counted IDs

        # for track_id, points in paths.items():
        #     if point_inside_polygon(points[-1][0], points[-1][1], self.polygon):
        #         # Check if the track is already counted
        #         if track_id not in already_counted:
        #             entering_count += 1
        #             already_counted.add(track_id)
        #     else:
        #         # Check if the track is already counted
        #         if track_id not in already_counted:
        #             exiting_count += 1
        #             already_counted.add(track_id)

        # Initialize a dictionary to store the movement history of each tracked person
        movement_history = {}

        # for video_path, img, ori_img, vid_cap in self.dataset:
        for idx_frame, (video_path, img, ori_img, vid_cap) in enumerate(self.dataset):

            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            t1 = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # OpenCV color channels are in BGR order, so (0, 255, 255) corresponds to yellow
            draw_polygon(ori_img, self.polygon, color=(0, 255, 255))

            # 2. 统计人数
            for track in outputs:
                bbox = track[:4]
                track_id = track[-1]
                midpoint, x_coordinate = tlbr_midpoint(bbox)

                if track_id not in movement_history:
                    movement_history[track_id] = []

                movement_history[track_id].append(midpoint)

                # Determine the direction based on whether the pedestrian's movement intersects with the polygon's edges
                if len(movement_history[track_id]) > 1:
                    previous_midpoint = movement_history[track_id][-2]
                    movement_intersects_polygon = intersect(midpoint, self.polygon) or intersect(previous_midpoint,
                                                                                                 self.polygon)

                    if movement_intersects_polygon and track_id not in already_counted:
                        direction = "entering" if point_inside_polygon(midpoint[0], midpoint[1],
                                                                       self.polygon) else "exiting"

                        if direction == "entering":
                            entering_count += 1
                        elif direction == "exiting":
                            exiting_count += 1

                        already_counted.append(track_id)
                        draw_polygon(ori_img, self.polygon, color=(0, 0, 255), thickness=10)  # draw red line

                if len(paths) > 50:
                    del paths[list(paths)[0]]

            # 3. 绘制人员
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                # results.append((idx_frame - 1, bbox_tlwh, identities))
            # print("yolo+deepsort:", time_synchronized() - t1)

            # 4. 绘制统计信息
            label = "总行人数: {}".format(str(total_track))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 50
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

            label = "进入区域人数: {}  退出区域人数: {}".format(str(entering_count), str(exiting_count))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 100
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

            # if last_track_id >= 0:
            #     label = "最新行人: {}号，方向: {}".format(str(last_track_id), "向上/左" if angle >= 0 else "向下/右")
            #     t_size = get_size_with_pil(label, 25)
            #     x1 = 20
            #     y1 = 150
            #     color = compute_color_for_labels(2)
            #     cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            #     ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))

            for track_id, points in paths.items():
                for i in range(len(points) - 1):
                    pt1, pt2 = points[i], points[i + 1]
                    if intersect(pt1, self.polygon) and intersect(pt2, self.polygon):
                        cv2.line(ori_img, pt1, pt2, (0, 0, 255), 2)
                        label = "行人: {} 号，{}".format(str(track_id), "进入" if point_inside_polygon(pt1[0], pt1[1],
                                                                                                      self.polygon) else "退出")
                        t_size = get_size_with_pil(label, 25)
                        x1 = int((pt1[0] + pt2[0]) / 2)
                        y1 = int((pt1[1] + pt2[1]) / 2)
                        color = compute_color_for_labels(2)
                        cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
                        ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2),
                                                               (255, 0, 0))
            end = time_synchronized()

            if self.args.display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.logger.info("{}/time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(idx_frame, end - t1, 1 / (end - t1),
                                     bbox_xywh.shape[0], len(outputs)))
            self.logger.info("Entering count: {}, Exiting count: {}".format(entering_count, exiting_count))

            # Write the data to the CSV file
            row_data = [idx_frame, entering_count, exiting_count, total_counter]
            self.csv_writer.writerow(row_data)

            # Save the processed frame to the output video
            self.output_video.write(ori_img)

        self.csv_file.close()
        self.output_video.release()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", default=None, type=str)

    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0,1,2], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--config_deepsort", type=str, default="D:/downloads/yolov5-deepsort-pedestrian-counting-master/configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show result')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    return parser.parse_args()



def main():
    args = parse_args()

    folder_path = r"E:\videos\automation"  # CES
    automation_csv_path = r'E:\videos\automation\automation_door.csv'
    video_info_list = read_automation_csv(automation_csv_path)
    print(len(video_info_list))

    # Record the time
    start_time = time.time()

    for video_info in video_info_list:
        print(video_info)
        video_path = os.path.join(folder_path, video_info['videoName'])
        args.video_path = video_path

        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)

        with torch.no_grad():
            yolo_reid_instance = yolo_reid(cfg, args, path=args.video_path)
            # Extract polygon coordinates from CSV
            _, _, ori_img, _ = next(iter(yolo_reid_instance.dataset))
            polygon_coordinates = [
                (int(float(video_info['Cx1']) * ori_img.shape[1]), int(float(video_info['Cy1']) * ori_img.shape[0])),
                (int(float(video_info['Cx2']) * ori_img.shape[1]), int(float(video_info['Cy2']) * ori_img.shape[0])),
                (int(float(video_info['Cx3']) * ori_img.shape[1]), int(float(video_info['Cy3']) * ori_img.shape[0])),
                (int(float(video_info['Cx4']) * ori_img.shape[1]), int(float(video_info['Cy4']) * ori_img.shape[0])),
                # ...Add more coordinates as needed
            ]
            yolo_reid_instance.set_polygon(polygon_coordinates)

            yolo_reid_instance.deep_sort()

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for {video_path}: {execution_time} seconds")

if __name__ == '__main__':
    main()