#!/usr/bin/python3
# -*- coding: utf-8 -*-

import yaml
import os
import cv2
import csv
import time
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

def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
    return midpoint


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))


def get_size_with_pil(label, size=25):
    # font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf
    font = ImageFont.truetype("./configs/IBMPlexSans-Bold.ttf", size, encoding="utf-8")  # simhei.ttf

    return font.getsize(label)


#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color,size=25):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8") #simhei.ttf
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
    def __init__(self, cfg, args, path):
        self.logger = get_logger("root")
        self.args = args
        self.video_path = path
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
        csv_filename = os.path.join(video_dir, f'{video_name_without_extension + extension}_output.csv')

        # CSV Logging
        self.csv_file = open(csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Frame+Total", 'line1', 'line2', 'line3', 'line4', "(Up+Down+Total)"])

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
        self.output_video = cv2.VideoWriter(os.path.join(video_dir, f'{video_name_without_extension + extension}_output_video.avi'), fourcc, 15.0, (frame_width, frame_height)) # int(frame_width/2), int(frame_height/2)


    def deep_sort(self):
        idx_frame = 0
        results = []
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        # total_counter = 0
        # up_count = 0
        # down_count = 0
        # Initialize separate counters for each line

        # Define the coordinates of your multiple lines


        class_counter = Counter()   # store counts of each detected class
        # Create separate queues for each line
        # line_already_counted = [deque(maxlen=50) for _ in range(len(lines))]

        # print(line_already_counted)
        # print(len(lines))

        sr = 1  # CES scalar

        # videoSize = [408, 720]
        videoSize = [1080,1920]

        # line1 = [(int(0.2 * videoSize[0]), 0), (int(0.2 * videoSize[0]), int(videoSize[1]))] # CES 线条坐标，记得把上下两组全都同步改掉

        line1 = [(int(0.1 * videoSize[0]), 0), (int(0.1 * videoSize[0]), int(videoSize[1]))]  # 划纵线
        line2 = [(int(0.2 * videoSize[0]), 0), (int(0.2 * videoSize[0]), int(videoSize[1]))]  # CES 线条坐标，记得把上下两组全都同步改掉
        line3 = [(int(0.3 * videoSize[0]), 0), (int(0.3 * videoSize[0]), int(videoSize[1]))]
        line4 = [(int(0.4 * videoSize[0]), 0), (int(0.4 * videoSize[0]), int(videoSize[1]))]
        # line5 = [(int(0.5 * videoSize[0]), 0), (int(0.5 * videoSize[0]), int(videoSize[1]))]
        # line6 = [(int(0.6 * videoSize[0]), 0), (int(0.6 * videoSize[0]), int(videoSize[1]))]
        # line7 = [(int(0.7 * videoSize[0]), 0), (int(0.7 * videoSize[0]), int(videoSize[1]))]
        # line8 = [(int(0.8 * videoSize[0]), 0), (int(0.8 * videoSize[0]), int(videoSize[1]))]
        # line9 = [(int(0.9 * videoSize[0]), 0), (int(0.9 * videoSize[0]), int(videoSize[1]))]
        # line10 = [(0, int(0.2 * videoSize[1])), (int(videoSize[0]), int(0.2 * videoSize[1]))]  # 划横线
        # line11 = [(0, int(0.4 * videoSize[1])), (int(videoSize[0]), int(0.4 * videoSize[1]))]
        # line12 = [(0, int(0.6 * videoSize[1])), (int(videoSize[0]), int(0.6 * videoSize[1]))]
        # line13 = [(0, int(0.8 * videoSize[1])), (int(videoSize[0]), int(0.8 * videoSize[1]))]

        # lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13]

        lines = [line1, line2, line3, line4]

        line1 = [(int(0.1 * videoSize[0]), 0), (int(0.1 * videoSize[0]), int(videoSize[1]))]  # 划纵线
        line2 = [(int(0.2 * videoSize[0]), 0), (int(0.2 * videoSize[0]), int(videoSize[1]))]  # CES 线条坐标，记得把上下两组全都同步改掉
        line3 = [(int(0.3 * videoSize[0]), 0), (int(0.3 * videoSize[0]), int(videoSize[1]))]
        line4 = [(int(0.4 * videoSize[0]), 0), (int(0.4 * videoSize[0]), int(videoSize[1]))]
        # line5 = [(int(0.5 * videoSize[0]), 0), (int(0.5 * videoSize[0]), int(videoSize[1]))]
        # line6 = [(int(0.6 * videoSize[0]), 0), (int(0.6 * videoSize[0]), int(videoSize[1]))]
        # line7 = [(int(0.7 * videoSize[0]), 0), (int(0.7 * videoSize[0]), int(videoSize[1]))]
        # line8 = [(int(0.8 * videoSize[0]), 0), (int(0.8 * videoSize[0]), int(videoSize[1]))]
        # line9 = [(int(0.9 * videoSize[0]), 0), (int(0.9 * videoSize[0]), int(videoSize[1]))]
        # line10 = [(0, int(0.2 * videoSize[1])), (int(videoSize[0]), int(0.2 * videoSize[1]))]  # 划横线
        # line11 = [(0, int(0.4 * videoSize[1])), (int(videoSize[0]), int(0.4 * videoSize[1]))]
        # line12 = [(0, int(0.6 * videoSize[1])), (int(videoSize[0]), int(0.6 * videoSize[1]))]
        # line13 = [(0, int(0.8 * videoSize[1])), (int(videoSize[0]), int(0.8 * videoSize[1]))]

        # line1 = [(int(0.2 * ori_img.shape[1]), 0),
        #          (int(0.2 * ori_img.shape[1]), int(ori_img.shape[0]))]  # 划纵线  shape[1] - width, shape[0] - height
        # line2 = [(int(0.4 * ori_img.shape[1]), 0),
        #          (int(0.4 * ori_img.shape[1]), int(ori_img.shape[0]))]  # CES 线条坐标，记得把上下两组全都同步改掉
        # line3 = [(int(0.6 * ori_img.shape[1]), 0), (int(0.6 * ori_img.shape[1]), int(ori_img.shape[0]))]
        # line4 = [(int(0.8 * ori_img.shape[1]), 0), (int(0.8 * ori_img.shape[1]), int(ori_img.shape[0]))]
        # line1 = [(0, int(0.48 * ori_img.shape[0])), (int(ori_img.shape[1]), int(0.48 * ori_img.shape[0])]  # 划横线

        line_up_counts = [0] * len(lines)
        line_down_counts = [0] * len(lines)
        line_total_counts = [0] * len(lines)


        # Real-Time Video with Overlays
        for video_path, img, ori_img, vid_cap in self.dataset:

            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            t1 = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # Create a list to store counters for each line
            line_counters = [0] * len(lines)

            # print('line counters', line_counters)

            line_display_coords = [(int(20*sr), int(50*sr)), (int(20*sr), int(100*sr)), (int(20*sr), int(150*sr)), (int(20*sr), int(200*sr)),
                                   (int(20*sr), int(250*sr)), (int(20*sr), int(300*sr)), (int(20*sr), int(350*sr)), (int(20*sr), int(400*sr)),
                                   (int(20*sr), int(450*sr)), (int(20*sr), int(500*sr)), (int(20*sr), int(550*sr)), (int(20*sr), int(600*sr)),
                                   (int(20*sr), int(650*sr))]  # Adjust as needed

            # Iterate over each line
            for line_index, line in enumerate(lines):

                line_counters[line_index] += 1  # Update the counter for the current line

                # 1. Draw the lines
                cv2.line(ori_img, line[0], line[1], (0, 255, 255), 1)

                # 2. 统计人数
                for track in outputs:
                    bbox = track[:4]
                    track_id = track[-1]
                    midpoint = tlbr_midpoint(bbox)
                    origin_midpoint = (midpoint[0], ori_img.shape[0] - midpoint[1])  # get midpoint respective to botton-left

                    if track_id not in paths:
                        paths[track_id] = deque(maxlen=2)
                        total_track = track_id
                    paths[track_id].append(midpoint)
                    previous_midpoint = paths[track_id][0]
                    origin_previous_midpoint = (previous_midpoint[0], ori_img.shape[0] - previous_midpoint[1])

                    # if intersect(midpoint, previous_midpoint, line[0], line[1]) and track_id not in line_already_counted[line_index]:
                    if intersect(midpoint, previous_midpoint, line1[0], line1[1]):
                        # print('lineIndex, Line_alreadyCounted', line_index, line_already_counted[line_index])
                        class_counter[track_cls] += 1
                        # total_counter += 1
                        # line_counters[line_index] += 1  # Update the counter for the current line
                        # print(line_counters)
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line1[0], line1[1], (0, 0, 255), int(10*sr))
                        # line_already_counted[line_index].append(track_id)  # Set already counted for ID to true.
                        # print('lineAreadyCounted_intersect', line_already_counted)
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        # Update counts specific to the current line
                        line_up_counts[line_index] += 1 if angle > 0 else 0
                        line_down_counts[line_index] += 1 if angle < 0 else 0
                        line_total_counts[line_index] += 1
                        crossed_line = 1

                    if intersect(midpoint, previous_midpoint, line2[0], line2[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line2[0], line2[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[1] += 1 if angle > 0 else 0
                        line_down_counts[1] += 1 if angle < 0 else 0
                        line_total_counts[1] += 1
                        crossed_line = 2
                    if intersect(midpoint, previous_midpoint, line3[0], line3[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line3[0], line3[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[2] += 1 if angle > 0 else 0
                        line_down_counts[2] += 1 if angle < 0 else 0
                        line_total_counts[2] += 1
                        crossed_line = 3
                    if intersect(midpoint, previous_midpoint, line4[0], line4[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line4[0], line4[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[3] += 1 if angle > 0 else 0
                        line_down_counts[3] += 1 if angle < 0 else 0
                        line_total_counts[3] += 1
                        crossed_line = 4

                    if intersect(midpoint, previous_midpoint, line5[0], line5[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line5[0], line5[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[4] += 1 if angle > 0 else 0
                        line_down_counts[4] += 1 if angle < 0 else 0
                        line_total_counts[4] += 1
                        crossed_line = 5
                    if intersect(midpoint, previous_midpoint, line6[0], line6[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line6[0], line6[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[5] += 1 if angle > 0 else 0
                        line_down_counts[5] += 1 if angle < 0 else 0
                        line_total_counts[5] += 1
                        crossed_line = 6
                    if intersect(midpoint, previous_midpoint, line7[0], line7[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line7[0], line7[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[6] += 1 if angle > 0 else 0
                        line_down_counts[6] += 1 if angle < 0 else 0
                        line_total_counts[6] += 1
                        crossed_line = 7
                    if intersect(midpoint, previous_midpoint, line8[0], line8[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line8[0], line8[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[7] += 1 if angle > 0 else 0
                        line_down_counts[7] += 1 if angle < 0 else 0
                        line_total_counts[7] += 1
                        crossed_line = 8
                    if intersect(midpoint, previous_midpoint, line9[0], line9[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line9[0], line9[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[8] += 1 if angle > 0 else 0
                        line_down_counts[8] += 1 if angle < 0 else 0
                        line_total_counts[8] += 1
                        crossed_line = 9
                    if intersect(midpoint, previous_midpoint, line10[0], line10[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line10[0], line10[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[9] += 1 if angle > 0 else 0
                        line_down_counts[9] += 1 if angle < 0 else 0
                        line_total_counts[9] += 1
                        crossed_line = 10
                    if intersect(midpoint, previous_midpoint, line11[0], line11[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line11[0], line11[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[10] += 1 if angle > 0 else 0
                        line_down_counts[10] += 1 if angle < 0 else 0
                        line_total_counts[10] += 1
                        crossed_line = 11
                    if intersect(midpoint, previous_midpoint, line12[0], line12[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line12[0], line12[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[11] += 1 if angle > 0 else 0
                        line_down_counts[11] += 1 if angle < 0 else 0
                        line_total_counts[11] += 1
                        crossed_line = 12
                    if intersect(midpoint, previous_midpoint, line13[0], line13[1]):
                        last_track_id = track_id;
                        # draw red line
                        cv2.line(ori_img, line13[0], line13[1], (0, 0, 255), int(10*sr))
                        angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                        line_up_counts[12] += 1 if angle > 0 else 0
                        line_down_counts[12] += 1 if angle < 0 else 0
                        line_total_counts[12] += 1
                        crossed_line = 13

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
                # print('绘制统计信息')
                # label = "客流总数: {}".format(str(total_track))

                # label = f"Line{line_index + 1}客流总数:{line_counters[line_index]}"
                # t_size = get_size_with_pil(label, int(25 * sr))
                #
                # x1, y1 = line_display_coords[line_index]  # Adjust coordinates based on line index
                #
                # color = compute_color_for_labels(2)
                # cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 30, y1 - t_size[1]), color, 1)
                # ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 235, 42), size=int(25*sr))

                # label = "穿过黄线人数: {} ({} 向上, {} 向下)".format(str(total_counter), str(up_count), str(down_count))
                label = f"穿过黄线{line_index + 1}人数:{line_total_counts[line_index]}({line_up_counts[line_index]}向上(第1、2象限),{line_down_counts[line_index]}向下(第3、4象限))"
                t_size = get_size_with_pil(label, int(25 * sr))
                x1, y1 = line_display_coords[line_index]
                y1 = y1 + int(50*sr)
                color = compute_color_for_labels(2)
                cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 200, y1 - t_size[1]), color, 1)
                ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 235, 42), size=int(25*sr))

            if last_track_id >= 0:
                label = f"最新:行人{last_track_id}号{'向上' if angle >= 0 else '向下'}穿过黄线{crossed_line}"
                t_size = get_size_with_pil(label, int(25 * sr))
                # x1 = 20
                # y1 = 150
                x1, y1 = line_display_coords[line_index]
                y1 = y1 + int(100*sr)
                color = compute_color_for_labels(2)
                cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 200, y1 - t_size[1]), (0, 0, 0, 128), -1)
                ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 235, 42), size=int(25*sr))

            end = time_synchronized()

            if self.args.display:   # CES
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.logger.info("{}/time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(idx_frame, end - t1, 1 / (end - t1),
                                     bbox_xywh.shape[0], len(outputs)))

            # Write the data to the CSV file
            row_data = ([idx_frame, total_track], [line_up_counts[1], line_down_counts[1], line_total_counts[1]], \
                [line_up_counts[2], line_down_counts[2], line_total_counts[2]], \
                [line_up_counts[3], line_down_counts[3], line_total_counts[3]])
            self.csv_writer.writerow(row_data)

            # Save the processed frame to the output video
            self.output_video.write(ori_img)

        self.csv_file.close()
        self.output_video.release()


def parse_args():
    parser = argparse.ArgumentParser()
    path = r"E:\iCloud照片\2023\videos\stationPedCount\5_20131110_new\New folder_hori\VID_20231105_184538.mp4" # CES 文件路径
    # parser.add_argument("--video_path", default='./MOT16-03.mp4', type=str)
    parser.add_argument("--video_path", default=path, type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
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


if __name__ == '__main__':
    args = parse_args()
    # print(args.use_cuda)
    # print(torch.cuda.is_available())
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    # Record the time
    start_time = time.time()

    yolo_reid = yolo_reid(cfg, args, path=args.video_path)
    with torch.no_grad():
        yolo_reid.deep_sort()

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
