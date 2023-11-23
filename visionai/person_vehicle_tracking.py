#!/usr/bin/env python3
import cv2
import threading
import queue
import time
import os
import sys

from vision_ai_common import proto_parser, get_boxes, convert_norm_bbox, BB_TIMEOUT, ANNOTATION_THICKNESS, FONT_SCALE, LABELS_TO_COLORS
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

class ByteTrackArgument:
    track_thresh = 0.30 # High_threshold
    track_buffer = 75 # Number of frame lost tracklets are kept
    match_thresh = 0.8 # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0 # Minimum bounding box aspect ratio
    min_box_area = 1.0 # Minimum bounding box area
    mot20 = False # If used, bounding boxes are not clipped.

def byte_track(targets, trackers):
    all_tlwhs = []
    all_ids = []
    all_classes = []
    all_scores = []
    for i, tracker in enumerate(trackers):
        if targets[i]:
            online_targets = tracker.update(np.array(targets[i]), [img_h, img_w], [img_h, img_w])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = [i] * len(online_targets)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    box = (tlwh[0], tlwh[1], tlwh[2], tlwh[3])

            all_tlwhs += online_tlwhs
            all_ids += online_ids
            all_classes += online_classes
            all_scores += online_scores


    return all_tlwhs, all_ids, all_classes, all_scores


if __name__ == "__main__":
    rtsp_url = sys.argv[1]
    shared_dir = sys.argv[2] # Vertex AI Vision receiver output dir

    #one tracker per class
    trackers = [BYTETracker(ByteTrackArgument) for _ in LABELS_TO_COLORS.keys()]

    q = queue.Queue()

    vcap = cv2.VideoCapture(rtsp_url)

    threading.Thread(target=proto_parser, args=(shared_dir, q)).start()

    g_bbs = [] #store predictions between consecutive updates
    check_time = time.time()
    labels = LABELS_TO_COLORS.keys()

    while(1):
        ret, frame = vcap.read()
        if ret == False:
            continue

        bbs = get_boxes()
        if bbs:
            g_bbs = bbs
            check_time = time.time()
        elif ((time.time() - check_time > BB_TIMEOUT)): # discard boxes if timeout reached
            g_bbs = []

        img_w = vcap.get(3)  # width
        img_h = vcap.get(4)  # height
        targets = [[] for _ in trackers]

        for bb in g_bbs:
            (xmin, ymin), (xmax, ymax) = convert_norm_bbox((img_w, img_h), bb.normalized_bounding_box)
            class_id = bb.entity.label_id - 1
            targets[class_id].append([xmin, ymin, xmax, ymax, bb.score])

        all_tlwhs, all_ids, all_classes, all_scores = byte_track(targets, trackers)
    
        for j, tlwh in enumerate(all_tlwhs):
            start_point = (int(tlwh[0]), int(tlwh[1]))
            end_point = (int(tlwh[0] + tlwh[2]),  int(tlwh[1] + tlwh[3]))
            class_id = all_classes[j] - 1
            cv2.putText(frame, labels[class_id] + " Id: " + str(all_ids[j]) + " {:.3f}".format(all_scores[j]), (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABELS_TO_COLORS.get(labels[class_id]), ANNOTATION_THICKNESS)
            cv2.rectangle(frame, start_point, end_point, LABELS_TO_COLORS.get(labels[class_id]), ANNOTATION_THICKNESS)

        cv2.imshow('Person/Vehicle detector & BYTEtrack', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
