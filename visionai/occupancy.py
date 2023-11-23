#!/usr/bin/env python3
import cv2
# from proto.occ_pb2 import OccupancyCountingPredictionResult
import threading
import queue
import time
import re
import sys
from vision_ai_common import proto_parser, get_boxes, convert_norm_bbox, BB_TIMEOUT, ANNOTATION_THICKNESS, FONT_SCALE, LABELS_TO_COLORS

if __name__ == "__main__":
    rtsp_url = sys.argv[1]
    shared_dir = sys.argv[2] # Vertex AI Vision receiver output dir

    q = queue.Queue() # predictions

    vcap = cv2.VideoCapture(rtsp_url)

    threading.Thread(target=proto_parser, args=(shared_dir, q)).start()

    g_bbs = [] #store predictions between consecutive updates
    check_time = time.time()

    while(1):
        ret, frame = vcap.read()
        if ret == False:
            continue

        bbs = get_boxes(q)
        if bbs:
            g_bbs = bbs
            print(bbs)
            check_time = time.time()
        elif ((time.time() - check_time > BB_TIMEOUT)): # discard boxes if timeout reached
            g_bbs = []

        img_w  = vcap.get(3) # width
        img_h = vcap.get(4)  # height
        for bb in g_bbs:
            start_point, end_point = convert_norm_bbox((img_w, img_h), bb.normalized_bounding_box)
            cv2.putText(frame, bb.entity.label_string + " Id: " + str(bb.track_id) + " {:.3f}".format(bb.score), (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABELS_TO_COLORS.get(bb.entity.label_string), ANNOTATION_THICKNESS)
            cv2.rectangle(frame, start_point, end_point, LABELS_TO_COLORS.get(bb.entity.label_string), ANNOTATION_THICKNESS)
    
        cv2.imshow('Occupancy analytics', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()