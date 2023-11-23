import time
from google.protobuf import text_format
from proto.occ_pb2 import OccupancyCountingPredictionResult
from streaming_resources_pb2 import Packet
import queue
import os
import re

BB_TIMEOUT = 1.0 #discard bounding-boxes during playback after timeout
ANNOTATION_THICKNESS = 2
FONT_SCALE = 0.5

LABELS_TO_COLORS = {
    "Person": (36,255,12),
    "Vehicle": (255, 12, 117)
}

# Follow receiver output line by line as it's being written
def monitor(file):
    file.seek(0,2) # Go to the end of the file
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.05)
            continue
        yield line

def proto_parser(shared_dir, q):
    read_packet = False
    packet_str = ""
    print(f"{shared_dir}/" + os.readlink(f"{shared_dir}/receive_cat_app.INFO"))
    with open(f"{shared_dir}/" + os.readlink(f"{shared_dir}/receive_cat_app.INFO"), "r") as sf:
        sf.seek(0,2)
        x = monitor(sf)
        for line in x:
            if not line:
                continue
            # read only Packet header and payload
            if read_packet and not 'payload: "' in line:
                packet_str += line

            if "header" in line:
                read_packet = True
                packet_str = re.sub(r'.* header', 'header', line)

            if "payload" in line:
                read_packet = False
                packet_str += line
                packet = Packet()
                try:
                    text_format.Parse(packet_str, packet)
                    proto_mess = OccupancyCountingPredictionResult()
                    proto_mess.ParseFromString(packet.payload)
                    q.put(proto_mess)
                except:
                    # pass
                    print("Could not parse proto message")

def get_boxes(q):
    bbs = []
    if(q.empty() == False):
        mess = q.get(block=True)
        for bb in mess.identified_boxes:
            bbs.append(bb)
        q.task_done()
    return bbs

def convert_norm_bbox(size, bb):
    img_w  = size[0]
    img_h = size[1]
    start_point = (int(bb.xmin * img_w), int(bb.ymin * img_h))
    end_point = (int(start_point[0] + bb.width * img_w), int(start_point[1] + bb.height * img_h))
    return start_point, end_point

if __name__ == "__main__":
    print("Nothing to run")