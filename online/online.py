import base64
import cv2
import time
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import threading
import queue
import sys

BB_TIMEOUT = 1.0 #discard bounding-boxes during playback after timeout
ANNOTATION_THICKNESS = 2
FONT_SCALE = 0.5
VERTEX_AI_DETECT_MAX_W_H = 1024
PLAYBACK_DELAY = 0.5

LABELS_TO_COLORS  = {
    "pedestrian": (205, 92, 92),
    "people": (255, 160, 122),
    "bicycle": (255, 215, 0),
    "car": (255, 105, 180),
    "van": (230, 230, 250),
    "truck": (173, 255, 47),
    "tricycle": (0, 139, 139),
    "awning-tricycle" :(0, 255, 255),
    "bus": (255, 0, 255),
    "motor":(65, 105, 225)
}

# The max width/height for AutoML Object Detection models is 1024px
def preprocess_image(im, max_width, max_height):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    [height, width, _] = im.shape

    if height > max_height or width > max_width:
        ratio = max(height / float(max_width), width / float(max_height))
        new_height = int(height / ratio + 0.5)
        new_width = int(width / ratio + 0.5)
        resized_im = cv2.resize(
            im, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        _, processed_image = cv2.imencode(".jpg", resized_im, encode_param)
    else:
        _, processed_image = cv2.imencode(".jpg", im, encode_param)

    return base64.b64encode(processed_image).decode("utf-8")

def get_preds():
    preds = []
    while(q.empty() == False):
        pred = q.get(block=True)
        preds.append(dict(pred))
        q.task_done()
    return preds

# Get frame, send prediction request and push prediction to queue
def worker(q, project, endpoint_id, location, api_endpoint, conf_threshold, max_predictions):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    parameters = predict.params.ImageObjectDetectionPredictionParams(
        confidence_threshold=conf_threshold,
        max_predictions=max_predictions,
    ).to_value()
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)

    while True:
        if worker_q.empty() == False:
            id, img = worker_q.get(block=True)
            worker_q.task_done()

            instance = predict.instance.ImageObjectDetectionPredictionInstance(
                content=img,
            ).to_value()
        
            instances = [instance]
            response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
            predictions = response.predictions
            for prediction in predictions:
                q.put((id, dict(prediction)))
        else:
            time.sleep(0.01)

# Captures video stream, publishes frames to worker and playback queues
def publisher(rtsp_url, predict_every_x_frame):
    # v = cv2.VideoCapture(r"/Users/jakubbranicki/Downloads/VisDrone2019-VID/sequences/uav0000117_02622_v/%07d.jpg")
    v = cv2.VideoCapture(rtsp_url)
    print("Publisher thread started")

    frame_id = 0
    while(1):
        ret, frame = v.read()
        if ret != False:
            if frame_id % predict_every_x_frame == 0:
                worker_q.put((frame_id, preprocess_image(frame, VERTEX_AI_DETECT_MAX_W_H, VERTEX_AI_DETECT_MAX_W_H)))
            playback_q.put((frame_id, frame))
            frame_id += 1
            # time.sleep(0.005)

def get_preds(q):
    preds = None
    if q.empty() == False:
        preds = q.get(block=True)
        q.task_done()
    return preds

def convert_norm_bbox(size, bb):
    ratio=1.0
    img_w, img_h = size

    # Resize bbs if needed
    if img_h > VERTEX_AI_DETECT_MAX_W_H or img_w > VERTEX_AI_DETECT_MAX_W_H:
        ratio = max(img_h / float(VERTEX_AI_DETECT_MAX_W_H), img_w / float(VERTEX_AI_DETECT_MAX_W_H))
        img_h = int(img_h / ratio + 0.5)
        img_w = int(img_w / ratio + 0.5)

    start_point = (int(bb[0] * img_w * ratio), int(bb[2] * img_h * ratio))
    end_point = (int(bb[1] * img_w * ratio), int(bb[3] * img_h * ratio))
    return start_point, end_point

if __name__ == "__main__":
    project = sys.argv[1]
    endpoint_id = sys.argv[2]
    location = sys.argv[3]
    api_endpoint = sys.argv[4]
    rtsp_url = sys.argv[5]
    worker_count = int(sys.argv[6])
    predict_every_x_frame = int(sys.argv[7])
    conf_threshold = float(sys.argv[8])
    
    max_predictions = 20

    worker_q = queue.Queue()   # out for prediction
    playback_q = queue.Queue() # stream buffer

    output_q = queue.PriorityQueue() # Priority queue automatically handles sorting predictions by frame_id, which ensures ascending order for synchornization

    for _ in range(worker_count):
        threading.Thread(target=worker, args=(worker_q, project, endpoint_id, location, api_endpoint, conf_threshold, max_predictions)).start()

    threading.Thread(target=publisher, args=(rtsp_url, predict_every_x_frame)).start()

    # Postpone playback after stream arrives
    while(playback_q.qsize() == 0):
        time.sleep(0.01)
    time.sleep(PLAYBACK_DELAY)
    print("Starting playback loop")

    g_bbs = dict() # store predictions between consecutive updates

    frame_id = 0
    frame = None

    check_time = time.time()

    while(1):
        if playback_q.empty():
            time.sleep(0.01)
            continue

        frame_id, frame = playback_q.get(block=True)
        playback_q.task_done()    

        preds = get_preds(output_q)
        
        # drop stale predictions
        while preds is not None and preds[0] < frame_id:
            preds = get_preds(output_q)

        # synchronise frame with prediction by id
        if preds:
            id, pred = preds
            if id == frame_id:
                g_bbs = pred
                check_time = time.time()
            elif id > frame_id: # put again if fetched too early
                output_q.put(preds)

        if (time.time() - check_time > BB_TIMEOUT): # discard boxes if timeout reached
            g_bbs = dict()

        [height, width, _] = frame.shape

        if 'bboxes' in g_bbs.keys():
            for i, bb in enumerate(g_bbs["bboxes"]):
                start_point, end_point = convert_norm_bbox((width, height), bb)
                label = g_bbs["displayNames"][i]
                cv2.putText(frame, label + " {:.3f}".format(g_bbs["confidences"][i]), (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LABELS_TO_COLORS.get(label), ANNOTATION_THICKNESS)
                cv2.rectangle(frame, start_point, end_point, LABELS_TO_COLORS.get(label), ANNOTATION_THICKNESS)

        cv2.imshow('online visdrone', frame)

        cv2.waitKey(40) # adjust frame rate

    cv2.destroyAllWindows()
