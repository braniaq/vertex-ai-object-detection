import os 
from PIL import Image
import sys

dataset_path = sys.argv[1] # VisDrone2019-DET-* dataset
bucket_url = sys.argv[2] # dataset url in GCP bucket
csv_path = sys.argv[3]

labels = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]

# Convert VisDrone box to normalized x1,y1;x2,y1;x2,y2;x1,y2 coords
def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    return box[0] * dw, box[1] * dh, (box[0] + box[2]) * dw, box[1] * dh, (box[0] + box[2]) * dw, (box[1] + box[3]) * dh, box[0] * dw, (box[1] + box[3]) * dh


l = os.listdir(f"{dataset_path}/annotations")
for sample in l:
    lines = []
    img_size = Image.open(f"{dataset_path}/images/" + sample.replace("txt", "jpg")).size

    with open(f"{dataset_path}/annotations/{f}", 'r') as file:  # read annotation.txt
        for row in [x.split(',') for x in file.read().strip().splitlines()]:
            if row[4] == '0':  # VisDrone 'ignored regions' class 0
                continue
            class_index = int(row[5]) - 1
            
            box = convert_box(img_size, tuple(map(int, row[:4])))
            img = sample.replace("txt", "jpg")
            # Vertex AI csv import file format: img_path,label,bbox
            lines.append(f"{bucket_url}/{img},{labels[class_index]},{','.join([f'{x:.6f}' for x in box])}\n")

    with open(f"{csv_path}", 'a') as csv:
        csv.writelines(lines)