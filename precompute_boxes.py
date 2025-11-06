# precompute_boxes.py
from pathlib import Path
import json
import cv2
import dlib
from dataset import get_files
import tqdm

detector = dlib.get_frontal_face_detector()
files = get_files([Path("data/300W/train"), Path("data/Menpo/train")])

out = {}
for img_path, pts_path in tqdm.tqdm(files):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img, 1)
    if len(faces) == 0:
        h,w = img.shape[:2]
        box = [0,0,w-1,h-1]
    else:
        r = faces[0]
        box = [int(r.left()), int(r.top()), int(r.right()), int(r.bottom())]
    out[str(img_path)] = box

with open("precomputed_face_boxes.json", "w") as f:
    json.dump(out, f)