import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import dlib
import albumentations as A
import config

def read_pts(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    landmarks = []

    for line in lines[3:-1]:
        x, y = map(float, line.strip().split())
        landmarks.append([x, y])
    
    return np.array(landmarks)

def get_files(root_folder):
    files = []
    for folder in root_folder:
        for ext in ["*.jpg", "*.png"]:
            for img_path in folder.glob(ext):
                pts_path = img_path.with_suffix(".pts")
                if pts_path.exists() and len(read_pts(pts_path)) == 68:
                    files.append((img_path, pts_path))
    return files  

class FaceLandmarksDataset(Dataset):
    def __init__(self, files, train=True):
        self.samples = []
        self.transform = get_transforms(train)
        self.samples = files            
        self.detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pts_path = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        rect = self._get_face_rect(image)
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

        landmarks = read_pts(str(pts_path))

        landmarks_x_max = landmarks[:, 0].max()
        landmarks_y_max = landmarks[:, 1].max()

        landmarks_x_min = landmarks[:, 0].min()
        landmarks_y_min = landmarks[:, 1].min()

        x1_face_rectangle = int(min(x1, landmarks_x_min))
        x2_face_rectangle = int(max(x2, landmarks_x_max))
        y1_face_rectangle = int(min(y1, landmarks_y_min))
        y2_face_rectangle = int(max(y2, landmarks_y_max))

        x1_expanded = max(0, x1_face_rectangle - config.CROP_EXPANSION)
        y1_expanded = max(0, y1_face_rectangle - config.CROP_EXPANSION)
        x2_expanded = min(image.shape[1], x2_face_rectangle + config.CROP_EXPANSION)
        y2_expanded = min(image.shape[0], y2_face_rectangle + config.CROP_EXPANSION)

        face_crop = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

        landmarks = landmarks - np.array([x1_expanded, y1_expanded])

        transformed = self.transform(image=face_crop)
        transformed_image = transformed['image']

        scale_x = (x2_expanded - x1_expanded) / config.IMAGE_SIZE[0]
        scale_y = (y2_expanded - y1_expanded) / config.IMAGE_SIZE[1]

        scale = torch.tensor([scale_x, scale_y], dtype=torch.float32)

        transformed_landmarks = torch.tensor(landmarks, dtype=torch.float32) / scale
        
        sample = {
            "image": torch.tensor(transformed_image, dtype=torch.float32).permute(2, 0, 1),
            "landmarks": transformed_landmarks,
            "face_rect": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
            "scale": scale
        }
        return sample
    
    def _get_face_rect(self, image):
        faces = self.detector(image)
        if len(faces) == 0:
            h, w = image.shape[:2]
            return dlib.rectangle(0, 0, w, h)
        return faces[0]

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.1),
            A.ToGray(p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    