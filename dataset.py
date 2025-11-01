import numpy as np
import torch
from torch.utils.data import Dataset
import dlib
import albumentations as A
import config
from PIL import Image

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

def make_heatmaps(keypoints_px, heatmap_size, image_size, sigma=2.0):
    K = keypoints_px.shape[0]
    H_hm, W_hm = heatmap_size
    H_img, W_img = image_size
    hm = np.zeros((K, H_hm, W_hm), dtype=np.float32)

    sx = float(W_hm) / float(W_img)
    sy = float(H_hm) / float(H_img)

    tmp_size = int(3 * sigma)
    size = 2 * tmp_size + 1
    xg = np.arange(0, size, 1, np.float32)
    yg = xg[:, None]
    gaussian = np.exp(-((xg - tmp_size) ** 2 + (yg - tmp_size) ** 2) / (2 * sigma * sigma)).astype(np.float32)

    for i, (x, y) in enumerate(keypoints_px):
        if x < 0 or y < 0 or np.isnan(x) or np.isnan(y):
            continue
        mu_x = int(x * sx + 0.5)
        mu_y = int(y * sy + 0.5)
        ul = [mu_x - tmp_size, mu_y - tmp_size]
        br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
        if ul[0] >= W_hm or ul[1] >= H_hm or br[0] < 0 or br[1] < 0:
            continue
        g_x = max(0, -ul[0]), min(br[0], W_hm) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], H_hm) - ul[1]
        img_x = max(0, ul[0]), min(br[0], W_hm)
        img_y = max(0, ul[1]), min(br[1], H_hm)
        hm[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            hm[i, img_y[0]:img_y[1], img_x[0]:img_x[1]],
            gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
    return hm

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.1),
            A.ToGray(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=config.MEAN, std=config.STD)])
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=config.MEAN, std=config.STD)])

class FaceLandmarksDataset(Dataset):
    def __init__(self, files, train=True):
        self.samples = []
        self.transform = get_transforms(train)
        self.samples = files            
        self.detector = dlib.get_frontal_face_detector()
        self.img_h = config.IMAGE_SIZE
        self.img_w = config.IMAGE_SIZE
        self.hm_h = config.HEATMAP_SIZE
        self.hm_w = config.HEATMAP_SIZE
        self.sigma = config.SIGMA

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pts_path = self.samples[idx]
        image = Image.open(img_path)
        
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

        x_expansion = config.CROP_EXPANSION * image.shape[1]
        y_expansion = config.CROP_EXPANSION * image.shape[0]

        x1_expanded = max(0, x1_face_rectangle - x_expansion)
        y1_expanded = max(0, y1_face_rectangle - y_expansion)
        x2_expanded = min(image.shape[1], x2_face_rectangle + x_expansion)
        y2_expanded = min(image.shape[0], y2_face_rectangle + y_expansion)

        face_crop = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

        landmarks_crop = landmarks - np.array([x1_expanded, y1_expanded])

        transformed = self.transform(image=face_crop, keypoints=landmarks_crop)
        transformed_image = transformed['image']
        transformed_landmarks = transformed['keypoints']

        normalized_landmarks = transformed_landmarks / np.array([image.size[1], image.size[0]])
        heatmaps = make_heatmaps(transformed_landmarks, heatmap_size=(self.hm_h, self.hm_w), image_size=(self.img_h, self.img_w), sigma=self.sigma)
        
        img_tensor = torch.tensor(transformed_image, dtype=torch.float32).permute(2, 0, 1)
        kps_px_tensor = torch.tensor(transformed_landmarks, dtype=torch.float32)
        kps_norm_tensor = torch.tensor(normalized_landmarks, dtype=torch.float32)
        heatmap_tensor = torch.tensor(heatmaps, dtype=torch.float32) if heatmaps is not None else None

        sample = {
            "image": img_tensor,
            "keypoints_px": kps_px_tensor,
            "keypoints_norm": kps_norm_tensor,
            "heatmaps": heatmap_tensor,
            "crop_box": torch.tensor([x1_expanded, y1_expanded, x2_expanded, y2_expanded], dtype=torch.int32),
            "scale_unnorm": torch.tensor(([image.size[1], image.size[0]]), dtype=torch.float32),
            "orig_size": torch.tensor(image.size, dtype=torch.int32)
        }

        return sample
    
    def _get_face_rect(self, image):
        faces = self.detector(image)
        if len(faces) == 0:
            h, w = image.shape[:2]
            return dlib.rectangle(0, 0, w, h)
        return faces[0]
    