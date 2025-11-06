# dataset_fixed.py
import numpy as np
import torch
from torch.utils.data import Dataset
import dlib
import albumentations as A
import config
from PIL import Image
import json

def read_pts(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    landmarks = []
    for line in lines[3:-1]:
        x, y = map(float, line.strip().split())
        landmarks.append([x, y])
    return np.array(landmarks, dtype=np.float32)

def get_files(root_folder):
    files = []
    for folder in root_folder:
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.jpg", "*.JPG", "*.JPEG", "*.PNG"]:
            for img_path in folder.glob(ext):
                pts_path = img_path.with_suffix(".pts")
                if pts_path.exists() and len(read_pts(pts_path)) == config.NUM_POINTS:
                    files.append((img_path, pts_path))
    return files

def make_heatmaps(kps, hm_h, hm_w, img_h, img_w, sigma=2.0):
    K = kps.shape[0]
    sx = float(hm_w) / float(img_w)
    sy = float(hm_h) / float(img_h)
    mu_x = kps[:, 0] * sx
    mu_y = kps[:, 1] * sy

    tmp = int(3 * sigma)
    size = 2 * tmp + 1
    x = np.arange(0, size, 1, np.float32) - tmp
    y = x[:, None]
    g = np.exp(-(x**2 + y**2) / (2 * sigma * sigma)).astype(np.float32)

    hm = np.zeros((K, hm_h, hm_w), dtype=np.float32)

    for i in range(K):
        x0 = mu_x[i]
        y0 = mu_y[i]
        if np.isnan(x0) or np.isnan(y0):
            continue

        ulx = int(np.floor(x0)) - tmp
        uly = int(np.floor(y0)) - tmp
        brx = ulx + size
        bry = uly + size

        gx0 = 0
        gy0 = 0
        gx1 = size
        gy1 = size

        if ulx < 0:
            gx0 = -ulx
            ulx = 0
        if uly < 0:
            gy0 = -uly
            uly = 0
        if brx > hm_w:
            gx1 = size - (brx - hm_w)
            brx = hm_w
        if bry > hm_h:
            gy1 = size - (bry - hm_h)
            bry = hm_h

        if gx1 <= gx0 or gy1 <= gy0:
            continue

        patch = g[gy0:gy1, gx0:gx1]
        patch = patch / patch.max()
        hm[i, uly:bry, ulx:brx] = np.maximum(hm[i, uly:bry, ulx:brx], patch)

    return hm

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.1),
            A.ToGray(p=0.1),
            A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-15,15), p=0.3),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=config.MEAN, std=config.STD)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=config.MEAN, std=config.STD)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

class FaceLandmarksDataset(Dataset):
    def __init__(self, files, train=True):
        self.samples = files
        self.transform = get_transforms(train)
        self.img_h = config.IMAGE_SIZE if isinstance(config.IMAGE_SIZE, int) else config.IMAGE_SIZE[0]
        self.img_w = config.IMAGE_SIZE if isinstance(config.IMAGE_SIZE, int) else config.IMAGE_SIZE[1]
        self.hm_h = config.HEATMAP_SIZE
        self.hm_w = config.HEATMAP_SIZE
        self.sigma = config.SIGMA
        self.crop_expansion = config.CROP_EXPANSION
        self.precompute = None
        if getattr(config, "PRECOMPUTE", None):
            # PRECOMPUTE expected to be path to json
            with open(config.PRECOMPUTE, 'r') as f:
                self.precompute = json.load(f)
        else:
            self.detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pts_path = self.samples[idx]
        pil_image = Image.open(img_path).convert('RGB')
        image = np.array(pil_image)
        H_orig, W_orig = image.shape[:2]
        landmarks = read_pts(str(pts_path))

        if self.precompute:
            key = str(img_path)
            if key not in self.precompute:
                x1, y1, x2, y2 = 0, 0, W_orig-1, H_orig-1
            else:
                box = self.precompute[key]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        else:
            faces = self.detector(image, 0)
            if len(faces) == 0:
                x1, y1, x2, y2 = 0, 0, W_orig-1, H_orig-1
            else:
                r = faces[0]
                x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()

        x1_face = int(min(x1, float(landmarks[:,0].min())))
        x2_face = int(max(x2, float(landmarks[:,0].max())))
        y1_face = int(min(y1, float(landmarks[:,1].min())))
        y2_face = int(max(y2, float(landmarks[:,1].max())))

        if self.crop_expansion < 1.0:
            x_exp = int(self.crop_expansion * (x2_face - x1_face + 1))
            y_exp = int(self.crop_expansion * (y2_face - y1_face + 1))
        else:
            x_exp = int(self.crop_expansion)
            y_exp = int(self.crop_expansion)

        x1_exp = max(0, x1_face - x_exp)
        y1_exp = max(0, y1_face - y_exp)
        x2_exp = min(W_orig - 1, x2_face + x_exp)
        y2_exp = min(H_orig - 1, y2_face + y_exp)

        face_crop = image[y1_exp:y2_exp+1, x1_exp:x2_exp+1].copy()
        crop_h, crop_w = face_crop.shape[:2]
        landmarks_crop = landmarks - np.array([x1_exp, y1_exp], dtype=np.float32)

        keypoints_list = [(float(x), float(y)) for x,y in landmarks_crop]
        augmented = self.transform(image=face_crop, keypoints=keypoints_list)
        transformed_image = augmented['image']
        transformed_kps = np.array(augmented['keypoints'], dtype=np.float32)
        resized_h, resized_w = transformed_image.shape[:2]

        scale_x = float(crop_w) / float(resized_w) if resized_w > 0 else 1.0
        scale_y = float(crop_h) / float(resized_h) if resized_h > 0 else 1.0

        kps_norm = transformed_kps.copy()
        mask_vis = (kps_norm[:,0] >= 0) & (kps_norm[:,1] >= 0)
        if resized_w > 0 and resized_h > 0:
            kps_norm[mask_vis, 0] = kps_norm[mask_vis, 0] / float(resized_w)
            kps_norm[mask_vis, 1] = kps_norm[mask_vis, 1] / float(resized_h)
        kps_norm[~mask_vis] = -1.0

        heatmaps = make_heatmaps(transformed_kps, self.hm_h, self.hm_w, resized_h, resized_w, self.sigma)

        img_tensor = torch.from_numpy(transformed_image.astype(np.float32)).permute(2,0,1)
        kps_px_tensor = torch.from_numpy(transformed_kps)
        kps_norm_tensor = torch.from_numpy(kps_norm)
        heatmap_tensor = torch.from_numpy(heatmaps)

        sample = {
            "image": img_tensor,
            "keypoints_px": kps_px_tensor,
            "keypoints_norm": kps_norm_tensor,
            "heatmaps": heatmap_tensor,
            "crop_box": torch.tensor([x1_exp, y1_exp, x2_exp, y2_exp], dtype=torch.int32),
            "scale": torch.tensor([scale_x, scale_y], dtype=torch.float32),
            "face_rect": torch.tensor([x1, y1, x2, y2], dtype=torch.int32),
            "orig_size": torch.tensor([H_orig, W_orig], dtype=torch.int32)
        }
        return sample