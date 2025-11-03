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
            A.Normalize(mean=config.MEAN, std=config.STD)],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )
    else:
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=config.MEAN, std=config.STD)],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

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
        self.crop_expansion = config.CROP_EXPANSION

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pts_path = self.samples[idx]
        pil_image = Image.open(img_path).convert('RGB')
        image = np.array(pil_image)  # H_orig, W_orig, C
        H_orig, W_orig = image.shape[:2]

        # read GT
        landmarks = read_pts(str(pts_path))  # (K,2)

        # dlib face rect expects numpy RGB
        faces = self.detector(image, 1)
        if len(faces) == 0:
            face_rect = dlib.rectangle(0, 0, W_orig - 1, H_orig - 1)
        else:
            face_rect = faces[0]
        x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

        # expand to include annotated points
        x1_face = int(min(x1, float(landmarks[:,0].min())))
        x2_face = int(max(x2, float(landmarks[:,0].max())))
        y1_face = int(min(y1, float(landmarks[:,1].min())))
        y2_face = int(max(y2, float(landmarks[:,1].max())))

        # expansion either fraction of box or px
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

        # crop inclusive [y1:y2+1, x1:x2+1]
        face_crop = image[y1_exp:y2_exp+1, x1_exp:x2_exp+1].copy()
        crop_h, crop_w = face_crop.shape[:2]

        # shift landmarks to crop coords
        landmarks_crop = landmarks - np.array([x1_exp, y1_exp], dtype=np.float32)

        # prepare keypoints list for albumentations
        keypoints_list = [ (float(x), float(y)) for x,y in landmarks_crop ]

        augmented = self.transform(image=face_crop, keypoints=keypoints_list)
        transformed_image = augmented['image']  # H_resized, W_resized, C (float32 if Normalize)
        transformed_kps = np.array(augmented['keypoints'], dtype=np.float32)  # (K,2) in resized coords
        resized_h, resized_w = transformed_image.shape[:2]

        # compute scale from resized -> crop (pixels)
        # scale_x such that x_resized * scale_x = x_in_crop_pixels
        scale_x = float(crop_w) / float(resized_w) if resized_w > 0 else 1.0
        scale_y = float(crop_h) / float(resized_h) if resized_h > 0 else 1.0

        # normalized keypoints relative to resized image in [0,1]
        kps_norm = transformed_kps.copy()
        mask_vis = (kps_norm[:,0] >= 0) & (kps_norm[:,1] >= 0)
        if resized_w > 0 and resized_h > 0:
            kps_norm[mask_vis, 0] = kps_norm[mask_vis, 0] / float(resized_w)
            kps_norm[mask_vis, 1] = kps_norm[mask_vis, 1] / float(resized_h)
        kps_norm[~mask_vis] = -1.0

        # generate heatmaps using resized image size (important!)
        heatmaps = make_heatmaps(transformed_kps, heatmap_size=(self.hm_h, self.hm_w), image_size=(resized_h, resized_w), sigma=self.sigma)

        # tensors
        img_tensor = torch.tensor(transformed_image, dtype=torch.float32).permute(2,0,1)
        kps_px_tensor = torch.tensor(transformed_kps, dtype=torch.float32)
        kps_norm_tensor = torch.tensor(kps_norm, dtype=torch.float32)
        heatmap_tensor = torch.tensor(heatmaps, dtype=torch.float32)

        sample = {
            "image": img_tensor,                                # C,H_resized,W_resized
            "keypoints_px": kps_px_tensor,                      # K,2 in resized px (-1 if invisible)
            "keypoints_norm": kps_norm_tensor,                  # K,2 in [0,1] w.r.t resized
            "heatmaps": heatmap_tensor,                         # K,H_hm,W_hm
            "crop_box": torch.tensor([x1_exp, y1_exp, x2_exp, y2_exp], dtype=torch.int32),
            "scale": torch.tensor([scale_x, scale_y], dtype=torch.float32),
            "face_rect": torch.tensor([x1, y1, x2, y2], dtype=torch.int32),
            "orig_size": torch.tensor([H_orig, W_orig], dtype=torch.int32)
        }

        return sample
    
    def _get_face_rect(self, image):
        faces = self.detector(image)
        if len(faces) == 0:
            h, w = image.shape[:2]
            return dlib.rectangle(0, 0, w, h)
        return faces[0]
    