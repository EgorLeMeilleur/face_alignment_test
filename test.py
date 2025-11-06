import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import config
from dataset import FaceLandmarksDataset, read_pts
from models import FaceAlignmentModel
import dlib
from PIL import Image
import csv

import torch
import torch.nn.functional as F

def heatmaps_to_coords(heatmaps, normalize=True):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    soft = F.softmax(flat, dim=-1).view(B, K, H, W)

    xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
    xs = xs.view(1, 1, 1, W)
    ys = ys.view(1, 1, H, 1)

    exp_x = (soft * xs).view(B, K, -1).sum(-1)
    exp_y = (soft * ys).view(B, K, -1).sum(-1)

    coords = torch.stack([exp_x, exp_y], dim=-1)

    if normalize:
        coords[..., 0] = coords[..., 0] / (W - 1)
        coords[..., 1] = coords[..., 1] / (H - 1)

    return coords


def map_resized_to_orig(coords_resized, crop_box, scale):
    x1, y1 = float(crop_box[0]), float(crop_box[1])
    sx, sy = float(scale[0]), float(scale[1])
    coords_orig = np.zeros_like(coords_resized, dtype=np.float32)
    coords_orig[:, 0] = coords_resized[:, 0] * sx + x1
    coords_orig[:, 1] = coords_resized[:, 1] * sy + y1
    return coords_orig

def map_hm_to_resized(peaks_hm, resized_size, hm_size):
    H_resized, W_resized = int(resized_size[0]), int(resized_size[1])
    H_hm, W_hm = int(hm_size[0]), int(hm_size[1])
    coords_resized = np.zeros_like(peaks_hm, dtype=np.float32)
    coords_resized[:, 0] = peaks_hm[:, 0] * (W_resized / float(W_hm))
    coords_resized[:, 1] = peaks_hm[:, 1] * (H_resized / float(H_hm))
    return coords_resized

def compute_nme_one(pred_points, gt_points, face_rect):
    dists = np.linalg.norm(pred_points - gt_points, axis=1)
    mean_px = float(np.mean(dists))
    H = float(face_rect[3] - face_rect[1])
    W = float(face_rect[2] - face_rect[0])
    norm = np.sqrt(max(1.0, H * W))
    return mean_px / norm

def ced_and_auc(nmes, T=config.MAX_ERROR_THRESHOLD, num=200):
    thr = np.linspace(0.0, T, num)
    ced = np.array([np.mean(nmes < t) for t in thr])
    auc_raw = np.trapz(ced, thr)
    auc_norm = auc_raw / T
    return thr, ced, auc_raw, auc_norm

def count_ced(predicted_points, gt_points, normalizations):
    ceds = []
    for preds, gts, normalization in zip(predicted_points, gt_points, normalizations):
        x_pred, y_pred = preds[:, ::2], preds[:, 1::2]
        x_gt, y_gt = gts[:, ::2], gts[:, 1::2]
        n_points = x_pred.shape[0]

        diff_x = [x_gt[i] - x_pred[i] for i in range(n_points)]
        diff_y = [y_gt[i] - y_pred[i] for i in range(n_points)]
        dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
        avg_norm_dist = np.sum(dist) / (n_points * normalization)
        ceds.append(avg_norm_dist)
    ceds = np.sort(ceds)

    return ceds

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_gts = []
    all_nms = []
    for batch in loader:
        imgs = batch["image"].to(device)
        B = imgs.shape[0]
        out = model(imgs)
        for i in range(B):
            face_rect = batch["face_rect"][i].cpu().numpy()
            crop_box = batch.get("crop_box", None)
            scale = batch.get("scale", None)
            if crop_box is None or scale is None:
                resized_h, resized_w = imgs.shape[2], imgs.shape[3]
                crop_box = np.array([0, 0, resized_w - 1, resized_h - 1], dtype=np.float32)
                scale = np.array([1.0, 1.0], dtype=np.float32)
            else:
                crop_box = crop_box[i].cpu().numpy()
                scale = scale[i].cpu().numpy()

            resized_h, resized_w = imgs.shape[2], imgs.shape[3]
            kps_norm = batch["keypoints_norm"][i].cpu().numpy()
            kps_px = np.zeros_like(kps_norm)
            kps_px[:,0] = kps_norm[:,0] * float(resized_w)
            kps_px[:,1] = kps_norm[:,1] * float(resized_h)
            gt_orig = map_resized_to_orig(kps_px, crop_box, scale)

            if out.dim() == 3:
                pred_i = out[i].detach().cpu().numpy()
                pred_px = np.zeros_like(pred_i)
                pred_px[:,0] = pred_i[:,0] * float(resized_w)
                pred_px[:,1] = pred_i[:,1] * float(resized_h)
                pred_orig = map_resized_to_orig(pred_px, crop_box, scale)

            elif out.dim() == 4:
                hm = torch.nn.functional.sigmoid(out[i]).detach().cpu().numpy()
                peaks_hm = heatmaps_to_coords(hm)
                coords_resized = map_hm_to_resized(peaks_hm, (resized_h, resized_w), (hm.shape[1], hm.shape[2]))
                pred_orig = map_resized_to_orig(coords_resized, crop_box, scale)

            all_preds.append(pred_orig)
            all_gts.append(gt_orig)
            Hf = float(face_rect[3] - face_rect[1])
            Wf = float(face_rect[2] - face_rect[0])
            all_nms.append(np.sqrt(max(1.0, Hf * Wf)))

    return np.array(all_preds), np.array(all_gts), np.array(all_nms)

def evaluate_dlib(predictor, files, detector):
    preds = []
    gts = []
    norms = []
    for img_path, pts_path in files:
        pil = Image.open(img_path).convert("RGB")
        img_np = np.array(pil)
        faces = detector(img_np, 1)
        if len(faces) == 0:
            h, w = img_np.shape[:2]
            face_rect = dlib.rectangle(0, 0, w - 1, h - 1)
        else:
            face_rect = faces[0]
        shape = predictor(img_np, face_rect)
        pred_points = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        true_points = read_pts(str(pts_path))
        H = float(face_rect.bottom() - face_rect.top())
        W = float(face_rect.right() - face_rect.left())
        norm = np.sqrt(max(1.0, H * W))
        preds.append(pred_points)
        gts.append(true_points)
        norms.append(norm)
    return np.array(preds), np.array(gts), np.array(norms)

def test(experiment_name, model_type, loss_type, head_type):
    checkpoint = Path(config.CHECKPOINT_DIR) / f"{experiment_name}.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = Path(config.RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    log_file =  results_path / f"auc_results_{experiment_name}.txt"

    model = FaceAlignmentModel.load_from_checkpoint(
        str(checkpoint), map_location=device,
        model_type=model_type, loss_type=loss_type, head_type=head_type
    )
    model.eval()

    thresholds = np.linspace(0, 1, 100)
    plot_thresholds = np.linspace(0, config.MAX_ERROR_THRESHOLD, 100)

    for ds_name, folder in config.TEST_FOLDERS.items():
        print("Evaluating dataset:", ds_name)
        files = []
        folder = Path(folder)
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in folder.glob(ext):
                pts_path = img_path.with_suffix(".pts")
                if len(read_pts(str(pts_path))) == config.NUM_POINTS:
                    files.append((img_path, pts_path))

        dataset = FaceLandmarksDataset(files, train=False)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

        preds, gts, norms = evaluate_model(model, loader, device)

        ceds = count_ced(preds, gts, norms)
        ced_curve = np.array([np.mean(ceds < thr) for thr in thresholds])
        auc_model = np.trapezoid(ced_curve, thresholds)

        plt.figure()
        plt.plot(plot_thresholds, ced_curve, label=f'{experiment_name} (AUC: {auc_model:.4f})')

        if ds_name.lower() == "menpo":
            predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()
            preds, gt, normalizations = evaluate_dlib(predictor, files, detector)
            ceds = count_ced(preds, gt, normalizations)
            ced_curve = np.array([np.mean(ceds < thr) for thr in thresholds])
            auc_dlib = np.trapezoid(ced_curve, thresholds)
            plt.plot(plot_thresholds, ced_curve, label=f'dlib (AUC: {auc_dlib:.4f})')
            print(f"{ds_name} dataset - dlib AUC: {auc_dlib:.4f}")

        plt.xlabel('Normalized Mean Square Error')
        plt.ylabel('Percentage of Images')
        plt.title(f'CED Curve on {ds_name} dataset')
        plt.legend()
        
        plot_file = results_path / f"CED_{ds_name}_{experiment_name}.png"
        plt.savefig(str(plot_file))
        plt.close()

        with open(log_file, "a") as f:
            f.write(f"{ds_name}, AUC: {auc_model:.4f}\n")
        
        print(f"{ds_name} dataset - Model AUC: {auc_model:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=config.MODEL_TYPE)
    parser.add_argument("--loss_type", type=str, default=config.LOSS_TYPE)
    parser.add_argument("--head_type", type=str, default=config.HEAD_TYPE)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    test(args.experiment_name, args.model_type, args.loss_type, args.head_type)


#  try:
#         model = FaceAlignmentModel.load_from_checkpoint(
#             str(checkpoint_path), map_location='cpu',
#             model_type=model_type, loss_type=loss_type, head_type=head_type
#         )
#         print("Loaded model via Lightning.load_from_checkpoint(...) with provided args.")
#         return model

#     except Exception as e:
#         print("load_from_checkpoint with args failed (fallback to manual loading).")
#         print("Reason:", e)

#     # Попытка 2: ручная загрузка + фильтрация state_dict
#     ckpt = torch.load(str(checkpoint_path), map_location='cpu')

#     model = FaceAlignmentModel(model_type=model_type, loss_type=loss_type, head_type=head_type)

#     load_res = model.load_state_dict(filtered, strict=False)