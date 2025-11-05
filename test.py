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

def heatmaps_to_peaks(hm_numpy):
    if hm_numpy.ndim == 4:
        B = hm_numpy.shape[0]
        res = []
        for b in range(B):
            res.append(heatmaps_to_peaks(hm_numpy[b]))
        return res
    K, Hh, Wh = hm_numpy.shape
    peaks = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        idx = hm_numpy[k].argmax()
        y = idx // Wh
        x = idx % Wh
        peaks[k, 0] = float(x)
        peaks[k, 1] = float(y)
    return peaks

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
                hm = torch.nn.funnctional.sigmoid(out[i]).detach().cpu().numpy()
                peaks_hm = heatmaps_to_peaks(hm)
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

def test(checkpoint, experiment_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = Path(config.RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)

    model = FaceAlignmentModel.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    for ds_name, folder in config.TEST_FOLDERS.items():
        print("Evaluating dataset:", ds_name)
        files = []
        folder = Path(folder)
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in folder.glob(ext):
                pts_path = img_path.with_suffix(".pts")
                if pts_path.exists():
                    try:
                        pts = read_pts(str(pts_path))
                        if pts.shape[0] == config.NUM_POINTS:
                            files.append((img_path, pts_path))
                    except Exception:
                        continue
        if len(files) == 0:
            print(f"No files found for {ds_name} in {folder}; skipping.")
            continue

        dataset = FaceLandmarksDataset(files, train=False)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

        print("Running model predictions...")
        preds, gts, norms = evaluate_model(model, loader, device)
        n_images = preds.shape[0]
        print(f"Got predictions for {n_images} images")

        nmes = []
        for p, g, nrm in zip(preds, gts, norms):
            dists = np.linalg.norm(p - g, axis=1)
            nme = float(np.mean(dists) / float(nrm))
            nmes.append(nme)
        nmes = np.array(nmes)

        thr, ced, auc_raw, auc_norm = ced_and_auc(nmes, T=config.MAX_ERROR_THRESHOLD, num=200)
        auc_msg = f"AUC_raw={auc_raw:.6f}, AUC_norm={auc_norm:.6f}"
        print(f"{ds_name} model AUC (0..{config.MAX_ERROR_THRESHOLD}): {auc_msg}")

        plt.figure(figsize=(6,4))
        plt.plot(thr, ced, label=f'{experiment_name} (AUC={auc_norm:.4f})')

        if ds_name.lower() == "menpo":
            try:
                predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
                detector = dlib.get_frontal_face_detector()
                print("Running dlib baseline...")
                d_preds, d_gts, d_norms = evaluate_dlib(predictor, files, detector)
                d_nmes = np.array([float(np.mean(np.linalg.norm(dp - dg, axis=1)) / float(nrm)) for dp, dg, nrm in zip(d_preds, d_gts, d_norms)])
                thr_d, ced_d, auc_raw_d, auc_norm_d = ced_and_auc(d_nmes, T=config.MAX_ERROR_THRESHOLD, num=200)
                plt.plot(thr_d, ced_d, label=f'dlib (AUC={auc_norm_d:.4f})')
                print(f"{ds_name} dlib AUC_norm: {auc_norm_d:.6f}")
            except Exception as e:
                print("Failed to evaluate dlib baseline:", e)

        plt.xlabel("Normalized mean error")
        plt.ylabel("Fraction of images")
        plt.title(f"CED on {ds_name}")
        plt.xlim(0, config.MAX_ERROR_THRESHOLD)
        plt.ylim(0, 1.0)
        plt.legend()
        out_png = Path(config.RESULTS_DIR) / f"CED_{ds_name}_{args.experiment_name}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

        with open(results_path / f"auc_results_{args.experiment_name}.txt", "a") as f:
            f.write(f"{ds_name}, AUC_raw: {auc_raw:.6f}, AUC_norm: {auc_norm:.6f}\n")

        csv_out = results_path / f"nme_{ds_name}_{args.experiment_name}.csv"
        with open(csv_out, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["image_idx", "nme"])
            for i, nm in enumerate(nmes):
                writer.writerow([i, float(nm)])

        print(f"Saved CED plot and CSV for {ds_name}. Model AUC_norm: {auc_norm:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    test(args.checkpoint, args.experiment_name)