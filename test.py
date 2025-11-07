import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import dlib
from PIL import Image
import torch
import torch.nn.functional as F

import config
from dataset import FaceLandmarksDataset, read_pts
from models import FaceAlignmentModel

def heatmaps_to_coords_batch(heatmaps):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    prob = F.softmax(flat, dim=-1)
    prob = prob.view(B, K, H, W)

    xs = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, W)
    ys = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, H, 1)

    exp_x = (prob * xs).view(B, K, -1).sum(dim=-1)
    exp_y = (prob * ys).view(B, K, -1).sum(dim=-1)

    coords = torch.stack([exp_x, exp_y], dim=-1)
    return coords.cpu().numpy()


def map_hm_to_resized_batch(peaks_hm, resized_size, hm_size):
    H_resized, W_resized = int(resized_size[0]), int(resized_size[1])
    H_hm, W_hm = int(hm_size[0]), int(hm_size[1])

    scale_x = float(W_resized) / float(W_hm)
    scale_y = float(H_resized) / float(H_hm)

    out = np.zeros_like(peaks_hm, dtype=np.float32)
    out[..., 0] = peaks_hm[..., 0] * scale_x
    out[..., 1] = peaks_hm[..., 1] * scale_y
    return out


def map_resized_to_orig_batch(coords_resized, crop_box_batch, scale_batch):
    B = coords_resized.shape[0]
    out = np.zeros_like(coords_resized, dtype=np.float32)
    for i in range(B):
        x1, y1 = float(crop_box_batch[i, 0]), float(crop_box_batch[i, 1])
        sx, sy = float(scale_batch[i, 0]), float(scale_batch[i, 1])
        out[i, :, 0] = coords_resized[i, :, 0] * sx + x1
        out[i, :, 1] = coords_resized[i, :, 1] * sy + y1
    return out


def count_ced(predicted_points, gt_points, normalizations):
    n = predicted_points.shape[0]
    errs = []
    for i in range(n):
        preds = predicted_points[i]
        gts = gt_points[i]
        dists = np.linalg.norm(preds - gts, axis=1)
        avg = float(np.mean(dists))
        errs.append(avg / float(normalizations[i]))
    return np.sort(np.array(errs, dtype=np.float32))


@torch.no_grad()
def evaluate_model(model, loader, device, hm_size=(64,64)):
    model.to(device)
    model.eval()

    all_preds = []
    all_gts = []
    all_norms = []

    H_hm, W_hm = int(hm_size[0]), int(hm_size[1])

    for batch in loader:
        imgs = batch["image"].to(device)
        B = imgs.shape[0]

        out = model(imgs)

        crop_box_batch = batch["crop_box"].cpu().numpy().astype(np.float32)
        scale_batch = batch["scale"].cpu().numpy().astype(np.float32)

        kps_norm = batch["keypoints_norm"].cpu().numpy()
        resized_h, resized_w = imgs.shape[2], imgs.shape[3]
        kps_px = np.zeros_like(kps_norm, dtype=np.float32)
        kps_px[..., 0] = kps_norm[..., 0] * float(resized_w)
        kps_px[..., 1] = kps_norm[..., 1] * float(resized_h)
        gt_orig_batch = map_resized_to_orig_batch(kps_px, crop_box_batch, scale_batch)

        if out.dim() == 3:
            preds_norm = out.detach().cpu().numpy()
            pred_px = np.zeros_like(preds_norm, dtype=np.float32)
            pred_px[..., 0] = preds_norm[..., 0] * float(resized_w)
            pred_px[..., 1] = preds_norm[..., 1] * float(resized_h)
            pred_orig_batch = map_resized_to_orig_batch(pred_px, crop_box_batch, scale_batch)

        elif out.dim() == 4:
            hm_tensor = out.detach()
            if hm_tensor.shape[2] != H_hm or hm_tensor.shape[3] != W_hm:
                H_hm_model, W_hm_model = hm_tensor.shape[2], hm_tensor.shape[3]
            else:
                H_hm_model, W_hm_model = H_hm, W_hm

            peaks = heatmaps_to_coords_batch(hm_tensor)
            coords_resized_batch = map_hm_to_resized_batch(peaks, (resized_h, resized_w), (H_hm_model, W_hm_model))
            pred_orig_batch = map_resized_to_orig_batch(coords_resized_batch, crop_box_batch, scale_batch)

        face_rects = batch["face_rect"].cpu().numpy().astype(np.float32)
        norms = np.sqrt(np.maximum(1.0, (face_rects[:, 3] - face_rects[:, 1]) * (face_rects[:, 2] - face_rects[:, 0])))

        for i in range(B):
            all_preds.append(pred_orig_batch[i])
            all_gts.append(gt_orig_batch[i])
            all_norms.append(float(norms[i]))

    return np.array(all_preds), np.array(all_gts), np.array(all_norms)

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
    parser.add_argument("--model_type", type=str, default=config.MODEL_TYPE, options=['efficientnet', 'convnext'])
    parser.add_argument("--loss_type", type=str, default=config.LOSS_TYPE, options=['mse', 'wing', 'awing', 'focal'])
    parser.add_argument("--head_type", type=str, default=config.HEAD_TYPE, options=['regression', 'heatmap'])
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    test(args.experiment_name, args.model_type, args.loss_type, args.head_type)