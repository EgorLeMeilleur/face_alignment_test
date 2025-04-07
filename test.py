import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import config
from dataset import FaceLandmarksDataset, read_pts
from model import FaceAlignmentModel
import dlib
import cv2

def evaluate_model(model, loader):
    predictions = []
    gt = []
    normalizations = []
    img_paths = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to('cuda')
            landmarks = batch["landmarks"]
            scale = batch["scale"]
            preds = model(images)
            for i in range(preds.shape[0]):
                face_rect = batch["face_rect"][i].cpu().numpy()
                scale_i = scale[i].cpu().numpy()
                pred_points = preds[i].cpu().numpy() * scale_i
                pred_points = pred_points + np.array([face_rect[0], face_rect[1]])
                true_points = landmarks[i].cpu().numpy() * scale_i
                true_points = true_points + np.array([face_rect[0], face_rect[1]])
                H = face_rect[3] - face_rect[1]
                W = face_rect[2] - face_rect[0]
                norm_factor = np.sqrt(H * W)
                predictions.append(pred_points)
                gt.append(true_points)
                normalizations.append(norm_factor)

    return np.array(predictions), np.array(gt), np.array(normalizations)

def evaluate_dlib(predictor, files, detector):
    predictions = []
    gt = []
    normalizations = []
    for file in files:
        image = cv2.imread(file[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(image)
        if len(faces) == 0:
            h, w = image.shape[:2]
            face_rect = dlib.rectangle(0, 0, w, h)
        else:
            face_rect = faces[0]
        x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
        shape = predictor(image, face_rect)
        pred_points = np.array([[p.x, p.y] for p in shape.parts()])
        true_points = read_pts(file[1])
        H = y2 - y1
        W = x2 - x1
        norm_factor = np.sqrt(H * W)
        predictions.append(pred_points)
        gt.append(true_points)
        normalizations.append(norm_factor)
    return np.array(predictions), np.array(gt), np.array(normalizations)

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


def main(args):
    results_path = Path(config.RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    log_file =  results_path / f"auc_results_{args.experiment_name}.txt"
    thresholds = np.linspace(0, 1, 100)
    plot_thresholds = np.linspace(0, config.MAX_ERROR_THRESHOLD, 100)
    for ds_name, folder in config.TEST_FOLDERS.items():
        files = []
        for ext in ["*.jpg", "*.png"]:
            for img_path in folder.glob(ext):
                pts_path = img_path.with_suffix(".pts")
                if pts_path.exists() and len(read_pts(str(pts_path))) == 68:
                    files.append((img_path, pts_path))
        dataset = FaceLandmarksDataset(files, train=False)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        
        model = FaceAlignmentModel.load_from_checkpoint(args.checkpoint, map_location=torch.device('cuda'))
        preds, gt, normalizations = evaluate_model(model, loader)
        ceds = count_ced(preds, gt, normalizations)
        ced_curve = np.array([np.mean(ceds < thr) for thr in thresholds])
        auc_model = np.trapezoid(ced_curve, thresholds)
        
        plt.figure()
        plt.plot(plot_thresholds, ced_curve, label=f'{args.experiment_name} (AUC: {auc_model:.4f})')
        
        if ds_name.lower() == "menpo":
            try:
                predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
                detector = dlib.get_frontal_face_detector()
                preds, gt, normalizations = evaluate_dlib(predictor, files, detector)
                ceds = count_ced(preds, gt, normalizations)
                ced_curve = np.array([np.mean(ceds < thr) for thr in thresholds])
                auc_dlib = np.trapezoid(ced_curve, thresholds)
                plt.plot(plot_thresholds, ced_curve, label=f'dlib (AUC: {auc_dlib:.4f})')
                print(f"{ds_name} dataset - dlib AUC: {auc_dlib:.4f}")
            except Exception as e:
                print("Не удалось загрузить dlib shape predictor:", e)
        
        plt.xlabel('Normalized Mean Square Error')
        plt.ylabel('Percentage of Images')
        plt.title(f'CED Curve on {ds_name} dataset')
        plt.legend()
        
        plot_file = results_path / f"CED_{ds_name}_{args.experiment_name}.png"
        plt.savefig(str(plot_file))
        plt.close()

        with open(log_file, "a") as f:
            f.write(f"{ds_name}, AUC: {auc_model:.4f}\n")
        
        print(f"{ds_name} dataset - Model AUC: {auc_model:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    main(args)