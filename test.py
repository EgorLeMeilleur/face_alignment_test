import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import config
from dataset import FaceLandmarksDataset, get_transforms
from model import FaceAlignmentModel

def evaluate_model(model, loader):
    all_errors = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            landmarks = batch["landmarks"]
            preds = model(images)
            for i in range(preds.shape[0]):
                pred_points = preds[i].cpu().numpy()
                true_points = landmarks[i].cpu().numpy()
                face_rect = batch["face_rect"][i].cpu().numpy()
                H = face_rect[3] - face_rect[1]
                W = face_rect[2] - face_rect[0]
                norm_factor = np.sqrt(H * W)
                error = np.mean(np.linalg.norm(pred_points - true_points, axis=1)) / norm_factor
                all_errors.append(error)
    return np.array(all_errors)

def plot_ced(errors, dataset_name, experiment_name):
    thresholds = np.linspace(0, 0.08, 100)
    ced = [np.mean(errors < t) for t in thresholds]
    auc_area = auc(thresholds, ced)
    
    plt.figure()
    plt.plot(thresholds, ced, label=f'{experiment_name} (AUC: {auc_area:.4f})')
    plt.xlabel('Normalized Mean Square Error')
    plt.ylabel('Percentage of images')
    plt.title(f'CED Curve on {dataset_name} dataset')
    plt.legend()
    
    results_path = Path(config.RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    plot_file = results_path / f"CED_{dataset_name}_{experiment_name}.png"
    plt.savefig(str(plot_file))
    plt.close()
    return auc_area

def main(args):
    for ds_name, folder in config.TEST_FOLDERS.items():
        dataset = FaceLandmarksDataset([folder], transform=get_transforms(train=False))
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        
        model = FaceAlignmentModel.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
        errors = evaluate_model(model, loader)
        auc_area = plot_ced(errors, ds_name, args.experiment_name)
        print(f"{ds_name} dataset: AUC area = {auc_area:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    main(args)
