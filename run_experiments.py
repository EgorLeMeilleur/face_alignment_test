import subprocess
from pathlib import Path
import config

def run_experiment(exp):
    name = exp["name"]
    model_type = exp["model_type"]
    loss_type = exp["loss_type"]
    
    train_cmd = f"python train.py --experiment_name {name} --model_type {model_type} --loss_type {loss_type}"
    print(f"Running training: {train_cmd}")
    subprocess.run(train_cmd, shell=True, check=True)
    
    checkpoint = Path(config.CHECKPOINT_DIR) / f"{name}.ckpt"
    test_cmd = f"python test.py --checkpoint {checkpoint} --experiment_name {name}"
    print(f"Running testing: {test_cmd}")
    subprocess.run(test_cmd, shell=True, check=True)
    
if __name__ == "__main__":
    for exp in config.EXPERIMENTS:
        run_experiment(exp)

    # test_cmd = f"python report_generation.py"
    # subprocess.run(test_cmd, shell=True, check=True)
