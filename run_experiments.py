import subprocess
from pathlib import Path
import config

from train import train
from test import test

def run_experiment(exp):
    name = exp["name"]
    model_type = exp["model_type"]
    loss_type = exp["loss_type"]
    head_type = exp["head_type"]

    train(name, model_type, loss_type, head_type)
    
    checkpoint = Path(config.CHECKPOINT_DIR) / f"{name}.ckpt"
    test(checkpoint, name)
    
if __name__ == "__main__":
    for exp in config.EXPERIMENTS:
        run_experiment(exp)

    # test_cmd = f"python report_generation.py"
    # subprocess.run(test_cmd, shell=True, check=True)
