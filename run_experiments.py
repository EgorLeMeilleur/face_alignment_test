import argparse

import config
from train import train
from test import test

def run_experiment(exp):
    name = exp["name"]
    model_type = exp["model_type"]
    loss_type = exp["loss_type"]
    head_type = exp["head_type"]

    train(name, model_type, loss_type, head_type)
    test(name, model_type, loss_type, head_type)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true", help="Run only the best experiment")
    args = parser.parse_args()

    if args.best:
        print("Running best experiment only...")
        run_experiment(config.BEST_EXPERIMENT)
    else:
        for exp in config.EXPERIMENTS:
            run_experiment(exp)
