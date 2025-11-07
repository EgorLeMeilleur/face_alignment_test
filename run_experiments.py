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
    for exp in config.EXPERIMENTS:
        run_experiment(exp)
