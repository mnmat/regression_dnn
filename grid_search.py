from hyperparameter_search import train_model
import sys
import argparse
import os



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grid_search.py <tune_log>")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Grid Search for Hyperparameters")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the model",default=0.001)
    parser.add_argument("--batchsize", type=int, help="Batch size for training", default=512)
    parser.add_argument("--hidden_layers", type=list, nargs='+', help="Dimensions of each layer in the model", default=[512, 512, 512])
    parser.add_argument("--alphaL_min", type=float, help="Minimum alphaL value", default=1)
    parser.add_argument("--alphaR_min", type=float, help="Minimum alphaR value", default=1)
    parser.add_argument("--enable_gpu", action='store_true', help="Enable GPU training if available")


    args = parser.parse_args()
    
    config = {
        "learning_rate": float(args.learning_rate),
        "batchsize": int(args.batch_size),
        "hidden_layers": args.hidden_layers,
        "alphaL_min": args.alphaL_min,
        "alphaR_min": args.alphaR_min,
        "enable_gpu": args.enable_gpu,
        "tag": "grid_search"
    }
    
    train_model(config, tune_log=False)


