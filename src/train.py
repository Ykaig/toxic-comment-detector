# src/train.py
import argparse
import yaml
import pathlib


def train(config_path: str):
    """
    Dummy training function.
    Reads the configuration, prints some messages, and creates a fake artifact.
    """
    # Read the configuration from the provided path
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    model_name = config['model_name']

    print(f"--- Dummy Training Started for model: {model_name} ---")

    # Simulate a training process
    print("Step 1: Loading data (simulated)...")
    print("Step 2: Preprocessing data (simulated)...")
    print("Step 3: Training model (simulated)...")

    # Create a fake artifact to simulate training output
    # The pipeline will expect an output to be saved in the future
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/dummy_model.pkl", "w") as f:
        f.write("This is a dummy model artifact.")

    print("Artifact 'dummy_model.pkl' created in /artifacts folder.")
    print("--- Dummy Training Complete ---")


if __name__ == '__main__':
    # Setup to accept command-line arguments
    parser = argparse.ArgumentParser(description="Dummy training script.")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the model configuration file.')
    args = parser.parse_args()

    train(config_path=args.config_path)