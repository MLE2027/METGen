# METGen

Source codes for the Time-Frequency Diffusion Model "METGen".

## 1. Setup

* **Python**: 3.8+ recommended.
* **Dependencies**: Install necessary packages.
    ```bash
    pip install -r requirements.txt
    ```
    Ensure PyTorch is installed matching your CUDA version if GPU is used.
* **Data**:
    * For CMAPSS datasets (e.g., `FD001`): Place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in the `./data/` directory.

## 2. Operating `Main.py`

`Main.py` is the central script for all core tasks. Its behavior is primarily controlled by command-line arguments.

### Key Command-Line Arguments:

* `--dataset <name>`: Specifies the dataset (e.g., `FD001`, `FD002`).
* `--model_name <name>`: Selects the METGen model architecture (e.g., `METGen`).
* `--state <mode>`: Determines the operation:
    * `train`: Train a new model.
    * `sample`: Generate synthetic data using a trained model.
    * `eval`: Evaluate generated synthetic data.
    * `all`: Perform train, sample, and then eval sequentially.
* `--window_size <int>`: Length of the time-series window (e.g., `48`).
* `--input_size <int>`: Number of input features/sensors (e.g., `14` for FD001).
* `--T <int>`: Number of diffusion timesteps (e.g., `1000`).
* `--epoch <int>`: Number of training epochs (used with `state='train'`).
* `--lr <float>`: Learning rate (used with `state='train'`, e.g., `2e-3`).
* `--batch_size <int>` or `-b <int>`: Batch size for the model training/sampling (e.g., `64`).
* `--model_path <path>`: Path to save (during training) or load (during sampling/eval) model weights.
    * Default format: `weights/<model_name>_<dataset>_<window_size>.pth`
* `--syndata_path <path>`: Path to save (during sampling) or load (during eval) synthetic data.
    * Default format: `./weights/syn_data/syn_<dataset>_<model_name>_<window_size><sample_type>.npz`

### 2.1. Training (`state='train'`)

This mode trains the diffusion model on the specified dataset.

* **Purpose**: To learn the data distribution and how to generate RUL-conditioned time series.
* **Example Command**:
    ```bash
    python Main.py \
        --dataset FD001 \
        --model_name METGen \
        --state train \
        --window_size 48 \
        --input_size 14 \
        --T 1000 \
        --epoch 70 \
        --lr 2e-3 \
        --batch_size 64
    ```
* **Output**:
    * Trained model weights saved to the path specified by `--model_path` (or the default path).
    * After training, it automatically proceeds to sample data using the trained model and training labels as conditions.

### 2.2. Generating / Sampling Data (`state='sample'`)

This mode uses a pre-trained model to generate synthetic time-series data.

* **Purpose**: To create new data samples conditioned on RUL values (typically from the training set labels).
* **Prerequisite**: A trained model (weights file).
* **Example Command**:
    ```bash
    python Main.py \
        --dataset FD001 \
        --model_name METGen \
        --state sample \
        --window_size 48 \
        --input_size 14 \
        --T 1000 \
        --model_path weights/METGen_FD001_48.pth # Specify path to your trained model
    ```
* **Output**:
    * Synthetic data saved as an `.npz` file to the path specified by `--syndata_path` (or the default path).
    * After sampling, it automatically proceeds to evaluate the generated samples.

### 2.3. Evaluating Data (`state='eval'`)

This mode evaluates the quality of previously generated synthetic data against real test data.

* **Purpose**: To assess the fidelity and utility of the generated samples using various metrics.
* **Prerequisites**:
    * Generated synthetic data (an `.npz` file).
    * Real test data for the specified dataset.
* **Example Command**:
    ```bash
    python Main.py \
        --dataset FD001 \
        --model_name METGen \
        --state eval \
        --window_size 48 \
        --input_size 14 \
        --syndata_path ./weights/syn_data/syn_FD001_METGen_48ddpm.npz # Specify path to your synthetic data
    ```
* **Output**:
    * Prints main evaluation scores to the console.
    * Logs metrics to Weights & Biases (if configured, offline by default). Metrics include:
        * Predictive scores (RMSE, MAE from a downstream RUL predictor).
        * Discriminative score.

### 2.4. All-in-One (`state='all'`)

This mode sequentially performs training, then sampling, and finally evaluation.

* **Purpose**: For a complete end-to-end run from training to evaluation.
* **Example Command**:
    ```bash
    python Main.py \
        --dataset FD001 \
        --model_name METGen \
        --state all \
        --window_size 48 \
        --input_size 14 \
        --T 1000 \
        --epoch 70 \
        --lr 2e-3 \
        --batch_size 64
    ```
* **Output**: Combines outputs from training, sampling, and evaluation stages.

## 3. Default Run Behavior

If `Main.py` is run without any command-line arguments:
```bash
python Main.py
```
