# DA6401: Seq2Seq Transliteration with Attention and Vanilla RNN Models

## Overview

This repository implements Sequence-to-Sequence (Seq2Seq) transliteration models, converting English characters into Hindi characters using Recurrent Neural Networks (RNNs). Two distinct models are provided:

- **Attention-based Seq2Seq Model (`attention.py`)**: Utilizes an attention mechanism, allowing the decoder to selectively focus on specific parts of the encoder outputs.
- **Vanilla Seq2Seq Model (`vanilla.py`)**: Implements a straightforward encoder-decoder structure without attention, serving as a baseline for comparison.

Both models employ Long Short-Term Memory (LSTM) layers and are trained and evaluated on the Dakshina dataset, a widely-used transliteration dataset supporting robust experimentation.

The project integrates **Weights & Biases (wandb)** for extensive experiment tracking, hyperparameter tuning, and visualization of training metrics. Detailed training logs, model comparisons, and performance insights are available via the wandb dashboard.

- **Weights & Biases Report:** [View Detailed Experiment Report](https://wandb.ai/cs24m021-iit-madras/DA6401_A3/reports/Assignment-3--VmlldzoxMjg0MzgwNg?accessToken=lhh5nl2o7rdbjftu9gw8l4wl3pwvdv9hz3ce6s42wz4z6w5h8uuvlfa4a0hdgces)


---

## Project Structure

* **attention.py**: Implements Seq2Seq architecture enhanced with an attention mechanism.
* **vanilla.py**: Implements a basic Seq2Seq model without an attention mechanism.

---

## Requirements

* Python 3.x
* PyTorch
* NumPy
* pandas
* matplotlib
* seaborn
* tqdm
* wandb (Weights & Biases)

Install dependencies via:

```bash
pip install torch numpy pandas matplotlib seaborn tqdm wandb
```

---

## Dataset

This project utilizes the **Dakshina Dataset** for transliteration tasks, specifically converting English characters to Hindi.

Ensure the following dataset files are downloaded and placed in the **same directory** as your scripts:

- **`hi.translit.sampled.train.tsv`** – Training data.
- **`hi.translit.sampled.dev.tsv`** – Validation data.
- **`hi.translit.sampled.test.tsv`** – Test data.

You can access the complete Dakshina dataset here:
- [Dakshina Dataset on Kaggle](https://github.com/google-research-datasets/dakshina$0)

---

## Running the Models

Run the Attention Model:

```bash
python attention.py --wandb_project "YourProjectName" --batch_size 16 --emb_dim 64 --num_enc_layers 3 --num_dec_layers 3 --hidden_size 512
```

Run the Vanilla Model:

```bash
python vanilla.py --wandb_project "YourProjectName" --batch_size 32 --emb_dim 64 --num_enc_layers 3 --num_dec_layers 3 --hidden_size 256
```

You can configure other hyperparameters through command-line arguments as listed in each script.

---

## Hyperparameters

Default hyperparameters include:

* Learning Rate: `0.001`
* Batch Size: `16` (attention), `32` (vanilla)
* Embedding Dimension: `64`
* Encoder & Decoder Layers: `3`
* Hidden Size: `512` (attention), `256` (vanilla)
* Dropout: `0.5`
* Cell Type: `LSTM`
* Bidirectional: `True`

Adjust these parameters via CLI.

---

## Logging and Tracking

This project integrates Weights & Biases (wandb) for experiment tracking:

* Login using your wandb API key within the scripts.
* Track model training and visualize results directly on the wandb dashboard.

---

## Evaluation and Results

Model evaluation involves accuracy and loss metrics tracked across epochs and visualized through generated plots. Plots are saved automatically.

---

## Acknowledgements

* This project was developed as part of the DA6401 course assignment.
* Uses wandb for insightful experiment tracking and model management.


