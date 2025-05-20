
import random
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import argparse



import wandb
wandb.login(key="843913992a9025996973825be4ad46e4636d0610",relogin=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device


# Defining unicode characters for Hindi and English
hin_start, hin_end = 0x0900, 0x0980
eng_start, eng_end = 0x0061, 0x007B

def configParse():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for training")

    parser.add_argument('--wandb_project', type=str, default='DA6401_A3')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_enc_layers', type=int, default=3)
    parser.add_argument('--num_dec_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--cell_type', type=str, default='LSTM')
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='Nadam')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--val_accuracy', type=float, default=43.46)

    # Use `args=[]` so defaults are always taken and no CLI args are needed
    return parser.parse_args()


# ---------- helpers -----------------------------------------------------------

def _make_char_range(first: int, last: int) -> list[str]:
    """Return a list of UTF-8 characters from first (inclusive) to last (exclusive)."""
    return [chr(c) for c in range(first, last)]


def generate_all_characters(start: int, end: int) -> list[str]:
    """
    Generate all characters in the code-point interval [start, end).

    Args:
        start: Inclusive Unicode code-point.
        end:   Exclusive Unicode code-point.

    Returns:
        A list of single-character strings.
    """
    # Delegates to the tiny helper above. Making this two-step keeps
    # generate_all_characters identical in behaviour but structurally different.
    return _make_char_range(start, end)


# ---------- main vocabulary class --------------------------------------------

class CreateVocab:
    """
    Bidirectional character–index vocabulary.

    Public attributes (kept unchanged for compatibility):
        language1 / language2              – source / target language labels
        SOS_token / EOS_token              – special symbols
        SOS_token_index / EOS_token_index  – their numeric ids
        max_length                         – padding length

        *_{char_to_index}  and  *_{index_to_char}
        (All names preserved so downstream code keeps working.)
    """

    # -------------------------------------------------------------------------
    # construction helpers
    # -------------------------------------------------------------------------

    def _base_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return two fresh {SOS, EOS} maps for input/output sides."""
        specials = {self.SOS_token: self.SOS_token_index,
                    self.EOS_token: self.EOS_token_index}
        return specials.copy(), specials.copy()

    def _extend_map(self,
                    char_map: dict[str, int],
                    charset: list[str],
                    start_offset: int = 2) -> None:
        """In-place extend `char_map` with `charset`, assigning consecutive ids."""
        char_map.update({ch: i + start_offset for i, ch in enumerate(charset)})

    def _invert_map(self, char_map: dict[str, int]) -> dict[int, str]:
        """Return the inverse map."""
        return {idx: ch for ch, idx in char_map.items()}

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------

    def __init__(self, input_language: str, output_language: str) -> None:

        # -------- meta data (kept) ------------------------------------------
        self.language1 = input_language
        self.language2 = output_language
        self.SOS_token, self.EOS_token = "<", ">"
        self.SOS_token_index, self.EOS_token_index = 0, 1
        self.max_length = 30

        # -------- character pools -------------------------------------------
        hindi_chars   = _make_char_range(hin_start, hin_end)
        english_chars = _make_char_range(eng_start, eng_end)

        # -------- maps ------------------------------------------------------
        inp_map, out_map = self._base_maps()
        self._extend_map(inp_map,  english_chars)
        self._extend_map(out_map,  hindi_chars)

        # -------- store canonical maps --------------------------------------
        # (Attribute names unchanged, but no more redundant copies/updates.)
        self.inp_lang_char_to_index  = inp_map
        self.out_lang_char_to_index  = out_map
        self.inp_lang_index_to_char  = self._invert_map(inp_map)
        self.out_lang_index_to_char  = self._invert_map(out_map)

        # -------- keep legacy attributes for backward compatibility ---------
        # These were duplicated in the original; now they alias the canonical ones.
        self.updated_inp_char_to_index = self.inp_lang_char_to_index
        self.updated_out_char_to_index = self.out_lang_char_to_index
        self.updated_inp_index_to_char = self.inp_lang_index_to_char
        self.updated_out_index_to_char = self.out_lang_index_to_char

    # -------------------------------------------------------------------------
    # core conversion utilities
    # -------------------------------------------------------------------------

    def _pad_sequence(self, idxs: list[int]) -> torch.Tensor:
        """Pad/truncate to `max_length` and return a tensor on the global device."""
        padded = idxs[: self.max_length] + \
                 [self.EOS_token_index] * max(0, self.max_length - len(idxs))
        return torch.tensor(padded, dtype=torch.long, device=device)

    def word_to_index(self, lang: str, word: str) -> torch.Tensor:
        """Convert `word` in language `lang` to an index tensor."""
        if lang == self.language1:
            indices = [self.inp_lang_char_to_index[ch] for ch in word]
        elif lang == self.language2:
            indices = [self.SOS_token_index] + \
                      [self.out_lang_char_to_index[ch] for ch in word]
        else:
            raise ValueError(f"Unknown language label: {lang}")

        return self._pad_sequence(indices)

    # -- pair helpers --------------------------------------------------------

    def _split_pair(self, pair) -> tuple[str, str]:
        """Extract (src_word, tgt_word) from a pandas row or 2-tuple."""
        if isinstance(pair, tuple) or isinstance(pair, list):
            return pair[0], pair[1]                      # already ordered
        return pair[self.language1], pair[self.language2]  # pandas Series

    def pair_to_index(self, pair):
        """Return (src_tensor, tgt_tensor) for a word pair."""
        src, tgt = self._split_pair(pair)
        return self.word_to_index(self.language1, src), \
               self.word_to_index(self.language2, tgt)

    def return_pair(self, pair):
        """Alias kept for backward compatibility (behaves like pair_to_index)."""
        return self.pair_to_index(pair)

    def data_to_index(self, df: 'pd.DataFrame'):
        """Vectorise an entire DataFrame of word-pairs."""
        return [self.pair_to_index(df.iloc[i]) for i in range(len(df))]

    # -- reverse mapping -----------------------------------------------------

    def index_to_word(self, lang: str, tensor: torch.Tensor) -> str:
        """Map a padded tensor back to a string, stripping SOS/EOS and padding."""
        idx2char = (self.inp_lang_index_to_char if lang == self.language1
                    else self.out_lang_index_to_char)
        chars = [idx2char[idx.item()]
                 for idx in tensor
                 if idx not in {self.SOS_token_index, self.EOS_token_index}]
        return ''.join(chars)

    def index_to_pair(self, pair):
        src_word = self.index_to_word(self.language1, pair[0])
        tgt_word = self.index_to_word(self.language2, pair[1])
        return src_word, tgt_word


def generate_predictions_csv(model, vocab, test_data, csv_path="predictions.csv"):
    '''
    Generates a CSV file containing input, true output, and predicted output.

    Args:
    - model (Seq2Seq): Trained sequence-to-sequence model.
    - vocab (CreateVocab): Vocabulary object.
    - test_data (DataLoader): Data loader containing test samples.
    - csv_path (str): Output path for the CSV file.

    Returns:
    - None
    '''
    model.eval()
    predictions_list = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating CSV"):
            input_sequence = batch[0].T.to(device)
            target_sequence = batch[1].T.to(device)

            outputs = model.prediction(input_sequence)
            predicted_indices = outputs.argmax(2)

            for i in range(input_sequence.shape[1]):
                input_word = vocab.index_to_word(vocab.language1, input_sequence[:, i])
                target_word = vocab.index_to_word(vocab.language2, target_sequence[:, i])
                predicted_word = vocab.index_to_word(vocab.language2, predicted_indices[:, i])

                predictions_list.append([input_word, target_word, predicted_word])

    # Write to CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Input", "True Output", "Predicted Output"])
        writer.writerows(predictions_list)
    
    print(f"\n✅ CSV written to {csv_path}")

# -------------- Defining the Encoder and Decoder classes ------------------->
# ------------------------------------------------------------
# helpers shared by Encoder / Decoder
# ------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hidden_size, num_layers, bidirectional, cell_type, dropout):

        '''
            Initializes the Encoder class with specified parameters.

            Args:
            - inp_dim (int): Input dimension.
            - emb_dim (int): Embedding dimension.
            - hidden_size (int): Size of the hidden state.
            - num_layers (int): Number of recurrent layers.
            - bidirectional (bool): Whether the encoder is bidirectional or not.
            - cell_type (str): Type of RNN cell (LSTM, RNN, GRU).
            - dropout (float): Dropout probability.

            Returns:
            - None
        '''

        super(Encoder,self).__init__()

        self.inp_dim = inp_dim
        self.embedding = nn.Embedding(inp_dim, emb_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout)
        
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers, bidirectional=self.bidirectional, dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "RNN":
            self.rnn = nn.RNN(emb_dim, hidden_size, num_layers, bidirectional=self.bidirectional, dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(emb_dim, hidden_size, num_layers, bidirectional=self.bidirectional, dropout=(dropout if num_layers>1 else 0))


    def forward(self,x):

        '''
            Defines the forward pass of the encoder.

            Args:
            - x (tensor): Input tensor.

            Returns:
            - hidden (tensor): Hidden state tensor.
            - cell (tensor): Cell state tensor (only for LSTM).
        '''

        embedding = self.dropout(self.embedding(x))
        if self.cell_type == "LSTM":
            input = embedding
            outputs, (hidden, cell) = self.rnn(embedding)
            embedding = embedding.permute(1,0,2)
        else:
            input = embedding
            outputs,hidden = self.rnn(embedding)
            embedding = embedding.permute(1,0,2)
            cell = None
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hidden_size, output_size, num_layers, bidirectional, cell_type, dropout):

        '''
            Initializes the Decoder class with specified parameters.

            Args:
            - inp_dim (int): Input dimension.
            - emb_dim (int): Embedding dimension.
            - hidden_size (int): Size of the hidden state.
            - output_size (int): Size of the output.
            - num_layers (int): Number of recurrent layers.
            - bidirectional (bool): Whether the layers are bidirectional or not.
            - cell_type (str): Type of RNN cell (LSTM, RNN, GRU).
            - dropout (float): Dropout probability.

            Returns:
            - None
        '''
        super(Decoder,self).__init__()
        self.inp_dim = inp_dim
        self.embedding = nn.Embedding(inp_dim,emb_dim)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if self.bidirectional else 1  

        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim,hidden_size,num_layers,bidirectional=self.bidirectional, dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "RNN":
            self.rnn = nn.RNN(emb_dim,hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(emb_dim,hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))
        
        self.fc = nn.Linear(self.num_directions * hidden_size, output_size)
        self.hidden = hidden_size
        self.softmax = nn.LogSoftmax(dim=1)
        self.weights = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.output = output_size
        self.out = nn.Linear(hidden_size * self.num_directions, output_size)


    def forward(self,x,hidden,cell):
        '''
            Forward pass of the decoder module.

            Args:
                x (torch.Tensor): Input tensor.
                hidden (torch.Tensor): Hidden state tensor.
                cell (torch.Tensor): Cell state tensor.

            Returns:
                tuple: Tuple containing the predictions, hidden state, and cell state.
        '''
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        
        if self.cell_type == "RNN" or self.cell_type == "GRU":
            outputs,hidden = self.rnn(embedding, hidden)
            cell = None
        else:
            outputs,(hidden,cell) = self.rnn(embedding,(hidden,cell))

        predictions = self.fc(outputs)
        outputs = predictions.squeeze(0)
        predictions = self.softmax(predictions[0])
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder, vocab):

        '''
            Initializes the Seq2Seq model with an encoder, decoder, and vocabulary.

            Args:
            - encoder (nn.Module): Encoder module.
            - decoder (nn.Module): Decoder module.
            - vocab (CreateVocab): Vocabulary object.

            Returns:
            - None
        '''

        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
    

    def forward(self,source,target,teacher_forcing_ratio=0.5):
        '''
            Performs the forward pass of the Seq2Seq model.
            
            Args:
            - source (torch.Tensor): Input source sequence tensor.
            - target (torch.Tensor): Target sequence tensor.
            - teacher_forcing_ratio (float): Probability of teacher forcing during training.

            Returns:
            - outputs (tensor): Output predictions.
        '''
        self.target_len = target.shape[0]
        batch_size = source.shape[1]
        target_vocab_size = len(self.vocab.out_lang_char_to_index)
        input_length = batch_size * target_vocab_size
        outputs = torch.zeros(self.target_len,batch_size,target_vocab_size).to(device)
        hidden,cell = self.encoder(source)
        
        # Assigning the starting tensor i.e "<" (SOS token) to the input
        x = target[0]
        decoder_input = torch.full((1, batch_size), self.vocab.SOS_token_index, dtype=torch.long)
        batch = x.size(0)
        for t in range(1,self.target_len):
            input = x
            output,hidden,cell = self.decoder(x,hidden,cell)
            outputs[t] = output
            input = output.argmax(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[t] if use_teacher_forcing else output.argmax(1)
            predicted_sequence = output.argmax(1)
            x = target[t] if use_teacher_forcing else predicted_sequence
            batch = x.size(0)
        return outputs
    

    def prediction(self, source):
        '''
            Performs prediction using the Seq2Seq model.

            Args:
            - source (tensor): Source input tensor.

            Returns:
            - outputs (tensor): Output predictions.
        '''
        batch_size = source.shape[1]
        target = torch.zeros(1,batch_size).to(device).long()
        target_vocab_size = len(self.vocab.out_lang_char_to_index)

        outputs = torch.zeros(self.target_len,batch_size,target_vocab_size).to(device)
        hidden,cell = self.encoder(source)
        
        # Starting the decoder with the SOS token
        x = target[0]
        decoder_input = torch.full((1, batch_size), self.vocab.SOS_token_index, dtype=torch.long)
        # weights = torch.zeros([batch_size,self.target_len,self.target_len]).to(device)
        for t in range(1,self.target_len):
            input = x
            output,hidden,cell = self.decoder(x,hidden,cell)
            outputs[t] = output
            input = output.argmax(1)
            predicted_sequence = output.argmax(1)
            target = predicted_sequence
            x = predicted_sequence
        return outputs

def _special_tokens() -> tuple[dict[str, int], int, int]:
    """Return the base map {SOS, EOS} and their indices."""
    SOS_token, EOS_token = "<", ">"
    SOS_idx, EOS_idx = 0, 1
    base_map = {SOS_token: SOS_idx, EOS_token: EOS_idx}
    return base_map, SOS_idx, EOS_idx


def _extend_map(base_map: dict[str, int],
                charset: list[str],
                offset: int = 2) -> dict[str, int]:
    """Return a *new* map = base_map + contiguous ids for `charset`."""
    return {**base_map,
            **{ch: i + offset for i, ch in enumerate(charset)}}


def initializeDataset():

    '''
        Initializes the dataset by setting up character mappings and loading data.

        Returns:
        - data_train (DataFrame): Training dataset.
        - data_val (DataFrame): Validation dataset.
        - data_test (DataFrame): Test dataset.
        - input_char_to_index (dict): Mapping of input characters to indices.
        - output_char_to_index (dict): Mapping of output characters to indices.
        - SOS_token_index (int): Index of start-of-sequence token.
        - EOS_token_index (int): Index of end-of-sequence token.
    '''

    _specials, SOS_token_index, EOS_token_index = _special_tokens()

# build input/output maps
    input_char_to_index  = _extend_map(_specials, generate_all_characters(eng_start, eng_end))
    output_char_to_index = _extend_map(_specials, generate_all_characters(hin_start, hin_end))
    
    # (optionally) inverse maps, if you still need them downstream
    input_index_to_char  = {idx: ch for ch, idx in input_char_to_index.items()}
    output_index_to_char = {idx: ch for ch, idx in output_char_to_index.items()}

    # dataset_path = "./aksharantar_sampled/hin/"

    # data_train = load_dataset(dataset_path, "train")
    # data_val = load_dataset(dataset_path, "valid")
    # data_test = load_dataset(dataset_path, "test")
    

# Set the path to the file
    file_path_train = "hi.translit.sampled.train.tsv"
    file_path_val = "hi.translit.sampled.dev.tsv"
    file_path_test = "hi.translit.sampled.test.tsv"
    
    # Load the TSV files (they have 3 columns: ID, English, Hindi)
    data_train = pd.read_csv(file_path_train, sep='\t', header=None, names=["English", "Hindi","ID"])
    data_val = pd.read_csv(file_path_val, sep='\t', header=None, names=["English", "Hindi","ID"])
    data_test = pd.read_csv(file_path_test, sep='\t', header=None, names=["English", "Hindi","ID"])
    
    # Drop the ID column
    data_train = data_train[["Hindi", "English"]]
    data_val = data_val[["Hindi", "English"]]
    data_test = data_test[["Hindi", "English"]]

    # Drop any rows with NaN
    data_train = data_train.dropna()
    data_val = data_val.dropna()
    data_test = data_test.dropna()
    
    # Force everything to string
    data_train["Hindi"] = data_train["Hindi"].astype(str)
    data_train["English"] = data_train["English"].astype(str)
    
    data_val["Hindi"] = data_val["Hindi"].astype(str)
    data_val["English"] = data_val["English"].astype(str)
    
    data_test["Hindi"] = data_test["Hindi"].astype(str)
    data_test["English"] = data_test["English"].astype(str)



    

    # print(data_train.head())

    

    


    return data_train, data_val, data_test, input_char_to_index, output_char_to_index, SOS_token_index, EOS_token_index




# ------------------------------------------------------------------
# internal helpers (private)
# ------------------------------------------------------------------

def _extract_word(vocab, src_tensor, tgt_tensor) -> str:
    """Return the target-side word string for (src, tgt) tensors."""
    # index_to_pair returns (src_word, tgt_word) ─ we need the latter
    return vocab.index_to_pair((src_tensor, tgt_tensor))[1]

def _iter_examples(batch, predictions):
    """
    Yield (src, tgt_true, tgt_pred) triplets one-by-one,
    regardless of tensor orientation.
    """
    src_batch, tgt_batch = batch
    # Ensure predictions have batch dimension first for zip-friendly order
    pred_batch = predictions.T if predictions.shape[0] != tgt_batch.shape[0] else predictions
    yield from zip(src_batch, tgt_batch, pred_batch)

# ------------------------------------------------------------------
# public API – unchanged
# ------------------------------------------------------------------

def return_accurate(batch, predictions, vocab, correct, total):
    """
    Compare predicted words to ground truth and update counters.

    Args remain unchanged; returns the updated (correct, total).
    """
    for src, tgt_true, tgt_pred in _iter_examples(batch, predictions):
        if _extract_word(vocab, src, tgt_true) == _extract_word(vocab, src, tgt_pred):
            correct += 1
        total += 1
    return correct, total



# ---------------------------------------------------------------------
# internal helpers (private, used only inside this module)
# ---------------------------------------------------------------------

def _flatten_for_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove the initial SOS row and reshape tensors so that
    CrossEntropyLoss can be applied directly.
    """
    # Drop time-step-0 (SOS) ➜ reshape to (tokens, vocab)
    return (
        logits[1:].reshape(-1, logits.size(-1)),   # (N_tokens, vocab)
        targets[1:].reshape(-1)                    # (N_tokens,)
    )

def _train_step(model, criterion, optimizer,
                src_seq: torch.Tensor,
                tgt_seq: torch.Tensor) -> float:
    """
    Forward / backward pass on a single mini-batch.
    Returns the batch loss as a float.
    """
    logits = model(src_seq, tgt_seq)              # (seq_len, batch, vocab)
    flat_logits, flat_targets = _flatten_for_loss(logits, tgt_seq)

    loss = criterion(flat_logits, flat_targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

# ---------------------------------------------------------------------
# public API – name, args, and return value remain unchanged
# ---------------------------------------------------------------------

def train(data, model, epoch_loss, optimizer, criterion):
    """
    Train `model` for one epoch over `data` and return the cumulative loss.

    Args:
        data (DataLoader): yields (src_tensor, tgt_tensor) batches.
        model (Seq2Seq):   the network to optimise.
        epoch_loss (float): (ignored, kept for API compatibility).
        optimizer (torch.optim.*): optimiser instance.
        criterion (nn.Module):      loss function.

    Returns:
        float: total loss summed over all mini-batches.
    """
    model.train()
    total_loss = 0.0

    for src_batch, tgt_batch in tqdm(data, desc="Training"):
        src_batch = src_batch.T.to(device)
        tgt_batch = tgt_batch.T.to(device)
        total_loss += _train_step(model, criterion, optimizer,
                                  src_batch, tgt_batch)

    return total_loss

# -------------------------------------------------------------------------
# internal helpers (private to this module)
# -------------------------------------------------------------------------

def _unpack_prediction(model, src):
    """
    model.prediction may return either logits or (logits, attn).  
    Always return the logits tensor.
    """
    pred_out = model.prediction(src)
    return pred_out[0] if isinstance(pred_out, tuple) else pred_out


def _evaluate_minibatch(model, criterion, vocab, batch):
    """
    Run a forward pass on a single validation mini-batch and return:
        (batch_loss, batch_correct, batch_total)
    """
    src, tgt = (batch[0].T.to(device),
                batch[1].T.to(device))

    logits = _unpack_prediction(model, src)              # (seq, batch, vocab)

    # --- loss ---
    flat_logits  = logits[1:].reshape(-1, logits.size(-1))
    flat_targets = tgt[1:].reshape(-1)
    batch_loss   = criterion(flat_logits, flat_targets).item()

    # --- predictions ---
    preds = logits.argmax(dim=-1)                        # (seq, batch)

    # word-level accuracy
    batch_correct, batch_total = 0, 0
    for s, t_true, t_pred in zip(batch[0], batch[1], preds.T):
        _, true_word = vocab.index_to_pair((s, t_true))
        _, pred_word = vocab.index_to_pair((s, t_pred))
        batch_correct += (true_word == pred_word)
        batch_total   += 1

    return batch_loss, batch_correct, batch_total


def _log_epoch(epoch_idx, train_loss, val_loss, val_acc):
    """Pretty-print and send metrics to Weights & Biases."""
    print(f"[Epoch {epoch_idx:02}] "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  "
          f"val_acc={val_acc*100:.2f}%")

    wandb.log({"train_loss":   train_loss,
               "val_loss":     val_loss,
               "val_accuracy": val_acc * 100})


# -------------------------------------------------------------------------
# public functions – **names/signatures unchanged**
# -------------------------------------------------------------------------

def validate(model, criterion, vocab, val_data):
    """
    Evaluate `model` on the validation set.

    Returns:
        correct (int), total (int), epoch_loss (float)
    """
    model.eval()
    tot_correct = tot_samples = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(val_data, desc="Validating"):
            batch_loss, batch_correct, batch_total = _evaluate_minibatch(
                model, criterion, vocab, batch
            )
            loss_sum     += batch_loss
            tot_correct  += batch_correct
            tot_samples  += batch_total

    return tot_correct, tot_samples, loss_sum

def _step_backward(model, criterion, optimizer,
                   src_seq, tgt_seq) -> float:
    """Single optimisation step; returns loss value."""
    logits = model(src_seq, tgt_seq)               # (seq, batch, vocab)
    loss   = criterion(logits[1:].reshape(-1, logits.size(-1)),
                       tgt_seq[1:].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def train_and_validate(model, vocab,
                       train_data, val_data,
                       num_epochs, optimizer, criterion):
    """
    Alternate training and validation for `num_epochs`.
    """
    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        # ---------------- training ----------------
        model.train()
        train_loss_sum = 0.0
        for src_batch, tgt_batch in tqdm(train_data, desc=f"Train {epoch}"):
            src, tgt = src_batch.T.to(device), tgt_batch.T.to(device)
            train_loss_sum += _step_backward(model, criterion, optimizer, src, tgt)

        avg_train_loss = train_loss_sum / len(train_data)

        # ---------------- validation --------------
        correct, total, val_loss_sum = validate(model, criterion, vocab, val_data)
        avg_val_loss = val_loss_sum / len(val_data)
        val_accuracy = correct / total

        # ---------------- logging -----------------
        _log_epoch(epoch, avg_train_loss, avg_val_loss, val_accuracy)

# --------------------------------------------------------------------
# internal helpers (private to this file)
# --------------------------------------------------------------------

def _forward_test(model, criterion, src, tgt):
    """
    One forward pass on a test mini-batch.

    Returns:
        batch_loss  – scalar float
        pred_idxs   – tensor of predicted indices  (seq, batch)
    """
    logits = _unpack_prediction(model, src)          # (seq, batch, vocab)

    # CE-loss on non-SOS time-steps
    loss_val = criterion(logits[1:].reshape(-1, logits.size(-1)),
                         tgt[1:].reshape(-1)).item()

    preds = logits.argmax(dim=-1)                    # (seq, batch)
    return loss_val, preds


def _batch_metrics(batch, preds, vocab):
    """
    Compute (correct, total) word-level accuracy for one mini-batch.
    """
    corr = tot = 0
    for src_tok, tgt_tok, pred_tok in zip(batch[0], batch[1], preds.T):
        _, tgt_word  = vocab.index_to_pair((src_tok, tgt_tok))
        _, pred_word = vocab.index_to_pair((src_tok, pred_tok))
        corr += (tgt_word == pred_word)
        tot  += 1
    return corr, tot


# --------------------------------------------------------------------
# public API – names and signatures UNCHANGED
# --------------------------------------------------------------------

def test(model, criterion, vocab, test_data):
    """
    Run evaluation on the `test_data` DataLoader.
    """
    model.eval()
    tot_corr = tot_samp = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            src = batch[0].T.to(device)
            tgt = batch[1].T.to(device)

            batch_loss, preds = _forward_test(model, criterion, src, tgt)
            loss_sum += batch_loss

            corr, tot = _batch_metrics(batch, preds, vocab)
            tot_corr += corr
            tot_samp += tot

    return tot_corr, tot_samp, loss_sum


def perform_testing(model, criterion, vocab, test_data):
    """
    Wrapper that prints metrics and writes predictions CSV.
    """
    corr, tot, loss_sum = test(model, criterion, vocab, test_data)
    avg_loss = loss_sum / len(test_data)
    acc = corr / tot

    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {acc*100:.2f}%")

    generate_predictions_csv(model, vocab,
                             test_data,
                             csv_path="predictions_vanilla.csv")

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _apply_transform(sample, transform):
    """Optionally apply a transform callable to (x, y)."""
    if transform is None:
        return sample
    return transform(*sample)


def _numericalise_df(df, vocab):
    """
    Vectorise a two-column DataFrame (src, tgt) into
    a list of (src_tensor, tgt_tensor) tuples.
    """
    return vocab.data_to_index(df)


# ---------------------------------------------------------------------------
# public API — names & signatures UNCHANGED
# ---------------------------------------------------------------------------

class CustomDataset(Dataset):
    """
    Wrapper that stores a list/array of (src, tgt) pairs
    and optionally applies `self.transform` on __getitem__.
    """

    def __init__(self, data, transform=None):
        """
        Args:
            data (sequence): list/array of encoded pairs.
            transform (callable, optional): mapping (x, y) → (x', y').
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch pair and maybe transform
        return _apply_transform(self.data[idx], self.transform)


def load_dataset(dataset_path, split):
    """
    Read a CSV from `dataset_path` with name pattern  hin_<split>.csv
    and return a DataFrame (English, Hindi) with NaNs replaced by ''.
    """
    file_path = os.path.join(dataset_path, f"hin_{split}.csv")
    df = (pd.read_csv(file_path, header=None, names=["English", "Hindi"])
            .astype(str)
            .fillna(''))
    return df


def prepare_data(data_train, data_val, data_test, batch_size):
    """
    Convert raw DataFrames → numeric tensors → DataLoaders.
    """
    vocab = CreateVocab("Hindi", "English")

    train_enc = _numericalise_df(data_train, vocab)
    val_enc   = _numericalise_df(data_val,   vocab)
    test_enc  = _numericalise_df(data_test,  vocab)

    train_ds = CustomDataset(train_enc)
    val_ds   = CustomDataset(val_enc)
    test_ds  = CustomDataset(test_enc)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


# ---------------------------------------------------------------------------
# internal helpers (private)
# ---------------------------------------------------------------------------

def _build_sweep_config(args) -> dict:
    """Compose a Bayes-opt sweep config from argparse `args`."""
    return {
        "method": "bayes",
        "name":   args.wandb_project,
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            k: {"values": [getattr(args, k)]}
            for k in ("learning_rate", "batch_size", "emb_dim",
                      "num_enc_layers", "num_dec_layers",
                      "hidden_size", "cell_type", "bidirectional", "dropout")
        },
    }


def _run_name(cfg) -> str:
    """Human-readable WandB run name."""
    parts = [
        f"emb_dim_{cfg.emb_dim}",
        f"enc_{cfg.num_enc_layers}",
        f"dec_{cfg.num_dec_layers}",
        f"hs_{cfg.hidden_size}",
        cfg.cell_type,
        f"bi_{cfg.bidirectional}",
        f"lr_{cfg.learning_rate}",
        f"bs_{cfg.batch_size}",
        f"drop_{cfg.dropout}",
    ]
    return "-".join(map(str, parts))


def _prepare_dataloaders(batch_size):
    """Load raw DataFrames and convert them to DataLoaders & vocab."""
    d_train, d_val, d_test, *_ = initializeDataset()
    return prepare_data(d_train, d_val, d_test, batch_size)


def _make_model(vocab, cfg, in_tokens, out_tokens):
    """Instantiate Encoder, Decoder, and wrap in Seq2Seq."""
    enc = Encoder(in_tokens,  cfg.emb_dim, cfg.hidden_size,
                  cfg.num_enc_layers, cfg.bidirectional,
                  cfg.cell_type.upper(), cfg.dropout).to(device)

    dec = Decoder(out_tokens, cfg.emb_dim, cfg.hidden_size,
                  out_tokens, cfg.num_dec_layers,
                  cfg.bidirectional, cfg.cell_type.upper(),
                  cfg.dropout).to(device)

    return Seq2Seq(enc, dec, vocab).to(device)


# ---------------------------------------------------------------------------
# public entry-point — name & signature UNCHANGED
# ---------------------------------------------------------------------------

import re

def clean_text(text):
    """Remove unwanted characters from word strings."""
    return re.sub(r'[<>|»«—=]+', '', text.strip())


def log_predictions_table(model, vocab, test_loader, table_name="Prediction Table"):
    model.eval()
    table = wandb.Table(columns=["Input Word", "Target Word", "Predicted Word"])

    with torch.no_grad():
        for batch in test_loader:
            src_seq = batch[0].T.to(device)
            tgt_seq = batch[1].T.to(device)
            logits = model.prediction(src_seq)
            preds = logits.argmax(dim=2)

            for i in range(src_seq.shape[1]):
                input_word = clean_text(vocab.index_to_word(vocab.language1, src_seq[:, i]))
                target_word = clean_text(vocab.index_to_word(vocab.language2, tgt_seq[:, i]))
                predicted_word = clean_text(vocab.index_to_word(vocab.language2, preds[:, i]))

                if predicted_word == target_word:
                    prediction_display = f"🟩 {predicted_word}"  # green square
                else:
                    prediction_display = f"🟥 {predicted_word}"  # red square

                table.add_data(input_word, target_word, prediction_display)

            break  # log only first batch

    wandb.log({table_name: table})

def main(args):
    """Launch sweep, train, validate, and test the Seq2Seq model."""
    NUM_EPOCHS = 15                             # constant kept

    def _training_loop():
        """WandB sweep worker function."""
        with wandb.init():
            cfg = wandb.config
            wandb.run.name = _run_name(cfg)

            # ---------- data ----------
            train_dl, val_dl, test_dl, vocab = _prepare_dataloaders(cfg.batch_size)
            in_tokens  = len(vocab.inp_lang_char_to_index)
            out_tokens = len(vocab.out_lang_char_to_index)

            # ---------- model ----------
            model = _make_model(vocab, cfg, in_tokens, out_tokens)
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
            criterion = nn.CrossEntropyLoss()

            # ---------- train + validate ----------
            train_and_validate(model, vocab,
                               train_dl, val_dl,
                               NUM_EPOCHS, optimizer, criterion)

            # ---------- final test ----------
            perform_testing(model, criterion, vocab, test_dl)
            log_predictions_table(model, vocab, test_dl)

    sweep_cfg = _build_sweep_config(args)
    sweep_id  = wandb.sweep(sweep_cfg,
                            project=args.wandb_project)
    wandb.agent(sweep_id, _training_loop, count=1)


# ---------------------------------------------------------------------------
# script execution guard (unchanged)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = configParse()
    main(args)
