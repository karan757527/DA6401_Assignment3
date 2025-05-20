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
matplotlib.use('Agg')
import argparse
import csv
from contextlib import nullcontext


import wandb
wandb.login(key="")


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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_enc_layers', type=int, default=3)
    parser.add_argument('--num_dec_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--cell_type', type=str, default='LSTM')
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='Nadam')
    parser.add_argument('--epochs', type=int, default=15)

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

def _tensor_to_sentence(tensor, lang, vocab):
    """Convert a 1-D tensor into a plain-text string using `vocab`."""
    return vocab.index_to_word(lang, tensor)

def _gather_rows(batch, device):
    """
    Convert a (src, tgt) batch from DataLoader into (src_seq, tgt_seq) tensors
    on the right device, each with shape (seq_len, batch_sz).
    """
    src_seq = batch[0].T.to(device)
    tgt_seq = batch[1].T.to(device)
    return src_seq, tgt_seq

def _predict_batch(model, src_seq):
    """
    Run forward prediction; unwrap the (output, attention) tuple and return the
    arg-max indices with shape (seq_len, batch_sz).
    """
    logits, _ = model.prediction(src_seq)
    return logits.argmax(dim=2)          # (seq_len, batch)

def _accumulate_examples(src_seq, tgt_seq, pred_seq, vocab):
    """
    Yield (input_word, true_word, pred_word) triples for every item
    in the current batch.
    """
    batch_sz = src_seq.shape[1]
    for idx in range(batch_sz):
        yield (_tensor_to_sentence(src_seq[:, idx], vocab.language1, vocab),
               _tensor_to_sentence(tgt_seq[:, idx], vocab.language2, vocab),
               _tensor_to_sentence(pred_seq[:, idx], vocab.language2, vocab))

# ---------------------------------------------------------------------------
# public API – signature MUST remain unchanged
# ---------------------------------------------------------------------------

def generate_predictions_csv(model, vocab, test_data, csv_path="predictions.csv"):
    """
    Write a CSV containing ⟨input, target, prediction⟩ for every example
    in `test_data`.

    Args:
        model (Seq2Seq): trained sequence-to-sequence model.
        vocab (CreateVocab): vocabulary instance for (de)tokenisation.
        test_data (DataLoader): DataLoader over the test split.
        csv_path (str): destination file.

    Returns:
        None
    """

    model.eval()

    # Collect rows incrementally instead of building a huge intermediate list
    with open(csv_path, "w", newline="", encoding="utf-8") as fh, \
         torch.no_grad() if torch.is_grad_enabled() else nullcontext():

        writer = csv.writer(fh)
        writer.writerow(["Input", "True Output", "Predicted Output"])

        for batch in tqdm(test_data, desc="Generating CSV"):
            src_seq, tgt_seq = _gather_rows(batch, device)
            pred_seq = _predict_batch(model, src_seq)

            writer.writerows(_accumulate_examples(src_seq, tgt_seq, pred_seq, vocab))

    print(f"\n✅ CSV written to {csv_path}")
# -------------- Define the Encoder and Decoder classes ------------------->

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
        # Call parent constructor
        super().__init__()
        
        # Store parameters
        self.input_dimension = inp_dim
        self.embed_dimension = emb_dim
        self.hidden_dimension = hidden_size
        self.layer_count = num_layers
        self.is_bidirectional = bidirectional
        self.rnn_type = cell_type
        
        # Create embedding layer and dropout layer
        self.word_embedding = nn.Embedding(inp_dim, emb_dim)
        self.regularization = nn.Dropout(dropout)
        
        # Calculate effective dropout based on layer count
        effective_dropout = 0 if num_layers <= 1 else dropout
        
        # Initialize the appropriate RNN type
        if self.rnn_type == "LSTM":
            self.rnn_layer = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=effective_dropout
            )
        elif self.rnn_type == "RNN":
            self.rnn_layer = nn.RNN(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=effective_dropout
            )
        elif self.rnn_type == "GRU":
            self.rnn_layer = nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=effective_dropout
            )

    def forward(self, x):
        '''
            Defines the forward pass of the encoder.

            Args:
            - x (tensor): Input tensor.

            Returns:
            - outputs (tensor): Output tensor from the RNN.
            - hidden (tensor): Hidden state tensor.
            - cell (tensor): Cell state tensor (only for LSTM).
        '''
        # Generate embeddings and apply dropout
        embedded_sequence = self.regularization(self.word_embedding(x))
        
        # Process through the RNN layer
        if self.rnn_type == "LSTM":
            # For LSTM, we get both hidden state and cell state
            rnn_outputs, (final_hidden, final_cell) = self.rnn_layer(embedded_sequence)
            # Transpose embedding for consistent dimensionality
            embedded_sequence = embedded_sequence.permute(1, 0, 2)
            return rnn_outputs, final_hidden, final_cell
        else:
            # For RNN and GRU, we only get hidden state
            rnn_outputs, final_hidden = self.rnn_layer(embedded_sequence)
            # Transpose embedding for consistent dimensionality  
            embedded_sequence = embedded_sequence.permute(1, 0, 2)
            # Return None for cell state to maintain interface consistency
            return rnn_outputs, final_hidden, None


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
            - bidirectional (bool): Whether the decoder is bidirectional or not.
            - cell_type (str): Type of RNN cell (LSTM, RNN, GRU).
            - dropout (float): Dropout probability.

            Returns:
            - None
        '''
        # Initialize parent class
        super().__init__()
        
        # Store configuration parameters
        self.input_dimension = inp_dim
        self.embed_dimension = emb_dim
        self.hidden_dimension = hidden_size
        self.vocab_size = output_size
        self.layer_count = num_layers
        self.is_bidirectional = bidirectional
        self.rnn_type = cell_type
        self.direction_factor = 2 if bidirectional else 1
        
        # Create layers
        self.word_embedding = nn.Embedding(inp_dim, emb_dim)
        self.regularization = nn.Dropout(dropout)
        
        # Effective dropout rate depends on layer count
        effective_dropout = 0 if num_layers <= 1 else dropout
        
        # Calculate input size for RNN cell (includes embedding + context)
        rnn_input_size = hidden_size * self.direction_factor + emb_dim
        
        # Initialize the RNN cell based on the specified type
        if self.rnn_type == "LSTM":
            self.rnn_cell = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=effective_dropout
            )
        elif self.rnn_type == "RNN":
            self.rnn_cell = nn.RNN(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=effective_dropout
            )
        elif self.rnn_type == "GRU":
            self.rnn_cell = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=effective_dropout
            )
        
        # Attention mechanism components
        self.attention_scorer = nn.Linear(hidden_size * (self.direction_factor + 1), 1)
        
        # Output projection
        self.projection = nn.Linear(hidden_size * self.direction_factor, output_size)
        
        # Additional layers for processing
        self.activation = nn.ReLU()
        self.attention_normalizer = nn.Softmax(dim=0)
        self.output_layer = nn.Linear(hidden_size * self.direction_factor, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, encoder_states, hidden, cell):
        '''
            Defines the forward pass of the decoder.

            Args:
            - x (tensor): Input tensor.
            - encoder_states (tensor): States from the encoder.
            - hidden (tensor): Hidden state tensor.
            - cell (tensor): Cell state tensor (only for LSTM).

            Returns:
            - predictions (tensor): Output predictions.
            - hidden (tensor): Hidden state tensor.
            - cell (tensor): Cell state tensor (only for LSTM).
            - attention (tensor): Attention weights.
        '''
        # Add sequence dimension if needed
        x_reshaped = x.unsqueeze(0)
        
        # Update output layer to ensure correct dimensions
        self.output_layer = nn.Linear(self.hidden_dimension * self.direction_factor, self.vocab_size)
        
        # Generate embeddings with dropout
        embeddings = self.regularization(self.word_embedding(x_reshaped))
        
        # Get sequence length from encoder states
        seq_length = encoder_states.shape[0]
        
        # Create repeat of hidden state for attention calculation
        hidden_expanded = hidden.repeat(int(seq_length / self.direction_factor), 1, 1)
        
        # Reorder dimensions for calculation
        decoder_hidden_permuted = hidden.permute(1, 0, 2)
        
        # Calculate attention scores
        attention_input = torch.cat((hidden_expanded, encoder_states), dim=2)
        attention_energies = self.activation(self.attention_scorer(attention_input))
        
        # Create expanded hidden state for broadcast
        decoder_hidden_expanded = decoder_hidden_permuted.repeat(1, seq_length, 1)
        
        # Normalize attention scores
        attention_weights = self.attention_normalizer(attention_energies)
        
        # Reorder dimensions for matrix multiplication
        attention_weights_permuted = attention_weights.permute(1, 2, 0)
        encoder_states_permuted = encoder_states.permute(1, 0, 2)
        
        # Calculate context vector using attention weights
        context_vector = torch.bmm(attention_weights_permuted, encoder_states_permuted).permute(1, 0, 2)
        
        # Combine context vector with input embeddings
        rnn_input = torch.cat((context_vector, embeddings), dim=2)
        
        # Process through RNN cell
        if self.rnn_type == "LSTM":
            rnn_output, (new_hidden, new_cell) = self.rnn_cell(rnn_input, (hidden, cell))
        else:  # RNN or GRU
            rnn_output, new_hidden = self.rnn_cell(rnn_input, hidden)
            new_cell = None
        
        # Generate predictions
        logits = self.projection(rnn_output)
        output_squeezed = logits.squeeze(0)
        predictions = self.log_softmax(logits[0])
        
        return predictions, new_hidden, new_cell, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab):
        '''
            Initializes the Seq2Seq model with an encoder, decoder, and vocabulary.

            Args:
            - encoder (nn.Module): Encoder module.
            - decoder (nn.Module): Decoder module.
            - vocab (CreateVocab): Vocabulary object.

            Returns:
            - None
        '''
        # Initialize parent class
        super().__init__()
        
        # Store components
        self.encoder_model = encoder
        self.decoder_model = decoder
        self.vocabulary = vocab

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        '''
            Performs the forward pass of the Seq2Seq model.

            Args:
            - source (tensor): Source input tensor.
            - target (tensor): Target input tensor.
            - teacher_forcing_ratio (float): Probability of teacher forcing.

            Returns:
            - outputs (tensor): Output predictions.
        '''
        # Extract dimensions
        target_length = target.shape[0]
        batch_size = source.shape[1]
        vocab_size = len(self.vocabulary.out_lang_char_to_index)
        
        # Initialize output tensor
        result_sequences = torch.zeros(target_length, batch_size, vocab_size).to(device)
        
        # Process source sequence through encoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder_model(source)
        
        # Initialize decoder input with first token from target
        current_input = target[0]
        
        # Initialize decoder with SOS token
        decoder_input = torch.full((1, batch_size), self.vocabulary.SOS_token_index, dtype=torch.long)
        
        # Generate sequence iteratively
        for time_step in range(1, target_length):
            # Process current token through decoder
            decoder_output, encoder_hidden, encoder_cell, _ = self.decoder_model(
                current_input, encoder_outputs, encoder_hidden, encoder_cell
            )
            
            # Store predictions
            result_sequences[time_step] = decoder_output
            
            # Get predicted token
            predicted_token = decoder_output.argmax(1)
            
            # Apply teacher forcing based on probability
            use_ground_truth = random.random() < teacher_forcing_ratio
            
            # Select next input: either from target (teacher forcing) or from prediction
            current_input = target[time_step] if use_ground_truth else predicted_token
        
        return result_sequences
    
    def calculate_accuracy(self, predicted_batch, target_batch):
        '''
            Calculates the accuracy of the predicted batch.

            Args:
            - predicted_batch (tensor): Predicted batch.
            - target_batch (tensor): Target batch.

            Returns:
            - correct (int): Number of correct predictions.
            - total (int): Total number of predictions.
        '''
        # Initialize counters
        correct_count = 0
        total_count = 0
        
        # Compare predictions with targets
        for idx in range(target_batch.shape[0]):
            # Convert indices to words
            predicted_word = self.vocabulary.index_to_word(self.language2, predicted_batch[idx])
            target_word = self.vocabulary.index_to_word(self.language2, target_batch[idx])
            
            # Increment total counter
            total_count += 1
            
            # Check if prediction matches target
            if predicted_word == target_word:
                correct_count += 1
                
        return correct_count, total_count
    
    def prediction(self, source, attn_weights=False):
        '''
            Performs prediction using the Seq2Seq model.

            Args:
            - source (tensor): Source input tensor.
            - attn_weights (bool): Whether to return attention weights.

            Returns:
            - outputs (tensor): Output predictions.
            - Attention_Weights (tensor): Attention weights if `attn_weights` is True, else None.
        '''
        # Extract batch size
        batch_size = source.shape[1]
        
        # Initialize target tensor
        current_tokens = torch.zeros(1, batch_size).to(device).long()
        
        # Get vocabulary size
        vocab_size = len(self.vocabulary.out_lang_char_to_index)
        
        # Initialize output tensor
        predicted_sequences = torch.zeros(self.target_len, batch_size, vocab_size).to(device)
        
        # Process source sequence through encoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder_model(source)
        
        # Initialize decoder input with first token
        current_input = current_tokens[0]
        
        # Initialize decoder with SOS token
        decoder_input = torch.full((1, batch_size), self.vocabulary.SOS_token_index, dtype=torch.long)
        
        # Handle attention weights if requested
        if attn_weights:
            # Initialize attention weight storage
            attention_matrix = torch.zeros([batch_size, self.target_len, self.target_len]).to(device)
            
            # Generate sequence iteratively with attention tracking
            for time_step in range(1, self.target_len):
                # Process current token through decoder
                decoder_output, encoder_hidden, encoder_cell, step_attention = self.decoder_model(
                    current_input, encoder_outputs, encoder_hidden, encoder_cell
                )
                
                # Store predictions
                predicted_sequences[time_step] = decoder_output
                
                # Get predicted token for next input
                predicted_token = decoder_output.argmax(1)
                current_input = predicted_token
                
                # Store attention weights for this step
                attention_matrix[:, :, time_step] = step_attention.permute(1, 0, 2).squeeze()
                
        else:
            # No need to track attention weights
            attention_matrix = None
            
            # Generate sequence iteratively without attention tracking
            for time_step in range(1, self.target_len):
                # Process current token through decoder
                decoder_output, encoder_hidden, encoder_cell, _ = self.decoder_model(
                    current_input, encoder_outputs, encoder_hidden, encoder_cell
                )
                
                # Store predictions
                predicted_sequences[time_step] = decoder_output
                
                # Get predicted token for next input
                predicted_token = decoder_output.argmax(1)
                current_input = predicted_token
                
        return predicted_sequences, attention_matrix   
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

# ------------------------------------------------------------------
# internal helpers (private to this module)
# ------------------------------------------------------------------

def _predict_and_loss(model, criterion, src, tgt):
    """
    Run prediction, compute CE-loss on all non-SOS tokens and
    return (pred_indices, batch_loss).
    """
    # model.prediction returns (logits, attn); we need logits only
    logits, _ = model.prediction(src)          # shape: (seq, batch, vocab)

    # flatten for CE-loss
    flat_logits  = logits[1:].reshape(-1, logits.size(-1))
    flat_targets = tgt[1:].reshape(-1)

    batch_loss = criterion(flat_logits, flat_targets)

    # arg-max along vocab dimension, handling 2-or-3-D safety
    if logits.dim() == 3:
        preds = logits.argmax(dim=2)           # (seq, batch)
    else:  # rare fallback
        preds = logits.argmax(dim=1).unsqueeze(0)

    return preds, batch_loss.item()


def _epoch_log(epoch_idx, train_loss, val_loss, val_acc):
    """Print and log metrics to Weights & Biases."""
    print(f"[Epoch {epoch_idx:02}] "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  "
          f"val_acc={val_acc*100:.2f}%")

    wandb.log(
        {"train_loss": train_loss,
         "val_loss":   val_loss,
         "val_accuracy": val_acc * 100}
    )

# ------------------------------------------------------------------
# public functions – names & signatures unchanged
# ------------------------------------------------------------------

def validate(model, criterion, vocab, val_data):
    """
    Evaluate on validation set; return (correct, total, loss_sum).
    """

    model.eval()
    correct = total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(val_data, desc="Validating"):
            src, tgt = src_batch.T.to(device), tgt_batch.T.to(device)
            preds, batch_loss = _predict_and_loss(model, criterion, src, tgt)
            loss_sum += batch_loss
            correct, total = return_accurate((src_batch, tgt_batch),
                                             preds, vocab, correct, total)

    return correct, total, loss_sum


def train_and_validate(model, vocab, train_data,
                       val_data, num_epochs, optimizer, criterion):
    """
    Run `num_epochs` loops of training + validation.
    """

    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        # ------------------ training ------------------
        model.train()
        train_loss_sum = 0.0
        for src_batch, tgt_batch in tqdm(train_data, desc=f"Train {epoch}"):
            src, tgt = src_batch.T.to(device), tgt_batch.T.to(device)
            train_loss_sum += _step_backward(model, criterion, optimizer,
                                             src, tgt)

        train_loss_avg = train_loss_sum / len(train_data)

        # ------------------ validation ----------------
        correct, total, val_loss_sum = validate(model, criterion, vocab, val_data)
        val_loss_avg = val_loss_sum / len(val_data)
        val_accuracy = correct / total

        _epoch_log(epoch, train_loss_avg, val_loss_avg, val_accuracy)


# -------------------------------------------------------------------
# internal helpers (private to this module)
# -------------------------------------------------------------------

def _forward_inference(model, criterion, src, tgt):
    """
    One forward pass on test mini-batch.
    Returns:
        batch_loss  – scalar float
        preds       – tensor (seq_len, batch) of arg-max indices
    """
    logits, _ = model.prediction(src)                     # (seq, batch, vocab)
    flat_logits  = logits[1:].reshape(-1, logits.size(-1))
    flat_targets = tgt[1:].reshape(-1)
    loss_val     = criterion(flat_logits, flat_targets).item()
    preds        = logits.argmax(dim=-1)                  # (seq, batch)
    return loss_val, preds


def _batch_accuracy(batch, preds, vocab):
    """
    Compute word-level accuracy for one mini-batch.
    """
    corr = tot = 0
    for src_tok, tgt_tok, pred_tok in zip(batch[0], batch[1], preds.T):
        _, tgt_word  = vocab.index_to_pair((src_tok, tgt_tok))
        _, pred_word = vocab.index_to_pair((src_tok, pred_tok))
        corr += int(tgt_word == pred_word)
        tot  += 1
    return corr, tot

# -------------------------------------------------------------------
# public API – names & signatures **unchanged**
# -------------------------------------------------------------------

def test(model, criterion, vocab, test_data):
    """
    Evaluate `model` on the test set.
    Returns: (correct, total, loss_sum)
    """
    model.eval()
    tot_corr = tot_samples = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            src = batch[0].T.to(device)
            tgt = batch[1].T.to(device)

            batch_loss, preds = _forward_inference(model, criterion, src, tgt)
            loss_sum += batch_loss

            corr, tot = _batch_accuracy(batch, preds, vocab)
            tot_corr  += corr
            tot_samples += tot

    return tot_corr, tot_samples, loss_sum


def perform_testing(model, criterion, vocab, test_data):
    """
    Wrapper that prints test metrics and writes a CSV of predictions.
    """
    corr, tot, loss_sum = test(model, criterion, vocab, test_data)
    avg_loss = loss_sum / len(test_data)
    accuracy = corr / tot

    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy*100:.2f}%")

    generate_predictions_csv(
        model, vocab, test_data,
        csv_path="predictions_attention.csv"
    )

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


import re

def clean_text(text):
    """Remove unwanted characters from word strings."""
    return re.sub(r'[<>|»«—=]+', '', text.strip())
    
def create_heatmap(model, vocab, test_data):

    '''
        Generates attention heatmaps for the test data using the model and logs them to WandB.

        Args:
            model (nn.Module): Trained sequence-to-sequence model.
            vocab (CreateVocab): Vocabulary object.
            test_data (DataLoader): Test data loader.

        Returns:
            None
    '''

    indx = 1
    for batch in test_data:
            inp_data = batch[0].T.to(device)
            output_val, Weight = model.prediction(inp_data,True)
            best_guess = output_val.argmax(2)
            predictions = best_guess.squeeze()
            break
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.flatten()

    # Path to Devanagari font file
    hindi_font_path = '/kaggle/input/nirmala/Nirmala.ttf'
    
    # Load the font
    hindi_font = fm.FontProperties(fname=hindi_font_path)

    for i, ax in enumerate(axes):
        if i < 9:  # Plot only the first 9 heatmaps
            Pairs_P = vocab.index_to_pair((batch[0][i], predictions.T[i]))
            eng_word1 = Pairs_P[0]
            eng_word=clean_text(eng_word1)
            
            predicted_word1 = Pairs_P[1]
            predicted_word=clean_text(predicted_word1)
            print(eng_word, predicted_word)
            Mat = Weight[i].cpu().detach().numpy()[1:len(Pairs_P[1])+1,1:len(Pairs_P[0])+1]
    
            # Plot heatmap
            im = ax.imshow(Mat, cmap='viridis', aspect='auto')
            cbar = fig.colorbar(im, ax=ax)  # Add color bar
    
            # Set x and y tick labels with the custom font
            ax.set_xticks(np.arange(len(eng_word)))
            ax.set_yticks(np.arange(len(predicted_word)))
            ax.set_xticklabels(list(eng_word), fontproperties=hindi_font, fontsize=14)
            ax.set_yticklabels(list(predicted_word), fontproperties=hindi_font, fontsize=14)
            
            # Set title and labels
            ax.set_title(f'English: {eng_word}\nHindi :{predicted_word}', fontproperties=hindi_font, fontsize=22, pad=20)
            
            ax.tick_params(axis='x', which="major", labelsize=22, pad=10)
            ax.tick_params(axis='y', which="major", rotation=-90, labelsize=22, pad=10)
    
            # Adjust color bar font size
            cbar.ax.tick_params(labelsize=18)
            plt.tight_layout()
            wandb.log({"attention_heatmaps" : plt})


def main(args):

    '''
        Main function for training the sequence-to-sequence model with hyperparameter sweep.

        Args:
            args: Command-line arguments.

        Returns:
            None
    '''

    num_epochs = 15           # Number of epochs

    # Sweep configuration
    sweep_config={
        'method':'bayes',
        'name':args.wandb_project,
        'metric' : {
            'name':'val_accuracy',
            'goal':'maximize'
            },
            'parameters':{ 
                'learning_rate' : {
                    'values' : [args.learning_rate]
                },
                'batch_size' : {
                    'values' : [args.batch_size]
                },
                'emb_dim' : {
                    'values' : [args.emb_dim]
                },
                'num_enc_layers' : {
                    'values' : [args.num_enc_layers]
                },
                'num_dec_layers' : {
                    'values' : [args.num_dec_layers]
                },
                'hidden_size' : {
                    'values' : [args.hidden_size]
                },
                'cell_type' : {
                    'values' : [args.cell_type]
                },
                'bidirectional' : {
                    'values' : [args.bidirectional]
                },
                'dropout' : {
                    'values' : [args.dropout]
                }
            }
        }

    
    # Function to run the sweep configuration on training and validation datasets
    def training():
        '''
            Function for training the sequence-to-sequence model and logging the plots in WandB.

            Returns:
                None
        '''
        with wandb.init():
            config = wandb.config
            wandb.run.name='emb_dim_'+str(wandb.config.emb_dim)+'_num_enc_layers_'+str(wandb.config.num_enc_layers)+'_num_dec_layers_'+str(wandb.config.num_dec_layers)+'_hs_'+str(wandb.config.hidden_size)+'_cell_type_'+config.cell_type+'_bidirectional_'+str(config.bidirectional)+'_lr_'+str(config.learning_rate)+'_bs_'+str(config.batch_size)+'_dropout_'+str(config.dropout)
            
            # Defining hyperparameters
            learning_rate = config.learning_rate
            batch_size = config.batch_size
            data_train, data_val, data_test, input_char_to_index, output_char_to_index, SOS_token_index, EOS_token_index = initializeDataset()
            input_encoder = len(input_char_to_index)
            input_decoder = len(output_char_to_index)
            output_size = len(output_char_to_index)
            
            train_dataset, valid_dataset, test_dataset, vocab = prepare_data(data_train, data_val, data_test, batch_size)

            num_enc_layers = config.num_enc_layers
            num_dec_layers = config.num_dec_layers
            num_enc_layers = 1
            emb_dim = config.emb_dim
            hidden_size = config.hidden_size
            bidirectional = config.bidirectional
            cell_type = config.cell_type.upper()
            dropout = config.dropout

            # Initialize the encoder and decoder
            encoder = Encoder(input_encoder, emb_dim, hidden_size, 
                            num_enc_layers, bidirectional,cell_type, dropout).to(device)
            decoder = Decoder(input_decoder, emb_dim, hidden_size, 
                            output_size, num_enc_layers, bidirectional, cell_type, dropout).to(device)

            # Initialize the model, optimizer and criterion
            model = Seq2Seq(encoder, decoder, vocab).to(device)
            optimizer = optim.Adam(model.parameters(), lr = learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Train and validate the model
            train_and_validate(model, vocab, train_dataset, valid_dataset, num_epochs, optimizer, criterion)

            # Test the model
            perform_testing(model, criterion, vocab, test_dataset)

            # ---------------- Code for heatmap - uncomment and run when you want to generate heatmaps ---------------->
            
            create_heatmap(model, vocab, test_dataset)

            # --------------------------------------- END ------------------------------------------------------------->

    sweep_id=wandb.sweep(sweep_config, project=args.wandb_project)

    wandb.agent(sweep_id, training, count=1)

if __name__ == "__main__":
    args = configParse()
    main(args)