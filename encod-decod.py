# Copyright (c) Brian Ratliff under Apache License 2.0 (see LICENSE.txt).
#
# This file implements an encoder/decoder transformer model based on
# "Attention is All You Need" paper, using the same flexible configuration
# approach as the existing LLM_homebrew code.

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tiktoken # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import logging
from tqdm import tqdm
import sys
from datetime import datetime
import argparse
import os
import re
from functools import partial


#####################################
# Encoder/Decoder Transformer Components
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, is_causal=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.is_causal = is_causal
        self.context_length = context_length

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_value=None):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        
        if key_value is not None:
            # Cross-attention: use provided key_value for keys and values
            keys = self.W_key(key_value)
            values = self.W_value(key_value)
        else:
            # Self-attention: use input x for keys and values
            keys = self.W_key(x)
            values = self.W_value(x)

        # Reshape for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask if needed - create dynamically
        if self.is_causal:
            # Create causal mask dynamically based on actual sequence length
            mask = torch.triu(torch.ones(num_tokens, keys.size(2), device=x.device), diagonal=1)
            mask_bool = mask.bool()
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            is_causal=False  # Encoder uses bidirectional attention
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Self-attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.self_att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # Feed-forward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            is_causal=True  # Decoder uses causal attention
        )
        self.cross_att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            is_causal=False  # Cross-attention is not causal
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.norm3 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, encoder_output):
        # Self-attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.self_att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # Cross-attention with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.cross_att(x, encoder_output)
        x = self.drop_resid(x)
        x = x + shortcut

        # Feed-forward with residual connection
        shortcut = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-np.log(10000.0) / emb_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x):
        # Handle sequences longer than max_len by extending if needed
        seq_len = x.size(0)
        if seq_len > self.max_len:
            # Extend positional encoding if needed
            additional_len = seq_len - self.max_len
            additional_pe = torch.zeros(additional_len, self.pe.size(1), device=x.device)
            position = torch.arange(self.max_len, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(1), 2).float() * (-np.log(10000.0) / self.pe.size(1)))
            
            additional_pe[:, 0::2] = torch.sin(position * div_term)
            additional_pe[:, 1::2] = torch.cos(position * div_term)
            additional_pe = additional_pe.unsqueeze(0).transpose(0, 1)
            
            # Concatenate with existing PE
            extended_pe = torch.cat([self.pe, additional_pe], dim=0)
            return x + extended_pe[:seq_len, :]
        else:
            return x + self.pe[:seq_len, :]


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(cfg["src_vocab_size"], cfg["emb_dim"])
        self.tgt_embedding = nn.Embedding(cfg["tgt_vocab_size"], cfg["emb_dim"])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(cfg["emb_dim"], cfg["context_length"])
        
        # Dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(cfg) for _ in range(cfg["n_encoder_layers"])
        ])
        self.encoder_norm = LayerNorm(cfg["emb_dim"])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg) for _ in range(cfg["n_decoder_layers"])
        ])
        self.decoder_norm = LayerNorm(cfg["emb_dim"])
        
        # Output projection
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["tgt_vocab_size"])

    def encode(self, src):
        # Source embedding and positional encoding
        src_emb = self.src_embedding(src) * np.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        x = self.drop_emb(src_emb)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        return self.encoder_norm(x)

    def decode(self, tgt, encoder_output):
        # Target embedding and positional encoding
        tgt_emb = self.tgt_embedding(tgt) * np.sqrt(self.tgt_embedding.embedding_dim)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        x = self.drop_emb(tgt_emb)
        
        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x, encoder_output)
        
        return self.decoder_norm(x)

    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output)
        return self.out_proj(decoder_output)


#####################################
# Dataset and DataLoader
#####################################

class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_length=512):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        entry = self.data[index]
        
        # Tokenize source and target (tiktoken doesn't support max_length/truncation)
        src_tokens = self.src_tokenizer.encode(entry['source'])
        tgt_tokens = self.tgt_tokenizer.encode(entry['target'])
        
        # Manually truncate if needed
        if len(src_tokens) > self.max_length:
            src_tokens = src_tokens[:self.max_length]
        if len(tgt_tokens) > self.max_length:
            tgt_tokens = tgt_tokens[:self.max_length]
        
        return {
            'src': torch.tensor(src_tokens),
            'tgt': torch.tensor(tgt_tokens),
            'src_len': len(src_tokens),
            'tgt_len': len(tgt_tokens)
        }

    def __len__(self):
        return len(self.data)


def collate_fn(batch, pad_token_id=0, device="cpu"):
    # Find max lengths
    max_src_len = max(item['src_len'] for item in batch)
    max_tgt_len = max(item['tgt_len'] for item in batch)
    
    # Pad sequences
    src_batch = []
    tgt_batch = []
    
    for item in batch:
        src_padded = torch.cat([
            item['src'],
            torch.full((max_src_len - item['src_len'],), pad_token_id, dtype=torch.long)
        ])
        tgt_padded = torch.cat([
            item['tgt'],
            torch.full((max_tgt_len - item['tgt_len'],), pad_token_id, dtype=torch.long)
        ])
        
        src_batch.append(src_padded)
        tgt_batch.append(tgt_padded)
    
    return (
        torch.stack(src_batch).to(device),
        torch.stack(tgt_batch).to(device)
    )


#####################################
# Training Functions
#####################################

def calc_loss_batch(src_batch, tgt_batch, model, device, pad_token_id=0):
    src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
    
    # For training, we use teacher forcing: input is tgt[:-1], target is tgt[1:]
    tgt_input = tgt_batch[:, :-1]
    tgt_target = tgt_batch[:, 1:]
    
    logits = model(src_batch, tgt_input)
    
    # Reshape for loss calculation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = tgt_target.reshape(-1)
    
    # Create mask for padding tokens
    mask = targets_flat != pad_token_id
    
    # Calculate loss only on non-padded tokens
    loss = torch.nn.functional.cross_entropy(
        logits_flat[mask], targets_flat[mask], ignore_index=pad_token_id
    )
    
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (src_batch, tgt_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(src_batch, tgt_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, src_tokenizer, tgt_tokenizer,
                save_checkpoints=False, checkpoint_freq=1, keep_checkpoints=5,
                output_dir=None, config=None, model_size=None, batch_size=None,
                start_epoch=0, resumed_from=None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch
        model.train()

        for src_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(src_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += src_batch.numel() + tgt_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {current_epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample translation after each epoch
        generate_and_print_sample(
            model, src_tokenizer, tgt_tokenizer, device, start_context
        )
        
        # Save checkpoint if enabled and it's time to save
        if save_checkpoints and output_dir and (epoch + 1) % checkpoint_freq == 0:
            try:
                checkpoint_path = save_checkpoint(
                    model, optimizer, train_losses, val_losses, track_tokens_seen,
                    config, model_size, current_epoch + 1, batch_size, output_dir, resumed_from
                )
                print(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")
                
                # Clean up old checkpoints
                cleanup_old_checkpoints(output_dir, keep_checkpoints)
                
            except Exception as e:
                logging.error(f"Error saving checkpoint: {str(e)}")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, src_tokenizer, tgt_tokenizer, device, start_context):
    model.eval()
    
    # Tokenize source
    src_tokens = src_tokenizer.encode(start_context)
    src_tensor = torch.tensor([src_tokens]).to(device)
    
    # Generate target
    with torch.no_grad():
        # Start with start token
        tgt_tokens = [tgt_tokenizer.encode("<START>")[0]]
        
        for _ in range(100):  # Max length
            tgt_tensor = torch.tensor([tgt_tokens]).to(device)
            logits = model(src_tensor, tgt_tensor)
            
            # Get next token
            next_token = torch.argmax(logits[0, -1, :]).item()
            tgt_tokens.append(next_token)
            
            # Stop if end token
            if next_token == tgt_tokenizer.encode("<END>")[0]:
                break
        
        # Decode
        src_text = src_tokenizer.decode(src_tokens)
        tgt_text = tgt_tokenizer.decode(tgt_tokens[1:])  # Remove start token
        
        print(f"Source: {src_text}")
        print(f"Target: {tgt_text}")
    
    model.train()


def generate_translation(model, src_tokenizer, tgt_tokenizer, device, source_text, max_length=100, temperature=1.0):
    """
    Generate a translation for the given source text.
    
    Args:
        model: The trained encoder-decoder transformer model
        src_tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer
        device: Device to run inference on
        source_text: Source text to translate
        max_length: Maximum length of generated translation
        temperature: Temperature for sampling (1.0 = greedy, higher = more random)
    
    Returns:
        Generated translation text
    """
    model.eval()
    
    # Tokenize source
    src_tokens = src_tokenizer.encode(source_text)
    src_tensor = torch.tensor([src_tokens]).to(device)
    
    # Generate target
    with torch.no_grad():
        # Start with start token
        tgt_tokens = [tgt_tokenizer.encode("<START>")[0]]
        
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens]).to(device)
            logits = model(src_tensor, tgt_tensor)
            
            # Get next token with temperature sampling
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            if temperature == 1.0:
                # Greedy decoding
                next_token = torch.argmax(logits).item()
            else:
                # Temperature sampling
                next_token = torch.multinomial(probs, 1).item()
            
            tgt_tokens.append(next_token)
            
            # Stop if end token
            if next_token == tgt_tokenizer.encode("<END>")[0]:
                break
        
        # Decode
        tgt_text = tgt_tokenizer.decode(tgt_tokens[1:])  # Remove start token
        
        return tgt_text
    
    model.train()


def save_checkpoint(model, optimizer, train_losses, val_losses, tokens_seen, config, 
                   model_size, current_epoch, batch_size, output_dir, resumed_from=None):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        train_losses: List of training losses
        val_losses: List of validation losses
        tokens_seen: List of token counts
        config: Model configuration
        model_size: Size of the model (base/large)
        current_epoch: Current epoch number
        batch_size: Training batch size
        output_dir: Directory to save checkpoints
        resumed_from: Path to checkpoint this was resumed from (if any)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen': tokens_seen,
        'config': config,
        'model_size': model_size,
        'epoch': current_epoch,
        'batch_size': batch_size,
        'resumed_from': resumed_from,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{current_epoch:03d}.pth")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def cleanup_old_checkpoints(output_dir, keep_count):
    """
    Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_count: Number of recent checkpoints to keep (0 to keep all)
    """
    if keep_count <= 0:
        return  # Keep all checkpoints
    
    # Find all checkpoint files
    checkpoint_files = []
    for file in os.listdir(output_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
            checkpoint_files.append(file)
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Remove old checkpoints
    files_to_remove = checkpoint_files[:-keep_count] if len(checkpoint_files) > keep_count else []
    
    for file in files_to_remove:
        file_path = os.path.join(output_dir, file)
        try:
            os.remove(file_path)
            logging.info(f"Removed old checkpoint: {file}")
        except Exception as e:
            logging.warning(f"Failed to remove old checkpoint {file}: {str(e)}")


def interactive_generation(model, src_tokenizer, tgt_tokenizer, device, model_path):
    """
    Interactive generation mode where user can input source text and get translations.
    """
    print(f"Loaded model from: {model_path}")
    print("Enter source text to translate (or 'quit' to exit):")
    
    while True:
        try:
            source_text = input("\nSource: ").strip()
            
            if source_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not source_text:
                continue
            
            # Generate translation
            translation = generate_translation(
                model, src_tokenizer, tgt_tokenizer, device, source_text
            )
            
            print(f"Translation: {translation}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("transformer-loss-plot.pdf")
    plt.show()


#####################################
# Main Training Script
#####################################

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train encoder/decoder transformer model')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'generate'],
        default='train',
        help='Mode of operation: train or generate'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default="slot_endec.json",
        help='Input JSON file containing translation data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="transformer_output",
        help='Directory to save the trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Training batch size'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default="base",
        choices=["base", "large"],
        help='Size of transformer model to use'
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=512,
        help='Maximum context length for the model'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to the trained model checkpoint (required for generate mode)'
    )
    parser.add_argument(
        '--source_text',
        type=str,
        help='Source text to translate (for single generation)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of generated translation'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling (1.0 = greedy, higher = more random)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode for multiple translations'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--save_checkpoints',
        action='store_true',
        help='Save checkpoints during training'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=1,
        help='Save checkpoint every N epochs (default: 1)'
    )
    parser.add_argument(
        '--keep_checkpoints',
        type=int,
        default=5,
        help='Number of recent checkpoints to keep (default: 5, 0 to keep all)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'transformer_training_{datetime.now():%Y%m%d_%H%M%S}.log')
        ]
    )

    # Load data
    try:
        with open(args.input_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        logging.info(f"Successfully loaded {len(data)} translation pairs from {args.input_file}")

        # Split data
        train_portion = int(len(data) * 0.8)
        val_portion = int(len(data) * 0.1)
        
        train_data = data[:train_portion]
        val_data = data[train_portion:train_portion + val_portion]
        test_data = data[train_portion + val_portion:]

        logging.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        sys.exit(1)

    # Setup tokenizers (using GPT-2 for simplicity)
    try:
        src_tokenizer = tiktoken.get_encoding("gpt2")
        tgt_tokenizer = tiktoken.get_encoding("gpt2")
        logging.info("Successfully initialized tokenizers")
    except Exception as e:
        logging.error(f"Error initializing tokenizers: {str(e)}")
        sys.exit(1)

    # Device setup
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            logging.info("Using Apple Silicon (MPS)")
        else:
            device = torch.device("cpu")
            logging.warning("No GPU found, using CPU")
    except Exception as e:
        logging.error(f"Error setting up device: {str(e)}")
        sys.exit(1)

    # Setup data loaders
    try:
        customized_collate_fn = partial(collate_fn, device=device)
        
        train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, max_length=args.context_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=customized_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, max_length=args.context_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        logging.info("Data loaders setup complete")
    except Exception as e:
        logging.error(f"Error setting up data loaders: {str(e)}")
        sys.exit(1)

    # Model configuration
    BASE_CONFIG = {
        "src_vocab_size": 50257,  # Source vocabulary size
        "tgt_vocab_size": 50257,  # Target vocabulary size
        "context_length": args.context_length,     # Context length
        "drop_rate": 0.1,          # Dropout rate
        "qkv_bias": True           # Query-key-value bias
    }

    model_configs = {
        "base": {
            "emb_dim": 512,
            "n_encoder_layers": 6,
            "n_decoder_layers": 6,
            "n_heads": 8
        },
        "large": {
            "emb_dim": 1024,
            "n_encoder_layers": 12,
            "n_decoder_layers": 12,
            "n_heads": 16
        }
    }

    BASE_CONFIG.update(model_configs[args.model_size])
    logging.info(f"Model config: {BASE_CONFIG}")

    # Create model
    try:
        model = EncoderDecoderTransformer(BASE_CONFIG)
        model.to(device)
        logging.info("Model setup complete")
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        sys.exit(1)

    if args.mode == 'train':
        # Create output directory
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logging.info(f"Created/verified output directory: {args.output_dir}")
        except Exception as e:
            logging.error(f"Error creating output directory: {str(e)}")
            sys.exit(1)

        # Training setup
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-7)
            num_epochs = args.epochs
            start_epoch = 0
            train_losses, val_losses, tokens_seen = [], [], []
            
            # Resume from checkpoint if specified
            if args.resume_from:
                try:
                    logging.info(f"Resuming training from checkpoint: {args.resume_from}")
                    checkpoint = torch.load(args.resume_from, map_location=device)
                    
                    # Load model state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info("Loaded model state from checkpoint")
                    
                    # Load optimizer state
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Loaded optimizer state from checkpoint")
                    
                    # Load training history
                    if 'train_losses' in checkpoint:
                        train_losses = checkpoint['train_losses']
                        logging.info(f"Loaded {len(train_losses)} previous training loss values")
                    if 'val_losses' in checkpoint:
                        val_losses = checkpoint['val_losses']
                        logging.info(f"Loaded {len(val_losses)} previous validation loss values")
                    if 'tokens_seen' in checkpoint:
                        tokens_seen = checkpoint['tokens_seen']
                        logging.info(f"Loaded {len(tokens_seen)} previous token counts")
                    
                    # Get starting epoch (if available)
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch']
                        logging.info(f"Resuming from epoch {start_epoch + 1}")
                    
                    logging.info("Successfully resumed from checkpoint")
                    
                except Exception as e:
                    logging.error(f"Error loading checkpoint: {str(e)}")
                    sys.exit(1)
            
            logging.info("Starting training...")
            logging.info(f"Training config - Epochs: {num_epochs}, Batch size: {args.batch_size}")
            if args.resume_from:
                logging.info(f"Resuming from epoch {start_epoch + 1}")
            if args.save_checkpoints:
                logging.info(f"Checkpoint saving enabled - Frequency: every {args.checkpoint_freq} epoch(s), Keep: {args.keep_checkpoints} recent checkpoints")
            else:
                logging.info("Checkpoint saving disabled")

            ##### change "start_context" to something more relevant #####
            new_train_losses, new_val_losses, new_tokens_seen = train_model(
                model, train_loader, val_loader, optimizer, device,
                num_epochs=num_epochs, eval_freq=10, eval_iter=5,
                start_context="SLOT WIDTH=50.00 HEIGHT=25.00\n#1 = CARTESIAN_POINT('NONE',(0.00, 0.00, 0.00));\n#2 = CARTESIAN_POINT('NONE',(100.00, 0.00, 0.00));", 
                src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer,
                save_checkpoints=args.save_checkpoints,
                checkpoint_freq=args.checkpoint_freq,
                keep_checkpoints=args.keep_checkpoints,
                output_dir=args.output_dir,
                config=BASE_CONFIG,
                model_size=args.model_size,
                batch_size=args.batch_size,
                start_epoch=start_epoch,
                resumed_from=args.resume_from
            )
            
            # Combine with previous training history
            train_losses.extend(new_train_losses)
            val_losses.extend(new_val_losses)
            tokens_seen.extend(new_tokens_seen)
            
            logging.info("Training complete")

            # Save model
            model_save_path = os.path.join(args.output_dir, f"transformer-{args.model_size}-finetuned.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'tokens_seen': tokens_seen,
                'config': BASE_CONFIG,
                'model_size': args.model_size,
                'epochs': num_epochs,
                'batch_size': args.batch_size,
                'epoch': start_epoch + num_epochs,  # Save current epoch
                'resumed_from': args.resume_from if args.resume_from else None
            }, model_save_path)
            logging.info(f"Saved model to {model_save_path}")

            # Save metrics
            metrics_save_path = os.path.join(args.output_dir, "training_metrics.json")
            metrics = {
                'final_train_loss': float(train_losses[-1]),
                'final_val_loss': float(val_losses[-1]),
                'total_tokens_seen': int(tokens_seen[-1]),
                'total_epochs': start_epoch + num_epochs,
                'epochs_this_run': num_epochs,
                'batch_size': args.batch_size,
                'model_size': args.model_size,
                'resumed_from': args.resume_from if args.resume_from else None
            }
            with open(metrics_save_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Saved metrics to {metrics_save_path}")

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            sys.exit(1)

        # Plot losses
        try:
            total_epochs = start_epoch + num_epochs
            epochs_tensor = torch.linspace(0, total_epochs, len(train_losses))
            plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
            
            plot_save_path = os.path.join(args.output_dir, "training_loss_plot.png")
            plt.savefig(plot_save_path)
            plt.close()
            logging.info(f"Saved loss plot to {plot_save_path}")
        except Exception as e:
            logging.error(f"Error generating plots: {str(e)}")

    elif args.mode == 'generate':
        # Check if model path is provided
        if not args.model_path:
            logging.error("--model_path is required for generate mode")
            sys.exit(1)
        
        # Load trained model
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Successfully loaded model from {args.model_path}")
            
            # Log model info
            if 'config' in checkpoint:
                logging.info(f"Model config: {checkpoint['config']}")
            if 'model_size' in checkpoint:
                logging.info(f"Model size: {checkpoint['model_size']}")
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            sys.exit(1)
        
        # Setup tokenizers (using GPT-2 for consistency)
        try:
            src_tokenizer = tiktoken.get_encoding("gpt2")
            tgt_tokenizer = tiktoken.get_encoding("gpt2")
            logging.info("Successfully initialized tokenizers")
        except Exception as e:
            logging.error(f"Error initializing tokenizers: {str(e)}")
            sys.exit(1)
        
        if args.interactive:
            # Interactive generation mode
            interactive_generation(model, src_tokenizer, tgt_tokenizer, device, args.model_path)
        elif args.source_text:
            # Single generation mode
            try:
                translation = generate_translation(
                    model, src_tokenizer, tgt_tokenizer, device, 
                    args.source_text, args.max_length, args.temperature
                )
                print(f"Source: {args.source_text}")
                print(f"Translation: {translation}")
            except Exception as e:
                logging.error(f"Error during generation: {str(e)}")
                sys.exit(1)
        else:
            logging.error("For generate mode, either --interactive or --source_text must be provided")
            sys.exit(1)

    logging.info("Script execution completed")


if __name__ == "__main__":
    main() 