
"""
AutoNex AI — Character-Level GPT SLM
------------------------------------
Loads the trained PyTorch checkpoint from backend/model_files.

Expected checkpoint keys:
    model_state_dict, vocab_size, stoi, itos, config

Public functions:
    load_slm()
    generate_text(prompt, max_new_tokens=80)
"""

from pathlib import Path
from threading import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"

CHECKPOINT_PATH = MODEL_DIR / "autonex_tiny_gpt_checkpoint.pt"

DEVICE = torch.device("cpu")

_LOADED_SLM = None
_LOAD_LOCK = Lock()


# ---------------------------------------------------------
# 2. Causal self-attention head
# ---------------------------------------------------------

class Head(nn.Module):
    """One self-attention head with causal masking."""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-2, -1)
        weights = weights * (key.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        value = self.value(x)
        output = weights @ value

        return output


# ---------------------------------------------------------
# 3. Multi-head self-attention
# ---------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()

        head_size = n_embd // n_head

        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout)
            for _ in range(n_head)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        output = self.proj(output)
        return self.dropout(output)


# ---------------------------------------------------------
# 4. Feed-forward network
# ---------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 5. Transformer block
# ---------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        self.sa = MultiHeadAttention(
            n_head, n_embd, block_size, dropout
        )

        self.ffwd = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------
# 6. AutoNex GPT language model
# ---------------------------------------------------------

class AutoNexGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_embd=128,
        n_head=4,
        n_layer=3,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        token_embeddings = self.token_embedding_table(idx)

        positions = torch.arange(
            sequence_length, device=idx.device
        )

        position_embeddings = self.position_embedding_table(
            positions
        )

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape

            logits = logits.view(
                batch_size * sequence_length, vocab_size
            )

            targets = targets.view(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(logits, targets)

        return logits.view(
            batch_size, sequence_length, -1
        ), loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.8):
        """Generate characters autoregressively."""

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / max(temperature, 0.01)

            probabilities = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(
                probabilities, num_samples=1
            )

            idx = torch.cat((idx, next_idx), dim=1)

        return idx


# ---------------------------------------------------------
# 7. Load the actual saved checkpoint
# ---------------------------------------------------------

def load_slm():
    """Load once and reuse the model for subsequent requests."""

    global _LOADED_SLM

    if _LOADED_SLM is not None:
        return _LOADED_SLM

    with _LOAD_LOCK:
        if _LOADED_SLM is not None:
            return _LOADED_SLM

        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"SLM checkpoint not found: {CHECKPOINT_PATH}"
            )

        # This checkpoint must be your own trusted training output.
        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=DEVICE,
                weights_only=True
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=DEVICE
            )

        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid SLM checkpoint format.")

        required_keys = {
            "model_state_dict",
            "vocab_size",
            "stoi",
            "itos",
            "config"
        }

        missing = required_keys - checkpoint.keys()

        if missing:
            raise ValueError(
                f"Checkpoint is missing keys: {sorted(missing)}"
            )

        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        config = checkpoint["config"]

        # Checkpoint stores itos keys as strings in some versions.
        itos = {int(key): value for key, value in itos.items()}
        stoi = {str(key): int(value) for key, value in stoi.items()}

        vocab_size = int(checkpoint["vocab_size"])

        if len(stoi) != vocab_size or len(itos) != vocab_size:
            raise ValueError(
                "Vocabulary size does not match checkpoint metadata."
            )

        # Use the training configuration rather than guessing.
        model = AutoNexGPT(
            vocab_size=vocab_size,
            block_size=int(config["BLOCK_SIZE"]),
            n_embd=int(config["N_EMBD"]),
            n_head=int(config["N_HEAD"]),
            n_layer=int(config["N_LAYER"]),
            dropout=float(config["DROPOUT"])
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        model.to(DEVICE)
        model.eval()

        _LOADED_SLM = (model, stoi, itos)

        return _LOADED_SLM


# ---------------------------------------------------------
# 8. Generate a continuation using the trained vocabulary
# ---------------------------------------------------------

def generate_text(
    prompt,
    max_new_tokens=80,
    temperature=0.8
):
    """
    Return newly generated text only.

    Characters absent from the trained vocabulary are skipped
    when encoding the prompt because this model has no UNK token.
    """

    model, stoi, itos = load_slm()

    prompt = str(prompt).strip()

    if not prompt:
        prompt = "AutoNex"

    encoded = [
        stoi[character]
        for character in prompt
        if character in stoi
    ]

    if not encoded:
        encoded = [stoi.get(" ", 0)]

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = model.generate(
        input_ids,
        # This model generates one character per token. Comparison answers in
        # the training set are usually around 300 characters long.
        max_new_tokens=max(1, min(int(max_new_tokens), 448)),
        temperature=temperature
    )

    new_tokens = generated_ids[0, len(encoded):].tolist()

    return "".join(
        itos[token]
        for token in new_tokens
        if token in itos
    )
