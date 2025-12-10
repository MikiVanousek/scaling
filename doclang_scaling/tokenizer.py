from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

ALL_CHARACTERS = ['B', 'n', 'Z', ':', '!', '\n', 'W', '.', 'F', 'S', 'V', 'd', 'r', 'D', 'Q', ' ', '9', 's', 'u', 'T', 'z', 'h', 'b', 'R', 'j', 'p', 'P', 'O', '=', 'o', 'k', '"', '-', 'f', '#', 'H', 'K', '6', '?', 'M', '5', '+', 'c', ')', 'E', 'g', '4', '*', 't', 'A', 'N', '1', '3', ',', 'e', 'q', '@', 'a', 'x', '(', 'v', '7', 'i', 'w', 'C', '%', '2', 'L', 'I', '&', 'J', '8', '/', 'm', '0', '$', 'y', 'Y', 'l', 'U', 'X', "'", 'G', ';']

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,  # Beginning of sequence
    "[EOS]": 3,  # End of sequence
}

# Add each character as a token
for idx, char in enumerate(ALL_CHARACTERS):
    vocab[char] = idx + 4  # Start after special tokens

# Step 3: Create the tokenizer
tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

# Set up pre-tokenizer to split on every character
tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")

# Set up decoder
tokenizer.decoder = decoders.WordPiece(prefix="")

# Step 4: Wrap with PreTrainedTokenizerFast for HF compatibility
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
)
