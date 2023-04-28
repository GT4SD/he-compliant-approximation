"""Tokenizer for smiles.
Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
"""

import collections
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from rxn.chemutils.tokenization import SMILES_TOKENIZER_PATTERN
from transformers.models.bert import BertTokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SmilesTokenizer(BertTokenizer):
    """Tokenizer for smiles.
    Adapted from https://github.com/huggingface/transformers.
    """

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ) -> None:
        """Initializes a SmilesTokenizer.

        Args:
            vocab_file: path to a SMILES character per line vocabulary file.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: CLS token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
        """
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max(
            [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")]
        )
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.basic_tokenizer = BasicSmilesTokenizer()
        self.init_kwargs["model_max_length"] = self.model_max_length

    @property
    def vocab_size(self) -> int:
        """Gets the vocabulary size.

        Returns:
            size of the vocabulary.
        """
        return len(self.vocab)

    @property
    def vocab_list(self) -> List[str]:
        """Gets a list of all the vocabulary tokens.

        Returns:
            list of all the vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes a text input using the basic smiles tokenizer.

        Args:
            text: a textual input.

        Returns:
            list of tokens.
        """
        split_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to the corresponding index in the vocabulary.

        Args:
            token: a token.

        Returns:
            index corresponding to the token in the vocabulary.
        """
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index to the corresponding token in the vocabulary.

        Args:
            index: an index.

        Returns:
            token corresponding to the index in the vocabulary.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string.

        Args:
            tokens: some tokens.

        Returns:
            untokenized string.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids: List[int]) -> List[int]:
        """Adds special tokens at the extremes of the token ids list.
        A BERT sequence has the following format: [CLS]_id ... token_ids ... [SEP]_id

        Args:
            token_ids: a list of token indexes.

        Returns:
            input token id sequence with added special token to the extremes.
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]) -> List[str]:
        """Adds special tokens at the extremes of the tokens list.
        A BERT sequence has the following format: [CLS] ... tokens ... [SEP]

        Args:
            tokens: a list of tokens.

        Returns:
            input token sequence with added special token to the extremes.
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_sequence_pair(
        self, token_0: List[str], token_1: List[str]
    ) -> List[str]:
        """Adds special tokens at the extremes and between two tokens lists.
        A BERT sequence pair has the following format: [CLS] token_0 [SEP] token_1 [SEP]

        Args:
            token_0: first token sequence.
            token_1: second token sequence.

        Returns:
            concatenated input tokens sequences with special tokens at the extremes and between them.
        """
        sep = [self.sep_token]
        cls = [self.cls_token]
        return cls + token_0 + sep + token_1 + sep

    def add_special_tokens_ids_sequence_pair(
        self, token_ids_0: List[int], token_ids_1: List[int]
    ) -> List[int]:
        """Adds special tokens at the extremes and between two token ids lists.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0: first token indexes sequence.
            token_ids_1: second token indexes sequence.

        Returns:
            concatenated input tokens sequences with special tokens at the extremes and between them.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(
        self, token_ids: List[str], length: int, right: bool = True
    ) -> List[int]:
        """Adds padding tokens until the sequence reaches a length of max_length.

        By  default padding tokens are added to the right of the sequence.

        Args:
            token_ids: token ids sequence.
            length: maximum sequence length.
            right: whether to adding the padding to the right. Defaults to True.

        Returns:
            padded sequence.
        """
        padding = [self.pad_token_id] * (length - len(token_ids))
        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(
        self, vocab_path: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        """Saves the tokenizer's vocabulary to a file.

        Args:
            vocab_path: _description_.
            filename_prefix: unused. Defaults to None.

        Returns:
            path of the file containing the saved vocabulary.
        """

        index = 0
        vocab_file = os.path.join(vocab_path, "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicSmilesTokenizer:
    """Runs basic SMILES tokenization."""

    def __init__(self, regex_pattern: str = SMILES_TOKENIZER_PATTERN) -> None:
        """Initializes a BasicSMILESTokenizer.

        Args:
            regex_pattern: a regular expression pattern. Defaults to SMILES_TOKENIZER_PATTERN.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a SMILES string (with the regex pattern).

        Args:
            text: SMILES string.

        Returns:
            list of tokens.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


def load_vocab(vocab_file: str) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary.

    Args:
        vocab_file: vocabulary file.

    Returns:
        mapping from tokens to integers based on the vocabulary.
    """

    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
