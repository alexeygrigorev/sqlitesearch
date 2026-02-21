"""
Tokenizer module for text processing.

A custom tokenizer that splits text into tokens and removes stop words.
Follows the same interface as minsearch.Tokenizer for compatibility.

Stemming example with minsearch (optional dependency):
    from minsearch.stemmers import porter_stemmer
    tokenizer = Tokenizer(stop_words='english', stemmer=porter_stemmer)
"""

import re
from pathlib import Path
from typing import Callable, Literal, Optional, Union


StopWordsOption = Union[Literal['english'], set[str]]


def _load_stop_words() -> set[str]:
    """Load stop words from stop_words.txt file."""
    module_dir = Path(__file__).parent
    stop_words_path = module_dir / "stop_words.txt"
    with open(stop_words_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


DEFAULT_ENGLISH_STOP_WORDS = _load_stop_words()


class Tokenizer:
    """
    A custom tokenizer that splits text into tokens, removes stop words,
    and optionally applies stemming.

    Compatible with the minsearch Tokenizer interface.

    Examples:
        >>> tokenizer = Tokenizer(stop_words='english')
        >>> tokenizer.tokenize("the quick brown fox")
        ['quick', 'brown', 'fox']

        >>> tokenizer = Tokenizer(stop_words={'quick', 'brown'})
        >>> tokenizer.tokenize("the quick brown fox")
        ['the', 'fox']

        >>> tokenizer = Tokenizer()  # no stop words
        >>> tokenizer.tokenize("the quick brown fox")
        ['the', 'quick', 'brown', 'fox']

        Stemming with minsearch (pip install minsearch):
        >>> from minsearch.stemmers import porter_stemmer
        >>> tokenizer = Tokenizer(stop_words='english', stemmer=porter_stemmer)
        >>> tokenizer.tokenize("the runners are running quickly")
        ['runner', 'run', 'quickli']
    """

    def __init__(
        self,
        pattern: str = r"[\s\W\d]+",
        stop_words: Optional[StopWordsOption] = None,
        stemmer: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            pattern: Regex pattern to split text on.
            stop_words: Stop words to remove. Can be:
                - None: No stop words removed (default)
                - 'english': Use default English stop words from stop_words.txt
                - set[str]: Custom set of stop words
            stemmer: Optional callable that stems a word (word -> stemmed_word).
                Any function with signature str -> str works. For convenience,
                minsearch provides porter_stemmer, snowball_stemmer, and
                lancaster_stemmer that are directly compatible.
        """
        self.pattern = re.compile(pattern)

        if stop_words == 'english':
            self.stop_words = DEFAULT_ENGLISH_STOP_WORDS
        elif stop_words is None:
            self.stop_words = set()
        else:
            self.stop_words = stop_words

        self.stemmer = stemmer

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the input text and remove stop words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens with stop words removed and stemmed if configured.
        """
        if not text:
            return []

        text = text.lower()
        tokens = []

        for token in self.pattern.split(text):
            if not token:
                continue
            if token in self.stop_words:
                continue
            if self.stemmer:
                token = self.stemmer(token)
            tokens.append(token)

        return tokens
