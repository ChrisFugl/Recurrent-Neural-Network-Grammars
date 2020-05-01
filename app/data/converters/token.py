from app.constants import PAD_INDEX, PAD_SYMBOL, TOKEN_EMBEDDING_OFFSET
from collections import Counter

class TokenConverter:
    """
    Responsible for:
    * converting tokens to integers and back again
    * doing so in a deterministic way given a list of tokens
    """

    def __init__(self, sentences, unknownified_sentences):
        """
        :type sentences: list of list of str
        :type unknownified_sentences: list of list of str
        """
        self._token2integer, self._integer2token = self._get_token_converters(sentences, unknownified_sentences)
        self._counter = self._get_token_counter(sentences)

    def count(self):
        """
        Count number of unique tokens.

        :rtype: int
        """
        return len(self._integer2token)

    def integer2token(self, integer):
        """
        :type integer: int
        :rtype: str
        """
        return self._integer2token[integer]

    def token2integer(self, token):
        """
        :type token: str
        :rtype: int
        """
        return self._token2integer[token]

    def dict(self):
        """
        :rtype: dict
        """
        return self._token2integer

    def tokens(self):
        """
        :rtype: list of str
        """
        return self._integer2token

    def is_singleton(self, token):
        """
        :type token: str
        :rtype: bool
        """
        return self._counter[token] == 1

    def _get_token_converters(self, sentences, unknownified_sentences):
        token2integer = {PAD_SYMBOL: PAD_INDEX}
        integer2token = [PAD_SYMBOL]
        counter = TOKEN_EMBEDDING_OFFSET

        def add_tokens_in_sentence(sentence, counter):
            for token in sentence:
                if not token in token2integer:
                    token2integer[token] = counter
                    integer2token.append(token)
                    counter += 1
            return counter

        for sentence, unknownified_sentence in zip(sentences, unknownified_sentences):
            counter = add_tokens_in_sentence(sentence, counter)
            counter = add_tokens_in_sentence(unknownified_sentence, counter)

        return token2integer, integer2token

    def _get_token_counter(self, sentences):
        counter = Counter()
        for sentence in sentences:
            for token in sentence:
                counter[token] += 1
        return counter
