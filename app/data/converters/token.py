from app.constants import PAD_INDEX, PAD_SYMBOL, START_TOKEN_INDEX, START_TOKEN_SYMBOL, TOKEN_EMBEDDING_OFFETSET

class TokenConverter:
    """
    Responsible for:
    * converting tokens to integers and back again
    * doing so in a deterministic way given a list of tokens
    """

    def __init__(self, sentences):
        """
        :type sentences: list of list of str
        """
        self._token2integer, self._integer2token = self._get_token_converters(sentences)

    def _get_token_converters(self, sentences):
        token2integer = {PAD_SYMBOL: PAD_INDEX, START_TOKEN_SYMBOL: START_TOKEN_INDEX}
        integer2token = [PAD_SYMBOL, START_TOKEN_SYMBOL]
        counter = TOKEN_EMBEDDING_OFFETSET
        for sentence in sentences:
            for token in sentence:
                if not token in token2integer:
                    token2integer[token] = counter
                    integer2token.append(token)
                    counter += 1
        return token2integer, integer2token

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

    def tokens(self):
        """
        :rtype: list of str
        """
        return self._integer2token
