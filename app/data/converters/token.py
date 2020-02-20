from app.constants import PAD_INDEX, PAD_SYMBOL

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
        self._token2integer = {PAD_SYMBOL: PAD_INDEX}
        self._integer2token = [PAD_SYMBOL]
        counter = PAD_INDEX + 1
        for sentence in sentences:
            for token in sentence:
                if not token in self._token2integer:
                    self._token2integer[token] = counter
                    self._integer2token.append(token)
                    counter += 1

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
