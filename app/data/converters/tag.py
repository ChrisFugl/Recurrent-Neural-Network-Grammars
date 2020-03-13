from app.constants import PAD_INDEX, PAD_SYMBOL, TAG_EMBEDDING_OFFSET

class TagConverter:

    def __init__(self, tags):
        """
        :type tags: list of list of str
        """
        self._tag2integer, self._integer2tag = self._get_tag_converters(tags)

    def _get_tag_converters(self, tags):
        tag2integer = {PAD_SYMBOL: PAD_INDEX}
        integer2tag = [PAD_SYMBOL]
        counter = TAG_EMBEDDING_OFFSET
        for sentence_tags in tags:
            for tag in sentence_tags:
                if not tag in tag2integer:
                    tag2integer[tag] = counter
                    integer2tag.append(tag)
                    counter += 1
        return tag2integer, integer2tag

    def count(self):
        """
        Count number of unique tags.

        :rtype: int
        """
        return len(self._integer2tag)

    def integer2tag(self, integer):
        """
        :type integer: int
        :rtype: str
        """
        return self._integer2tag[integer]

    def tag2integer(self, tag):
        """
        :type tag: str
        :rtype: int
        """
        return self._tag2integer[tag]

    def tags(self):
        """
        :rtype: list of str
        """
        return self._integer2tag
