from app.tasks.task import Task
import hydra
import random
import os

class CreateArtificialDataTask(Task):
    """
    Create an artificial dataset based on the following rules:

    s --> np , vp
    np --> det, n
    vp --> v
    vp --> v, np
    vp --> v, pa, vp
    """

    def __init__(self, train_size, val_size, test_size, max_depth, max_words, save_dir):
        """
        :type train_size: int
        :type val_size: int
        :type test_size: int
        :type max_depth: int
        :type max_words; int
        :type save_dir: str
        """
        super().__init__()
        self._train_size = train_size
        self._val_size = val_size
        self._test_size = test_size
        self._max_depth = max_depth
        self._save_dir = hydra.utils.to_absolute_path(save_dir)
        self._max_words = max_words
        self._determiners = [f'd{i}' for i in range(max_words)]
        self._nouns = [f'n{i}' for i in range(max_words)]
        self._particles = [f'p{i}' for i in range(max_words)]
        self._verbs = [f'v{i}' for i in range(max_words)]

    def run(self):
        n_trees = self._train_size + self._val_size + self._test_size
        trees = self._create_trees(n_trees)
        train = trees[:self._train_size]
        val = trees[self._train_size:self._train_size + self._val_size]
        test = trees[-self._test_size:]
        os.makedirs(self._save_dir, exist_ok=True)
        self._save_file(train, 'train.txt')
        self._save_file(val, 'val.txt')
        self._save_file(test, 'test.txt')

    def _create_trees(self, n_trees):
        trees = set([])
        while len(trees) < n_trees:
            n_remaining_trees = n_trees - len(trees)
            for _ in range(n_remaining_trees):
                tree = self._create_tree()
                trees.add(tree)
        return list(trees)

    def _create_tree(self):
        next_depth = 1
        # s --> np , vp
        noun_phrase, word_index = self._create_noun_phrase()
        verb_phrase = self._create_verb_phrase(next_depth, word_index)
        return f'(S {noun_phrase} {verb_phrase})'

    def _create_noun_phrase(self, word_index=None):
        # np --> det, n
        determiner, word_index = self._create_determiner(word_index)
        noun = self._create_noun(word_index)
        return f'(NP {determiner} {noun})', word_index

    def _create_verb_phrase(self, depth, word_index):
        next_depth = depth + 1
        if next_depth < self._max_depth:
            verb_phrase_type = random.randint(1, 3)
            if verb_phrase_type == 1:
                # vp --> v
                verb = self._create_verb(word_index)
                return f'(VP {verb})'
            elif verb_phrase_type == 2:
                # vp --> v, np
                verb = self._create_verb(word_index)
                noun_phrase, word_index = self._create_noun_phrase(word_index=word_index)
                return f'(VP {verb} {noun_phrase})'
            else:
                # vp --> v, pa, vp
                verb = self._create_verb(word_index)
                particle = self._create_particle(word_index)
                verb_phrase = self._create_verb_phrase(next_depth, word_index)
                return f'(VP {verb} {particle} {verb_phrase})'
        else:
            # vp --> v
            verb = self._create_verb(word_index)
            return f'(VP {verb})'

    def _create_determiner(self, word_index):
        if word_index is None:
            index = random.randint(0, self._max_words - 1)
        else:
            index = (word_index + 1) % self._max_words
        determiner = self._determiners[index]
        return f'(DT {determiner})', index

    def _create_noun(self, index):
        return self._create_leaf('NN', self._nouns, index)

    def _create_particle(self, index):
        return self._create_leaf('PA', self._particles, index)

    def _create_verb(self, index):
        return self._create_leaf('VB', self._verbs, index)

    def _create_leaf(self, prefix, choices, index):
        word = choices[index]
        return f'({prefix} {word})'

    def _save_file(self, sentences, filename):
        file_path = os.path.join(self._save_dir, filename)
        content = '\n'.join(sentences)
        with open(file_path, 'w') as file:
            file.write(content)
