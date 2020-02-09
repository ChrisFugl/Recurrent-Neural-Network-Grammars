from app.tasks.task import Task
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

    def __init__(self, train_size, val_size, test_size, max_depth, max_determiners, max_nouns, max_particles, max_verbs, save_dir):
        """
        :type train_size: int
        :type val_size: int
        :type test_size: int
        :type max_depth: int
        :type max_determiners: int
        :type max_nouns: int
        :type max_particles: int
        :type max_verbs: int
        :type save_dir: str
        """
        self._train_size = train_size
        self._val_size = val_size
        self._test_size = test_size
        self._max_depth = max_depth
        self._save_dir = save_dir
        self._determiners = [f'd{i}' for i in range(max_determiners)]
        self._nouns = [f'n{i}' for i in range(max_nouns)]
        self._particles = [f'p{i}' for i in range(max_particles)]
        self._verbs = [f'v{i}' for i in range(max_verbs)]

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
        noun_phrase = self._create_noun_phrase()
        verb_phrase = self._create_verb_phrase(next_depth)
        return f'( {noun_phrase} {verb_phrase})'

    def _create_noun_phrase(self):
        # np --> det, n
        determiner = self._create_determiner()
        noun = self._create_noun()
        return f'(NP {determiner} {noun})'

    def _create_verb_phrase(self, depth):
        next_depth = depth + 1
        if next_depth < self._max_depth:
            verb_phrase_type = random.randint(1, 3)
            if verb_phrase_type == 1:
                # vp --> v
                verb = self._create_verb()
                return verb
            elif verb_phrase_type == 2:
                # vp --> v, np
                verb = self._create_verb()
                noun_phrase = self._create_noun_phrase()
                return f'(VP {verb} {noun_phrase})'
            else:
                # vp --> v, pa, vp
                verb = self._create_verb()
                particle = self._create_particle()
                verb_phrase = self._create_verb_phrase(next_depth)
                return f'(VP {verb} {particle} {verb_phrase})'
        else:
            # vp --> v
            verb = self._create_verb()
            return verb

    def _create_determiner(self):
        determiner = random.choice(self._determiners)
        return f'(DT {determiner})'

    def _create_noun(self):
        return self._create_leaf('NN', self._nouns)

    def _create_particle(self):
        return self._create_leaf('PA', self._particles)

    def _create_verb(self):
        return self._create_leaf('VB', self._verbs)

    def _create_leaf(self, prefix, choices):
        word = random.choice(choices)
        return f'({prefix} {word})'

    def _save_file(self, sentences, filename):
        file_path = os.path.join(self._save_dir, filename)
        content = '\n'.join(sentences)
        with open(file_path, 'w') as file:
            file.write(content)
