from app.constants import ACTION_REDUCE_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE
from app.data.loaders.loader import Loader
from app.data.preprocessing.non_terminals import get_non_terminal_identifier
from app.data.preprocessing.terminals import get_terminal_node, is_start_of_terminal_node
import hydra
import os
import re

class PennLoader(Loader):
    """
    Assumes that data directory has the following structure: sections/{section number}/{filename}
    """

    _metatag_pattern = re.compile('<\/?\S+( *\S+=\S+)* *>')
    _tree_start_pattern = re.compile('\(\s*\(')

    def __init__(self, data_dir, train_sections, val_sections, test_sections):
        """
        :type data_dir: str
        :type train_sections: list of int
        :type val_sections: list of int
        :type test_sections: list of int
        """
        absolute_data_dir = hydra.utils.to_absolute_path(data_dir)
        self._sections_dir = os.path.join(absolute_data_dir, 'sections')
        self._train_sections = train_sections
        self._val_sections = val_sections
        self._test_sections = test_sections

    def load_train(self):
        """
        :rtype: list of str
        :returns: trees
        """
        return self._load(self._train_sections)

    def load_val(self):
        """
        :rtype: list of str
        :returns: trees
        """
        return self._load(self._val_sections)

    def load_test(self):
        """
        :rtype: list of str
        :returns: trees
        """
        return self._load(self._test_sections)

    def _load(self, sections):
        trees = []
        for section_number in sections:
            padding = 0
            while not os.path.exists(self._get_section_dir(section_number, padding)):
                padding += 1
            section_dir = self._get_section_dir(section_number, padding)
            file_names = sorted(os.listdir(section_dir))
            for file_name in file_names:
                file_path = os.path.join(section_dir, file_name)
                file_trees = self._read_file(file_path)
                trees.extend(file_trees)
        return trees

    def _get_section_dir(self, section_number, padding):
        section_name = f'{section_number:0{padding}d}'
        section_dir = os.path.join(self._sections_dir, section_name)
        return section_dir

    def _read_file(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
        line_count = len(lines)
        line_index = 0
        trees = []
        while line_index < line_count:
            line = lines[line_index].strip()
            if self._tree_start_pattern.match(line) is not None:
                tree_lines = [line]
                line_index += 1
                while line_index < line_count:
                    line = lines[line_index].strip()
                    if self._tree_start_pattern.match(line) is not None:
                        break
                    elif self._metatag_pattern.match(line) is None: # ignore metatag lines
                        tree_lines.append(line)
                    line_index += 1
                tree = ' '.join(tree_lines)
                trees.append(tree)
            else:
                line_index += 1
        trees_without_none_tags = list(map(self._remove_none_tags, trees))
        return trees_without_none_tags

    def _remove_none_tags(self, line):
        actions = self._line2actions(line)
        tree = self._actions2tree(actions)
        trees_without_none_tags = self._remove_none_tags_from_tree(tree)
        trees_without_empty_non_terminals  = self._remove_non_terminal_leafs(trees_without_none_tags)
        line_without_none_tags = self._tree2line(trees_without_empty_non_terminals)
        return line_without_none_tags

    def _line2actions(self, line):
        actions = []
        line_index = 0
        token_index = 0
        while line_index != -1:
            if line[line_index] == ')':
                action = ACTION_REDUCE_TYPE, None
                search_index = line_index + 1
            elif is_start_of_terminal_node(line, line_index):
                tag, word, terminal_end_index = get_terminal_node(line, line_index)
                action = ACTION_GENERATE_TYPE, (tag, word)
                token_index += 1
                search_index = terminal_end_index + 1
            else:
                non_terminal = get_non_terminal_identifier(line, line_index)
                action = ACTION_NON_TERMINAL_TYPE, non_terminal
                search_index = line_index + 1
            line_index = self._get_next_action_start_index(line, search_index)
            actions.append(action)
        return actions

    def _get_next_action_start_index(self, line, line_index):
        next_open_bracket_index = line.find('(', line_index)
        next_close_bracket_index = line.find(')', line_index)
        if next_open_bracket_index == -1 or next_close_bracket_index == -1:
            return max(next_open_bracket_index, next_close_bracket_index)
        else:
            return min(next_open_bracket_index, next_close_bracket_index)

    def _actions2tree(self, actions):
        # first action creates the root of the tree
        root = TreeNode(value=actions[0])
        parent = root
        for action in actions[1:]:
            type, value = action
            if type == ACTION_REDUCE_TYPE:
                parent = parent.parent
            elif type == ACTION_GENERATE_TYPE:
                node = TreeNode(action, parent=parent)
                parent.add_child(node)
            else:
                node = TreeNode(action, parent=parent)
                parent.add_child(node)
                parent = node
        return root

    def _remove_none_tags_from_tree(self, tree):
        root = TreeNode(tree.value)
        if tree.children is None:
            return root
        for child in tree.children:
            type, value = child.value
            if type != ACTION_GENERATE_TYPE or value[0] != '-NONE-':
                node = self._remove_none_tags_from_tree(child)
                node.parent = root
                root.add_child(node)
        return root

    def _remove_non_terminal_leafs(self, tree):
        changed = True
        while changed:
            tree, changed = self._non_terminal_leafs_remover(tree)
        return tree

    def _non_terminal_leafs_remover(self, tree):
        root = TreeNode(tree.value)
        if tree.children is None:
            return root, False
        changed = False
        for child in tree.children:
            type, value = child.value
            if type == ACTION_NON_TERMINAL_TYPE and (child.children is None or len(child.children) == 0):
                changed = True
            else:
                node, changed_child = self._non_terminal_leafs_remover(child)
                changed = changed or changed_child
                node.parent = root
                root.add_child(node)
        return root, changed

    def _tree2line(self, tree):
        line = []
        self._tree2line_recursive(tree, line)
        return ''.join(line)

    def _tree2line_recursive(self, tree, line):
        type, value = tree.value
        if type == ACTION_GENERATE_TYPE:
            line.append(f'({value[0]} {value[1]}) ')
        else:
            line.append(f'({value} ')
            for child in tree.children:
                self._tree2line_recursive(child, line)
            line.append(') ')

class TreeNode:

    def __init__(self, value, parent=None, children=None):
        self.value = value
        self.parent = parent
        self.children = children

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
