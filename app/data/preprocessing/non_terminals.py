from collections import Counter
import re

_NON_TERMINAL_PATTERN = re.compile('^\(\S+\s+\(')

def get_non_terminal_identifier(line, line_index):
    """
    :type line: str
    :type line_index: int
    :rtype: str
    """
    non_terminal_start_index = line_index + 1
    non_terminal_end_index = line.find(' ', non_terminal_start_index)
    non_terminal = line[non_terminal_start_index:non_terminal_end_index].strip()
    return non_terminal

def is_start_of_non_terminal_node(line, line_index):
    """
    :type line: str
    :type line_index: int
    """
    return _NON_TERMINAL_PATTERN.match(line[line_index:]) is not None

def get_non_terminals(bracket_lines):
    """
    :type bracket_lines: list of str
    :rtype: list of str, collections.Counter
    """
    non_terminals_counter = _get_non_terminals_counter(bracket_lines)
    non_terminals = []
    for word, count in non_terminals_counter.items():
        if count > 1:
            non_terminals.append(word)
    return non_terminals, non_terminals_counter

def _get_non_terminals_counter(bracket_lines):
    non_terminals_counter = Counter()
    for bracket_line in bracket_lines:
        bracket_line_stripped = bracket_line.strip()
        for character_index, character in enumerate(bracket_line_stripped):
            if is_start_of_non_terminal_node(bracket_line, character_index):
                non_terminal = get_non_terminal_identifier(bracket_line, character_index)
                non_terminals_counter[non_terminal] += 1
    return non_terminals_counter
