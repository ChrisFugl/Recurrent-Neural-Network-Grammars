from collections import Counter

def get_terminals(bracket_lines):
    """
    :type bracket_lines: list of str
    :rtype: list of str, collections.Counter
    """
    terminals_counter = _get_terminals_counter(bracket_lines)
    terminals = []
    for word, count in terminals_counter.items():
        if count > 1:
            terminals.append(word)
    return terminals, terminals_counter

def _get_terminals_counter(bracket_lines):
    terminals_counter = Counter()
    for bracket_line in bracket_lines:
        bracket_line_stripped = bracket_line.strip()
        for character_index, character in enumerate(bracket_line_stripped):
            if is_start_of_terminal_node(bracket_line, character_index):
                _, terminal, _ = get_terminal_node(bracket_line, character_index)
                terminals_counter[terminal] += 1
    return terminals_counter

def is_start_of_terminal_node(line, line_index):
    """
    :type line: str
    :type line_index: int
    """
    if line[line_index] != '(':
        return False
    for character in line[line_index + 1:]:
        if character == '(':
            return False
        elif character == ')':
            return True
    raise Exception(f'Illegal bracketing: Could not find a closing bracket: "{line}"')

def get_terminal_node(line, terminal_start_index):
    """
    :type line: str
    :type terminal_start_index: int
    :rtype: str, str, int
    :returns: termonal part-of-speech tag, terminal word, terminal end index
    """
    search_start_index = terminal_start_index + 1
    terminal_end_index = line.find(')', search_start_index)
    terminal_node = line[search_start_index:terminal_end_index].split(' ')
    assert len(terminal_node) == 2, 'Terminal nodes must contain two elements: Part of speech tag and word.'
    assert len(terminal_node[0]) != 0, 'Terminal node tag cannot be empty.'
    assert len(terminal_node[1]) != 0, 'Terminal node word cannot be empty.'
    tag = terminal_node[0].strip()
    word = terminal_node[1].strip()
    return tag, word, terminal_end_index

def find_next_terminal_start_index(line, start_index):
    """
    :type line: str
    :type start_index: int
    :rtype: int
    """
    line_index = start_index
    while not is_start_of_terminal_node(line, line_index) and line_index != -1:
        line_index = line.find('(', line_index + 1)
    return line_index
