from app.constants import ROOT_NON_TERMINAL

def get_non_terminal_identifier(line, line_index):
    """
    :type line: str
    :type line_index: int
    :rtype: str
    """
    non_terminal_start_index = line_index + 1
    non_terminal_end_index = line.find(' ', non_terminal_start_index)
    non_terminal = line[non_terminal_start_index:non_terminal_end_index].strip()
    if len(non_terminal) == 0:
        return ROOT_NON_TERMINAL
    else:
        return non_terminal
