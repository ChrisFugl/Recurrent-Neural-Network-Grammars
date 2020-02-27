import re

ACTION_REDUCE_INDEX = 0
ACTION_GENERATE_INDEX = 1
ACTION_SHIFT_INDEX = 1
ACTION_NON_TERMINAL_INDEX = 2

ACTION_REDUCE_TYPE = 0
ACTION_SHIFT_TYPE = 1
ACTION_GENERATE_TYPE = 2
ACTION_NON_TERMINAL_TYPE = 3

BROWN_CLUSTERING_TOOL_PATH = 'tools/percyliang_browncluster/wcluster'

ROOT_NON_TERMINAL = '<ROOT>'
UNKNOWN_IDENTIFIER = '<UNK>'

PAD_INDEX = 0
PAD_SYMBOL = '<PAD>'

MAX_OPEN_NON_TERMINALS = 100

# action patterns
PATTERN_NON_TERMINAL = re.compile(r'^NT\((\S+)\)$')
PATTERN_REDUCE = re.compile(r'^REDUCE$')
PATTERN_GEN = re.compile(r'^GEN\((\S+)\)$')
PATTERN_SHIFT = re.compile(r'^SHIFT$')
