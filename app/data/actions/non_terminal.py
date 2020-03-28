from app.constants import ACTION_NON_TERMINAL_INDEX, ACTION_NON_TERMINAL_TYPE
from app.data.actions.action import Action

class NonTerminalAction(Action):

    def __init__(self, argument, open=True):
        """
        :type argument: str
        :type open: bool
        """
        super().__init__()
        self.argument = argument
        self.open = open

    def index(self):
        """
        :rtype: int
        """
        return ACTION_NON_TERMINAL_INDEX

    def type(self):
        """
        :rtype: int
        """
        return ACTION_NON_TERMINAL_TYPE

    def __str__(self):
        return f'NT({self.argument})'
