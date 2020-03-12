from app.constants import ACTION_NON_TERMINAL_INDEX, ACTION_NON_TERMINAL_TYPE
from app.data.actions.action import Action

class NonTerminalAction(Action):

    def __init__(self, device, argument, argument_index, open=True):
        """
        :type device: torch.device
        :type argument: str
        :type argument_index: int
        :type open: bool
        """
        super().__init__(device)
        self.argument = argument
        self.argument_index = argument_index
        self.open = open

    def argument_index_as_tensor(self):
        """
        :rtype: torch.Tensor
        """
        return self._to_long_tensor(self.argument_index)

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
