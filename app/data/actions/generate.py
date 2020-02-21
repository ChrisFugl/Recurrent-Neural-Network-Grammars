from app.constants import ACTION_GENERATE_INDEX, ACTION_GENERATE_TYPE
from app.data.actions.action import Action

class GenerateAction(Action):

    def __init__(self, device, argument, argument_index):
        """
        :type device: torch.device
        :type argument: str
        :type argument_index: int
        """
        super().__init__(device)
        self.argument = argument
        self.argument_index = argument_index

    def argument_index_as_tensor(self):
        """
        :rtype: torch.Tensor
        """
        return self._to_long_tensor(self.argument_index)

    def index(self):
        """
        :rtype: int
        """
        return ACTION_GENERATE_INDEX

    def type(self):
        """
        :rtype: int
        """
        return ACTION_GENERATE_TYPE

    def __str__(self):
        return f'GEN({self.argument})'
