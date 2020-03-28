from app.constants import ACTION_GENERATE_INDEX, ACTION_GENERATE_TYPE
from app.data.actions.action import Action

class GenerateAction(Action):

    def __init__(self, argument):
        """
        :type argument: str
        """
        super().__init__()
        self.argument = argument

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
