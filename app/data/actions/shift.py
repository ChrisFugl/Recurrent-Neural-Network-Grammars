from app.constants import ACTION_SHIFT_INDEX, ACTION_SHIFT_TYPE
from app.data.actions.action import Action

class ShiftAction(Action):

    def index(self):
        """
        :rtype: int
        """
        return ACTION_SHIFT_INDEX

    def type(self):
        """
        :rtype: int
        """
        return ACTION_SHIFT_TYPE

    def __str__(self):
        return 'SHIFT'
