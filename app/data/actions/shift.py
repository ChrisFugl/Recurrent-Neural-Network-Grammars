from app.constants import ACTION_SHIFT_TYPE
from app.data.actions.action import Action

class ShiftAction(Action):

    def type(self):
        """
        :rtype: int
        """
        return ACTION_SHIFT_TYPE

    def __str__(self):
        return 'SHIFT'
