from app.constants import ACTION_REDUCE_TYPE
from app.data.actions.action import Action

class ReduceAction(Action):

    def type(self):
        """
        :rtype: int
        """
        return ACTION_REDUCE_TYPE

    def __str__(self):
        return 'REDUCE'
