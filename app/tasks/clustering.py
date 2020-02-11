from app.tasks.task import Task

class ClusteringTask(Task):

    def __init__(self, clustering):
        """
        :type clustering: app.clustering.clustering.Clustering
        """
        self._clustering = clustering

    def run(self):
        self._clustering.cluster()
