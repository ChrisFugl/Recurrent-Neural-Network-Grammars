from app.models.model import Model
import torch
from torch import nn

class RNNG(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp = nn.Linear(2, 2)
