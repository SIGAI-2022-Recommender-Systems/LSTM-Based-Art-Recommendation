import pandas as pd

class Logger:
    def __init__(self,metric_ftns):
        self.log = pd.DataFrame(columns=["Epoch"]+ metric_ftns)
        