from collections import OrderedDict
import gc
import torch
from jobs.process import BaseExtensionProcess


def flush():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


class DatasetTools(BaseExtensionProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)

    def run(self):
        super().run()

        raise NotImplementedError("This extension is not yet implemented")
