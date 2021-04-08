from typing import Optional
from fastai.learner import Learner


def get_first_index_else_last(list_obj: list, idx: int, total_iters: int):
    item = list_obj[0]

    if idx == total_iters - 1:
        item = list_obj[-1]

    return list_obj[idx] if idx < len(list_obj) - 1 else item

class Tuner:

    def __init__(self, learner: Learner):
        self.learner = learner

    def fit_with_presizing(self,
                           presizing_cycles: int,
                           max_image_size: int,
                           image_sizes: Optional[list] = None,
                           epochs_per_cycle: Optional[list] = None,
                           batch_size: int = 64):

        if isinstance(image_sizes, type(None)):
            image_sizes = [int(max_image_size/(presizing_cycles - i)) for i in range(presizing_cycles)]

        if isinstance(epochs_per_cycle, type(None)):
            epochs_per_cycle = [5]

        for i in range(presizing_cycles):
            image_size_cycle = get_first_index_else_last(image_sizes, i, presizing_cycles)
            epochs = get_first_index_else_last(epochs_per_cycle, i, presizing_cycles)

            print(f"Fine tuning cycle '{i + 1}' for {epochs} epochs with parameters\n"
                  f"\tbatch_size: {batch_size},\n"
                  f"\timage_size: {image_size_cycle}:")

            self.learner.dls = get_dls(batch_size, image_size_cycle)
            self.fine_tune_with_epoch(epochs)
            print("")

    def fine_tune_with_epoch(self, epochs: int):
        with self.learner.no_bar():
            self.learner.fine_tune(epochs)
