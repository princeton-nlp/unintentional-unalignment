from __future__ import annotations

import time
from typing import TYPE_CHECKING

from common.train.callbacks import Callback
from common.train.stop_fit_iteration import StopFitIteration

if TYPE_CHECKING:
    from ..trainer import Trainer


class StopOnTimeout(Callback):
    """
    Stops training after a certain amount of time has passed.
    """

    def __init__(self, timeout_in_seconds, use_process_time=False):
        """
        :param timeout_in_seconds: Number of seconds to stop fitting after.
        :param use_process_time: If true, will calculate timeout regarding only the process time (time it was executed on CPU) and not
        real time.
        """
        self.timeout_in_seconds = timeout_in_seconds
        self.use_process_time = use_process_time
        self.start_time = self.__get_time()

    def on_fit_start(self, trainer: Trainer, num_epochs: int):
        self.start_time = self.__get_time()

    def on_epoch_end(self, trainer: Trainer):
        curr_time = self.__get_time()
        if curr_time - self.start_time > self.timeout_in_seconds:
            time_passed_str = self.__get_timespan_str_format(curr_time - self.start_time)
            timeout_str = self.__get_timespan_str_format(self.timeout_in_seconds)
            used_timing_str = "real" if not self.use_process_time else "process"
            raise StopFitIteration(f"Stopping at end of epoch {trainer.epoch} because the timeout of {timeout_str} has expired. "
                                   f"Current training time is {time_passed_str}. Time was measured by {used_timing_str} time.")

    def __get_time(self):
        return time.time() if not self.use_process_time else time.process_time()

    def __get_timespan_str_format(self, time_in_seconds):
        mins, secs = divmod(time_in_seconds, 60)
        hours, mins = divmod(mins, 60)
        return '%02d:%02d:%02d' % (hours, mins, secs)
