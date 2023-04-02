import time
import numpy as np
import os
import zarr

class TimerError(Exception):
    pass

class PapyrusDataException(Exception):
    pass

class Timer():
    """This is a utility class that, when used as a context manager,
    will report the time spent on code inside its block.
    """
    def __init__(self, text=None):
        if text is not None:
            self.text = text + ": {:0.4f} seconds"
        else:
            self.text = "Elapsed time: {:0.4f} seconds"
        def logfunc(x):
            print(x)
        self.logger = logfunc
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError("Timer is already running.  Use .stop() to stop it.")
        self._start_time = time.time()

    def stop(self):
        if self._start_time is None:
            raise TimerError("Timer is not running.  Use .start() to start it.")
        elapsed_time = time.time() - self._start_time
        self._start_time = None

        if self.logger is not None:
            self.logger(self.text.format(elapsed_time))

        return elapsed_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

def get_min_datatype(array):
    """Given a numpy array, tries to find the smallest possible datatype that will
    fit the data.
    """
    dtype = np.find_common_type([np.min_scalar_type(array.min()), np.min_scalar_type(array.max())], [])
    if (dtype == np.uint8) and array.max() in [0, 1]:
        return np.bool8
    return dtype

def downsample_slice(item, scale):
    """Given a downsampled scale factor (a non-negative integer), downsamples an input slice to
    approximately match the same region on a range
    downsampled 2**scale-fold.
    """
    if scale == 0:
        return item
    if isinstance(item, int):
        return item // (2 ** scale)
    if not isinstance(item, slice):
        raise PapyrusDataException("Must index with slices")
    if item.step is not None:
        raise PapyrusDataException("Cannot use step slices on downsampled data")
    start = None if item.start is None else item.start // (2 ** scale)
    stop = None if item.stop is None else item.stop // (2 ** scale)
    return slice(start, stop, None)

def get_slice_size(item, dimsize):
    """Given a slice and the maximum size of the array along the dimension we are slicing,
    gets the size of the slice.
    """
    if item.step is not None:
        raise PapyrusDataException("Cannot use step slices on downsampled data")
    start = 0 if item.start is None else item.start
    stop = dimsize if item.stop is None else item.stop
    return stop - start

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def open_store(filepath, mode="r"):
    """We're going to abstract the store out to this function so that
    we can experiment with different store types across the 
    whole project easily.
    """
    return zarr.SQLiteStore(filepath)
