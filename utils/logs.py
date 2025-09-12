import logging
import os
import functools
from datetime import datetime

# Module-level variable to store the current rundir
_current_rundir = None


def setup_report(level, subtag: str = None):
    global _current_rundir
    runtag = f"{datetime.today():%Y%m%d-%H.%M.%S}"
    rundir = f"out-{runtag}"
    if subtag is not None:
        rundir += "_" + subtag
    os.makedirs(rundir, exist_ok=True)
    _current_rundir = rundir  # Store the rundir
    report_filename = os.path.join(os.getcwd(), rundir, f"report-{runtag}.txt")
    match level.lower():
        case "debug":
            _set_level = logging.DEBUG
        case _:
            _set_level = logging.INFO
    logging.basicConfig(
        filename=report_filename,
        filemode="w",
        format="%(asctime)s - %(levelname)s \n   %(message)s",
        level=_set_level,
    )
    logger = logging.getLogger(__name__)
    return rundir, runtag, logger


def get_current_rundir():
    """Return the current rundir or None if setup_report hasn't been called."""
    return _current_rundir


# Rest of your existing code remains the same
def log_function_call(func, logger=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _string_args = ""
        for a in args:
            _string_args += f"\n{'':32}{str(a)[:100]}"
        _string_kwargs = ""
        for k, v in kwargs.items():
            _string_kwargs += f"\n{'':32}{k}: {str(v)[:100]}"
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.debug(
            f"""Calling function `{func.__name__}` with 
                    ---> args: {_string_args} 
                    ---> kwargs: {_string_kwargs}
                    """
        )
        return func(*args, **kwargs)

    return wrapper


def get_logger():
    return logging.getLogger(__name__)
