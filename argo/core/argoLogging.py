# inspired from tf_logging
import logging as _logging
import os
import sys
import time
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading

# Don't use this directly. Use get_logger() instead.
_logger = None
_logger_lock = threading.Lock()


def get_logger():
    global _logger
    
    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger
    
    _logger_lock.acquire()
        
    try:
        if _logger:
            return _logger
            
        # Scope the TensorFlow logger to not conflict with users' loggers.
        logger = _logging.getLogger()
        # import pdb;pdb.set_trace()
        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not _logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if sys.ps1: _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = sys.flags.interactive
                
            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel(INFO)
                _logging_target = sys.stdout
            else:
                _logging_target = sys.stderr
            
            formatter = _logging.Formatter(' %(message)s')
            # Add the output handler.
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(formatter)
            # _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)
        
        logger.setLevel(INFO)
        _logger = logger
        return _logger
    
    finally:
        _logger_lock.release()

 
# def init_logger(logname):
#     logging.basicConfig(format='%(message)s', filename=logname, filemode='w', level=logging.DEBUG)
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#     # # create file handler which logs even debug messages
#     # fh = logging.FileHandler(logname)
#     # create console handler
#     ch = logging.StreamHandler(sys.stdout)
#
#     # add the handlers to the logger
#     # logger.addHandler(fh)
#     logger.addHandler(ch)
#
#     return logger
