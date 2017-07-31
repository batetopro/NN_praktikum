import logging
import sys

ALL = 0
WARNING = 1
INFO = 2
DEBUG = 3


CURRENT = ALL

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def output(message, level=DEBUG):
    if CURRENT >= level:
        if level == DEBUG or level == INFO:
            logging.info(message)
        else:
            logging.warning(message)


def array(array, level=DEBUG):
    pass


def dict(dict, level=DEBUG):
    pass
