import logging
import time


def setup_log(logfilename: str, timezone='GMT', date_format='%Y-%m-%d, %H:%M:%S'):
    # logging.basicConfig(filename=logfilename,
    #                     filemode='a',
    #                     format='%(asctime)s.%(msecs)d - %(levelname)s: %(message)s',
    #                     datefmt="("+timezone+") "+date_format,
    #                     level=logging.DEBUG)
    if timezone.lower() == 'gmt':
        logging.Formatter.converter = time.gmtime
    else:
        raise NotImplementedError
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s.%(msecs)d - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
