import logging


class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)

    def get_logger(self):
        return self.logger