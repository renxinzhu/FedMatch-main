"""
adapted version of logger
"""
__author__ = "Wonyong Jeong"
__email__ = "wyjeong@kaist.ac.kr"

import os
import json
from datetime import datetime
from typing import Union


class Logger:

    def __init__(self, client_id: Union[int, None]):
        """ Logging Module

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.client_id = client_id
        self.filepath = None
        # self.args = args
        # self.options = vars(self.args)

    def print(self, message):
        name = f'client-{self.client_id}' if self.client_id else 'server'
        print(f'[{datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}]' +
              f'[{name}] ' +
              f'{message}')

    def save_current_state(self, data, path="./log"):
        name = f'client-{self.client_id}' if self.client_id else 'server'
        filename = f'{name}-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.txt'
        if not self.filepath:
            self.filepath = os.path.join(path, filename)

        if not os.path.isdir(path):
            os.makedirs(path)

        with open(self.filepath, 'a+') as outfile:
            content = [str(i) for i in data.values()]
            outfile.write(",".join(content))
            outfile.write('\n')
