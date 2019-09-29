# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:17:04 2019

@author: Andreea
"""

import time


class Timer:
    start_time = []

    # Start runtime
    def tic(self):
        self.start_time.append(time.time())

    # Print runtime information
    def toc(self):
        diff = (time.time() - self.start_time.pop())
        print("Timer :: toc --- %s seconds ---" % diff)
        return diff

    def set_counter(self, c, max=100):
        self.counter_max = c
        self.counter = 0
        self.checkpoint = int(self.counter_max/max)
        self.step = self.checkpoint
        self.tic()

    def count(self, add=1):
        self.counter = self.counter + add

        if (self.counter >= self.checkpoint):
            print("Timer :: Checkpoint reached: {}%".format(int(
                    self.counter*100/self.counter_max)))
            self.toc()
            self.checkpoint += self.step
            if self.checkpoint <= self.counter_max:
                self.tic()
