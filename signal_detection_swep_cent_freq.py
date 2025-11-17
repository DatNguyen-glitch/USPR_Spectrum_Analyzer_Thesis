# this module will be imported in the into your flowgraph

class Sweeper:
    def __init__(self, start=2.5e7, stop=1e9):
        self.start = start
        self.stop = stop
        self.chunk_index = 0

    def next(self, step):
        cent_freq = self.start + self.chunk_index * step
        if cent_freq > self.stop:
            return None
        self.chunk_index += 1
        return cent_freq

sweeper = Sweeper()