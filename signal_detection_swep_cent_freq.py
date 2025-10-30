# this module will be imported in the into your flowgraph

class Sweeper:
    def __init__(self, start=4.5e6, stop=1e9):
        self.start = start
        self.stop = stop
        self.chunk_index = 0

    def next(self, step):
        cent_freq = self.start + self.chunk_index * step
        if cent_freq > self.stop:
            return None
        self.chunk_index += 1
        print(f"{self.chunk_index} Sweeper: Setting center frequency to {cent_freq/1e6} MHz")
        return cent_freq

sweeper = Sweeper()