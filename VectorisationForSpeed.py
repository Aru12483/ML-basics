import tensorflow as tf 
#import matplotlib as plt 
#import pandas as pd
import numpy as np
import math
import time 

# VECTORISATION for speed 
n = 1000
a = tf.ones(n)
b = tf.ones(n)
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self): #reference to the current instance of the class 
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

c = tf.Variable(tf.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'