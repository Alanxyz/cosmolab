import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import corner

from utils import *

matplotlib.use('PDF')

def pltxy(x, y):
    plt.plot(x, y)
    plt.show()

def plot_adjust(sol, df):
    plt.errorbar(
        df['z'],
        df['mu'],
        yerr=df['b'],
        fmt='o',
        color='dodgerblue',
        mec='black',
        ecolor='black',
        capsize=3,
        zorder=1
    )
    plt.plot(
        sol['z'],
        sol['mu'],
        color='orange',
        zorder=2
    )
    plt.show()

def triangle(sampler):
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    fig = corner.corner(
        flat_samples,
        labels=['a', 'b', 'mb1', 'dm'],
    )
    plt.savefig('plt/test.png')
