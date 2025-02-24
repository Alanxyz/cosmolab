import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import corner

flat_samples = np.load("out/chain.npy")
fig = corner.corner(flat_samples[:900], labels=[
    "a", "b", "mb", "dm", "Om"
] ,plot_datapoints=False,levels=(0.68,0.95,0.99),fill_contours=True,plot_density=False,quantiles=[0.16, 0.5, 0.84])
plt.show()
