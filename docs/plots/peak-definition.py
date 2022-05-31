import tidyms as ms
import numpy as np
import matplotlib.pyplot as plt

# always generate the same plot
np.random.seed(1234)

grid = np.arange(50)
signal = ms.utils.gauss(grid, 25, 2, 30)
noise = np.random.normal(size=signal.size, scale=1)
x = signal + noise + 3
peak = ms.lcms.Peak(19, 25, 30)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(grid, x, label="signal")
ax.scatter(grid[peak.start], x[peak.start], label="peak start", s=50)
ax.scatter(grid[peak.apex], x[peak.apex], label="peak apex", s=50)
ax.scatter(grid[peak.end], x[peak.end], label="peak end", s=50)
ax.fill_between(grid[peak.start:peak.end + 1],
                x[peak.start:peak.end + 1], alpha=0.2, label="peak region")
ax.annotate(text='', xy=(grid[peak.end + 5], x[peak.end]),
            xytext=(grid[peak.end + 5], x[peak.apex]),
            arrowprops=dict(arrowstyle='<->'))
ax.annotate(text='peak \n prominence',
            xy=(grid[peak.end + 10],x[peak.apex] / 2))
ax.legend()
