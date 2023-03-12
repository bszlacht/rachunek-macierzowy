import pandas as pd
from matplotlib import pyplot as plt


columns = ["time", "n", "l", "flops"]

df1 = pd.read_csv("results/case1", usecols=columns)
df2 = pd.read_csv("results/case2", usecols=columns)
df3 = pd.read_csv("results/case3", usecols=columns)

fig, axs = plt.subplots(3, 2, figsize=(12,15))
axs[0,0].plot(df1.n, df1.time)
axs[0,0].set_title('Size x Time for l = 4')
axs[0,1].plot(df1.n, df1.flops)
axs[0,1].set_title('Size x Flops for l = 4')

axs[1,0].plot(df2.n, df2.time)
axs[1,0].set_title('Size x Time for l = 6')
axs[1,1].plot(df2.n, df2.flops)
axs[1,1].set_title('Size x Flops for l = 6')

axs[2,0].plot(df3.n, df3.time)
axs[2,0].set_title('Size x Time for l = 8')
axs[2,1].plot(df3.n, df3.flops)
axs[2,1].set_title('Size x Flops for l = 8')

for ax in axs.flat:
    ax.set(xlabel='size')


# Hide x labels and tick labels for top plots and y ticks for right plots.

plt.show()
