import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_corr(df, vanish_diag=True):
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(16, 14))
    corr = df.corr()
    if vanish_diag:
        for c in corr.columns:
            corr[c][c] = 0
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    return corr

def plot_mean_and_error_bound(df, key, max_bins=16):
    if (key.endswith(('bin', 'cat')) or len(df[key].value_counts()) < max_bins):
        df0 = df[[key, 'target']].copy()
    else :
        df0 = df[[key, 'target']].copy()
        bins = np.unique(df0[key].quantile(np.arange(max_bins+1)*1./max_bins).values)
        bins[0] -= 0.00001
        df0.loc[:, key] = pd.cut(df0[key], bins, labels=np.arange(len(bins)-1))
    mean = df0[[key, 'target']].groupby(key, as_index=False).mean()
    mean = mean.rename(columns={'target':'mean'})
    count = df0[[key, 'target']].groupby(key, as_index=False).count()
    count = count.rename(columns={'target':'n'})
    std = df0[[key, 'target']].groupby(key).std().reset_index()
    std = std.rename(columns={'target':'std'})
    tmp = mean.merge(count, on = key)
    tmp = tmp.merge(std, on = key)
    tmp.loc[:, 'se'] = tmp['std'] / np.sqrt(tmp.n * 1.)
    tmp.loc[:, '-3ses'] = tmp['mean'] - 3 * tmp['se']
    tmp.loc[:, '+3ses'] = tmp['mean'] + 3 * tmp['se']
    num_bins = tmp.shape[0]
    f, axarr = plt.subplots(2, sharex=True, figsize=(12, 6))
    df0[key].hist(ax=axarr[0], bins=num_bins*2)
    tmp.plot(ax=axarr[1], x=key, y=['-3ses', 'mean', '+3ses'], style='-o')
    plt.show()
