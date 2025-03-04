#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:40:24 2024

@author: sergey
"""

import numpy as np
import matplotlib.pyplot as plt

update_readout = np.load('update_readout_epoch_0.npy')

p_update = update_readout[1, :] - update_readout[0, :]
g_update = update_readout[2, :] - update_readout[0, :]

sz = p_update.shape[0]
c = np.diag(np.corrcoef(p_update, g_update)[sz:, :sz])
print(f'Correlation: {np.mean(c):.4f} +- {np.std(c):.4f}')

c_shuffle = np.diag(np.corrcoef(p_update, g_update[:, np.random.permutation(np.arange(100))])[sz:, :sz])
print(f'Shuffled correlation: {np.mean(c_shuffle):.4f} +- {np.std(c_shuffle):.4f}')

plt.plot(c)
plt.title('Pearson correlation between p-Net and g-Net updates')
plt.xlabel('Batch')
plt.ylabel('Correlation')
plt.show()

pg_ratio = np.mean(np.divide(p_update, g_update), axis=1)
# pg_ratio = np.mean(p_update - g_update, axis = 1)
print(f'Ratio: {np.mean(pg_ratio):.4f} +- {np.std(pg_ratio):.4f}')

plt.plot(pg_ratio)
plt.title('Ratio of p-Net and g-Net updates')
plt.xlabel('Batch')
plt.ylabel('Ratio')
plt.show()

plt.plot(p_update[100, :], label='pNet')
plt.plot(g_update[100, :], label='gNet')
# plt.title('Example of pNet and gNet updates')
plt.title('А тѣмъ врѣменем пiздѣцъ обрѣталъ фѣерiчный размахъ')
plt.xlabel('Parameter')
plt.ylabel('Update')
plt.legend()
plt.show()

plt.imshow(np.transpose(p_update)[:, :150])
plt.title('pNet update')
plt.xlabel('Batch')
plt.ylabel('Update')
plt.show()

plt.imshow(np.transpose(g_update)[:, :150])
plt.title('gNet update')
plt.xlabel('Batch')
plt.ylabel('Update')
plt.show()
