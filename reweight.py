# Analysis script for histogram reweighting of the Blume Capel model
#
#!/usr/bin/python
import numpy as np
import scipy
import scipy.misc
import math
import os.path
import argparse
import glob
import matplotlib.pyplot as plt


global e_min
global e_max
global e_step
global num_threads

def add_log_numbers(a, b):
  if(a > b):
    log_max = a
    log_abs = a-b
  else:
    log_max = b
    log_abs = b-a
  return log_max + math.log1p( math.exp(-log_abs) )

def input_handler(items, **kwargs):
    kwargs.setdefault('unpack', True)
    items = glob.glob(os.path.expanduser(items))
    data = []
    for idx, item in enumerate(items):
        result = np.loadtxt(item, **kwargs)
        data.append(result)
    # return np.vstack(data)
    if len(data) > 1:
        return np.array(data)
    else:
        return data[0]

def get_bin(e):
    return int((e-e_min)/e_step+0.5)
def get_e(bin):
    return e_min+e_step*bin

# reweighted probabibility distribution from energy time series
def rw_ts(ts_e, logW, beta):
    h = np.zeros((e_max-e_min+1), dtype=float)
    p = np.zeros((e_max-e_min+1), dtype=float)
    log_norm=-1e50
    for i in range(len(ts_e)):
        bin_e = get_bin(ts_e[i])
        log_reweight = -1.0*beta*ts_e[i] - logW[bin_e]
        log_norm     = add_log_numbers(log_norm, log_reweight)
        h[bin_e] += 1
    for i in range(len(ts_e)):
        bin_e = get_bin(ts_e[i])
        bin_nd = int(ts_nd[i]+nd_min)
        log_reweight = -1.0*beta*ts_e[i] - logW[bin_e]
        p[bin_e] += math.exp(log_reweight - log_norm)
    return p


def rw_histogram_accumulated(e, h_e, o_acc, log_w, beta):
    """
        Canonical expecation value from histogram of accumulated observables.
        eq 2.75
        Assumes the last dimension in each element above contains the data
        entries and tries to broadcast.

        Parameters
        ----------
        e : ~numpy.ndarray
            energy labels -512 -508 ... 512
        h_e : ~numpy.ndarray
            histogram of energy occureneces
        o_acc : ~numpy.ndarray
            accumulated observable at energy
        log_w : ~numpy.ndarray
            logarithmic weights W(E)
        beta : float
            reweight temperature
    """
    assert log_w.shape[-1] == h_e.shape[-1] == e.shape[-1] == o_acc.shape[-1]
    # norm = np.sum(h_e * np.exp(-beta*e -log_w) )
    n2 = scipy.misc.logsumexp(np.log(h_e)-beta*e-log_w, axis=-1)
    # print(norm, n2)
    # ev = np.sum( o_acc * np.exp(-beta*e -log_w) )
    ev2 = scipy.misc.logsumexp(np.log(o_acc) -beta*e -log_w, axis=-1)
    # print(ev, ev2)
    return np.exp(ev2 - n2)


def jackknife_error(array):
    """
        Assumes binning along axis 0 where the block average was already
        calculated (no timeseries!)
    """
    nb = len(array)
    mask = np.ones(nb, dtype=bool)
    jackbins = np.zeros_like(array)
    for idx in range(nb):
        mask[idx] = False
        jackbins[idx] = np.mean(array[mask], axis=0)
        mask[idx] = True
    var = np.var(jackbins, axis=0)
    errs = np.sqrt(var*(nb-1))
    return errs


h_acc   = input_handler("/Users/paul/Desktop/mout2/ts*.dat")
weights = input_handler("/Users/paul/Desktop/mout2/production000.dat")

num_threads = h_acc.shape[0]

e_min  = weights[0, 0]
e_max  = weights[0, -1]
e_step = weights[0, 1] - e_min



t_c = 2 / np.log(1+np.sqrt(2))

# smooth
t_min = 1
t_max = 4
t_step = 0.1
t_range = np.arange(t_min, t_max, t_step)

# this way, arrays are broadcasted correctly
# ev = rw_histogram_accumulated(h_acc[:, 0:1], h_acc[:, 1:2], h_acc[:, 2:-1], weights[1], 1/t_c)

en = h_acc[0, 0, :]
h_en = np.sum(h_acc[:, 1, :], axis=0)
h_merged = np.sum(h_acc[:, 2:, :], axis=0)

ev = []
full = []
for t in t_range:
    ev.append(rw_histogram_accumulated(h_acc[:, 0:1], h_acc[0, 1:2], h_acc[:, 2:], weights[1], 1/t))
    full.append(rw_histogram_accumulated(en, h_en, h_merged, weights[1], 1/t))

ev=np.array(ev)
full=np.array(full)

# ev2 = ev.reshape(num_threads,len(t_range),5)
ev2 = np.swapaxes(ev, 0, 1)


errs = jackknife_error(ev2)
mean = np.mean(ev2, axis=0)

norm = [256, 64, 16, 9, 4, 64, 4]
for i in [0,5,2,6]:
# for i in [0,2]:
# for i in range(0,7):
    # plt.errorbar(t_range, mean[:,i]/norm[i], errs[:,i]/norm[i], label=norm[i])
    plt.errorbar(t_range/t_c, full[:,i]/norm[i], errs[:,i]/norm[i], label=norm[i])

plt.legend()
