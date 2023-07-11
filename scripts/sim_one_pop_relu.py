import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import figaspect
import time

import network
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument('--gbar', '-gb', type=float, default=0.0)
parser.add_argument('--g', '-g', type=float, default=0.0)
parser.add_argument('--cbar', '-cb', type=float, default=0.0)
parser.add_argument('--c', '-c', type=float, default=0.0)
parser.add_argument('--tau', '-t', type=float, default=1.0)
parser.add_argument('--N', '-N', type=int, default=10000)
parser.add_argument('--seed', '-s', type=int, default=0)

args = vars(parser.parse_args())
print(parser.parse_args())

gbar = np.array([args['gbar']])
g = np.array([args['g']])
cbar = np.array([args['cbar']])
c = np.array([args['c']])
tau = np.array([args['tau']])
N = args['N']
seed = args['seed']

g2 = g**2
c2 = c**2

net = network.Network(NC=N)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using",device)

net.set_seed(seed)

J = net.generate_gauss_conn(gbar[0],g[0])
I = net.generate_gauss_input(cbar[0],c[0])

J = torch.from_numpy(J).to(device)
I = torch.from_numpy(I).to(device)

T = tau[0]*torch.ones(net.N,dtype=torch.float32).to(device)

def phi(h):
    return np.fmax(h,0)

relu = torch.nn.ReLU()

Nt = 50*tau[0]
dt = tau[0]/3
int_ts = torch.linspace(0,12*Nt,round(12*Nt/dt)+1).to(device)
mask_time = int_ts>(2*Nt)
int_ts_mask = int_ts[mask_time].cpu().numpy()

start = time.process_time()

h,r,_ = integ.sim_dyn_tensor(int_ts,relu,T,J,I)

h_mask = h[:,mask_time].cpu().numpy()
r_mask = r[:,mask_time].cpu().numpy()
h = h.cpu().numpy()
r = r.cpu().numpy()

print("Integrating network took ",time.process_time() - start," s")
print('')

hbar_mask = np.mean(h_mask,-1)
rbar_mask = np.mean(r_mask,-1)

htild_mask = h_mask - np.mean(h_mask,-1)[:,None]
rtild_mask = r_mask - np.mean(r_mask,-1)[:,None]

h_fluct = np.std(np.mean(h_mask,0))
r_fluct = np.std(np.mean(r_mask,0))

u = np.mean(hbar_mask)
m = np.mean(rbar_mask)

Dbar = np.var(hbar_mask)
Cbar = np.mean(rbar_mask**2)

def off_diag(A):
    return A[~np.eye(A.shape[0],dtype=bool)]

def off_diag_sum(A):
    return np.sum(A)-np.sum(np.diag(A))

lags = np.arange(0,50+1,2)*tau[0]
lag_idxs = np.round(lags / dt).astype(np.int32)

Dtild_on_0 = np.zeros(len(lags))
Dtild_on_2 = np.zeros(len(lags))
Dtild_off_0 = np.zeros(len(lags))
Dtild_off_2 = np.zeros(len(lags))
Ctild_on_0 = np.zeros(len(lags))
Ctild_on_2 = np.zeros(len(lags))
Ctild_off_0 = np.zeros(len(lags))
Ctild_off_2 = np.zeros(len(lags))

for idx,lag_idx in enumerate(lag_idxs):
    h_cov_mat = htild_mask[:,lag_idx:]@htild_mask[:,:len(int_ts_mask)-lag_idx].T/(len(int_ts_mask)-lag_idx)
    r_cov_mat = rtild_mask[:,lag_idx:]@rtild_mask[:,:len(int_ts_mask)-lag_idx].T/(len(int_ts_mask)-lag_idx)
    
    Dtild_on_0[idx] = np.mean(np.diag(h_cov_mat))
    Dtild_on_2[idx] = np.mean(np.diag(h_cov_mat)**2)
    Dtild_off_0[idx] = np.mean(off_diag(h_cov_mat))*net.N
    Dtild_off_2[idx] = np.mean(off_diag(h_cov_mat**2))*net.N
    Ctild_on_0[idx] = np.mean(np.diag(r_cov_mat))
    Ctild_on_2[idx] = np.mean(np.diag(r_cov_mat)**2)
    Ctild_off_0[idx] = np.mean(off_diag(r_cov_mat))*net.N
    Ctild_off_2[idx] = np.mean(off_diag(r_cov_mat**2))*net.N

res_dict = {}

res_dict['h_fluct'] = h_fluct
res_dict['r_fluct'] = r_fluct
res_dict['u'] = u
res_dict['m'] = m
res_dict['Dbar'] = Dbar
res_dict['Cbar'] = Cbar
res_dict['Dtild_on_0'] = Dtild_on_0
res_dict['Dtild_on_2'] = Dtild_on_2
res_dict['Dtild_off_0'] = Dtild_off_0
res_dict['Dtild_off_2'] = Dtild_off_2
res_dict['Ctild_on_0'] = Ctild_on_0
res_dict['Ctild_on_2'] = Ctild_on_2
res_dict['Ctild_off_0'] = Ctild_off_0
res_dict['Ctild_off_2'] = Ctild_off_2

with open('./../results/res_one_pop_relu_gbar={:.1f}_g={:.1f}_cbar={:.1f}_c={:.1f}_tau={:.1f}_N={:d}_seed={:d}'.format(
        gbar[0],g[0],cbar[0],c[0],tau[0],N,seed)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)

