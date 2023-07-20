import time
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint, odeint_event

def sim_dyn(ts,phi,T,J,I,max_min=7.5,stat_stop=True):
    F=np.zeros_like(I,np.float32)
    start = time.process_time()
    max_time = max_min*60
    timeout = False

    def ode_fn(t,H):
        R = phi(H)
        F = np.matmul(J,R) + I - H
        F = F / T
        return F

    # This function determines if the system is stationary or not
    def stat_event(t,H):
        meanF = np.mean(np.abs(F)/np.maximum(H,1e-1)) - 5e-3
        if meanF < 0: meanF = 0
        return meanF
    stat_event.terminal = True

    # This function forces the integration to stop after 15 minutes
    def time_event(t,H):
        int_time = (start + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True

    H = np.zeros((len(H),len(ts)));
    if stat_stop:
        sol = solve_ivp(ode_fn,[np.min(ts),np.max(ts)],H[:,0], method='RK45', t_eval=T, events=[stat_event,time_event])
    else:
        sol = solve_ivp(ode_fn,[np.min(ts),np.max(ts)],H[:,0], method='RK45', t_eval=T, events=[time_event])
    if sol.t.size < len(ts):
        print("      Integration stopped after " + str(np.around(ts[sol.t.size-1],2)) + "s of simulation time")
        if time.process_time() - start > max_time:
            print("            Integration reached time limit")
            timeout = True
        H[:,0:sol.t.size] = sol.y
        H[:,sol.t.size:] = sol.y[:,-1:]
    else:
        H=sol.y
    
    return H,phi(H),timeout

def sim_dyn_tensor(ts,phi,T,J,I,method=None):
    F = torch.ones_like(I,dtype=torch.float32)

    def ode_fn(t,H):
        R = phi(H)
        F = torch.matmul(J,R)
        F = torch.add(F,I - H)
        F = torch.div(F,T)
        return F

    def event_fn(t,H):
        meanF = torch.mean(torch.abs(F)/torch.maximum(H,1e-1*torch.ones_like(H))) - 5e-3
        if meanF < 0: meanF = 0
        return torch.tensor(meanF)

    # H = odeint(ode_fn,torch.zeros_like(I),ts[[0,-1]],event_fn=event_fn)
    H = odeint(ode_fn,torch.zeros_like(I,dtype=torch.float32),ts,method=method)
    H = torch.transpose(H,0,1)

    return H,phi(H),False

