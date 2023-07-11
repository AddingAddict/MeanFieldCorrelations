import numpy as np

class Network:
    def __init__(self, seed=0, n=None, NC=None, normalize_by_mean=False):
        self.rng = np.random.default_rng(seed=seed)

        if NC is None:
            if n is None:
                self.NC = np.ones(1,int)
                self.n = 1
            else:
                self.NC = np.ones(n,int)
                self.n = int(n)
        elif np.isscalar(NC):
            if n is None:
                self.NC = NC*np.ones(1,int)
                self.n = 1
            else:
                self.NC = NC*np.ones(n,int)
                self.n = int(n)
        else:
            if n is None:
                self.NC = np.array(NC,int)
                self.n = len(self.NC)
            else:
                self.NC = np.array(NC,int)
                self.n = int(n)
                assert self.n == len(self.NC)

        self.N = np.sum(self.NC)

        self.C_idx = []
        prev_NC = 0
        for cidx in range(self.n):
            prev_NC += 0 if cidx == 0 else self.NC[cidx-1]
            this_NC = self.NC[cidx]
            self.C_idx.append(slice(int(prev_NC),int(prev_NC+this_NC)))

        self.normalize_by_mean = normalize_by_mean

    def set_seed(self,seed):
        self.rng = np.random.default_rng(seed=seed)

    def generate_sparse_conn(self,W,K):
        J = np.zeros((self.N,self.N),np.float32)

        for pstC in range(self.n):
            pstC_idx = self.C_idx[pstC]
            NpstC = self.NC[pstC]
            for preC in range(self.n):
                preC_idx = self.C_idx[preC]
                NpreC = self.NC[preC]

                if np.isscalar(W):
                    this_W = W
                elif W.ndim == 1:
                    this_W = W[preC]
                else:
                    this_W = W[pstC,preC]

                if np.isscalar(K):
                    this_K = K*NpreC/self.NC[0]
                elif K.ndim == 1:
                    this_K = K[preC]
                else:
                    this_K = K[pstC,preC]

                p = np.fmax(this_K/NpreC,1e-12)
                if p > 1:
                    raise Exception("Error: p > 1, please decrease K or increase NC")

                J[pstC_idx,preC_idx] = self.rng.binomial(1,p,size=(NpstC,NpreC)) * this_W

        return J

    def generate_sparse_input(self,WX,KX,NX=None):
        if NX is None:
            NX = self.NC[0]

        JX = np.zeros(self.N,np.float32)

        for pstC in range(self.n):
            pstC_idx = self.C_idx[pstC]
            NpstC = self.NC[pstC]

            if np.isscalar(WX):
                this_WX = WX
            elif WX.ndim == 1:
                this_WX = WX[preC]
            else:
                this_WX = WX[pstC,preC]

            if np.isscalar(KX):
                this_KX = KX
            elif KX.ndim == 1:
                this_KX = KX[preC]
            else:
                this_KX = KX[pstC,preC]

            p = np.fmax(this_KX/NX,1e-12)
            if p > 1:
                raise Exception("Error: p > 1, please decrease K or increase NC")

            JX[pstC_idx] = self.rng.binomial(NX,p,size=NpstC) * this_WX

        return JX

    def generate_gauss_conn(self,gbar,g):
        J = np.zeros((self.N,self.N),np.float32)

        for pstC in range(self.n):
            pstC_idx = self.C_idx[pstC]
            NpstC = self.NC[pstC]
            for preC in range(self.n):
                preC_idx = self.C_idx[preC]
                NpreC = self.NC[preC]

                if np.isscalar(gbar):
                    this_gbar = gbar
                elif gbar.ndim == 1:
                    this_gbar = gbar[preC]
                else:
                    this_gbar = gbar[pstC,preC]

                if np.isscalar(g):
                    this_g = g
                elif g.ndim == 1:
                    this_g = g[preC]
                else:
                    this_g = g[pstC,preC]

                mu = this_gbar/NpreC
                sig = this_g/np.sqrt(NpreC)

                J[pstC_idx,preC_idx] = self.rng.normal(loc=mu,scale=sig,size=(NpstC,NpreC))

        return J

    def generate_gauss_input(self,cbar,c):
        I = np.zeros(self.N,np.float32)

        for pstC in range(self.n):
            pstC_idx = self.C_idx[pstC]
            NpstC = self.NC[pstC]

            if np.isscalar(cbar):
                this_cbar = cbar
            elif cbar.ndim == 1:
                this_cbar = cbar[preC]
            else:
                this_cbar = cbar[pstC,preC]

            if np.isscalar(c):
                this_c = c
            elif c.ndim == 1:
                this_c = c[preC]
            else:
                this_c = c[pstC,preC]

            mu = this_cbar
            sig = this_c

            I[pstC_idx] = self.rng.normal(loc=mu,scale=sig,size=NpstC)

        return I