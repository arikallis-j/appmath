import matplotlib.pyplot as plt
from funcs import * 

class EcoSystem:
    def __init__(self, N=11, h=1):
        self.N = N
        self.z = np.linspace(z0, zm, N)
        self.h = zm / (N - 1) 
        self.T = T_local(self.z)
        self.phyto = np.zeros(self.z.shape)
        self.zoo = np.zeros(self.z.shape)
        self.nitro = np.zeros(self.z.shape)
        self.organic = np.zeros(self.z.shape)
        self.I = np.zeros(self.z.shape)

    def going(self, starts, ends, phyto=None, zoo=None, nitro=None, organic=None, I=None):
        p_0, z_0, n_0, y_0, I_0 = starts
        p_N, z_N, n_N, y_N = ends
        
        if phyto is not None:
            self.phyto = phyto
        else:
            self.phyto = np.linspace(0, p_N, self.N)
        if zoo is not None:
            self.zoo = zoo
        else:
            self.zoo = np.linspace(0, z_N, self.N)
        if nitro is not None:
            self.nitro = nitro
        else:
            self.nitro = np.linspace(0, n_N, self.N)
        if organic is not None:
            self.organic = organic
        else:
            self.organic = np.linspace(0, y_N, self.N)
        if I is not None:
            self.I = I
        else:
            self.I = np.linspace(I_0, 0, self.N)
            

        self.phyto[0], self.phyto[-1] = p_0, p_N
        self.zoo[0], self.zoo[-1] = z_0, z_N
        self.nitro[0], self.nitro[-1] = n_0, n_N
        self.organic[0], self.organic[-1] = y_0, y_N
        self.I[0] = I_0

        a_phyto, b_phyto = np.zeros(self.z.shape), np.zeros(self.z.shape)
        a_zoo, b_zoo = np.zeros(self.z.shape), np.zeros(self.z.shape)
        a_nitro, b_nitro = np.zeros(self.z.shape), np.zeros(self.z.shape)
        a_organic, b_organic = np.zeros(self.z.shape), np.zeros(self.z.shape)
        a_I, b_I = np.zeros(self.z.shape), np.zeros(self.z.shape)

        Going2D = GoingTo(1, 1, 2, self.N)
        Going1D = GoingTo(0, 1, 1, self.N)
        
        for k in range(0, self.N):
            param_phyto = k, a_phyto[k-1], b_phyto[k-1], self.h**2 * P_local(self.phyto[k-1],self.zoo[k-1],self.nitro[k-1],self.organic[k-1], self.I[k-1], self.T[k-1]), p_0
            a_phyto[k], b_phyto[k] = Going2D.going_forward(*param_phyto)

            param_zoo = k, a_zoo[k-1], b_zoo[k-1], self.h**2 * Z_local(self.phyto[k-1],self.zoo[k-1],self.nitro[k-1],self.organic[k-1], self.I[k-1], self.T[k-1]), z_0
            a_zoo[k], b_zoo[k] = Going2D.going_forward(*param_zoo)

            param_nitro = k, a_nitro[k-1], b_nitro[k-1], self.h**2 * N_local(self.phyto[k-1],self.zoo[k-1],self.nitro[k-1],self.organic[k-1], self.I[k-1], self.T[k-1]), n_0
            a_nitro[k], b_nitro[k] = Going2D.going_forward(*param_nitro)
            
            param_organic = k, a_organic[k-1], b_organic[k-1], self.h**2 * Y_local(self.phyto[k-1],self.zoo[k-1],self.nitro[k-1],self.organic[k-1], self.I[k-1], self.T[k-1]), y_0
            a_organic[k], b_organic[k] = Going2D.going_forward(*param_organic)

            param_I = k, a_I[k-1], b_I[k-1], self.h * I_local(self.phyto[k-1],self.zoo[k-1],self.nitro[k-1],self.organic[k-1], self.I[k-1], self.T[k-1]), I_0
            a_I[k], b_I[k] = Going1D.going_forward(*param_I)

            # print(param_phyto)
            # print(a_phyto[k], b_phyto[k])
        
        for k in range(0, self.N-1):
            self.phyto[k] = Going2D.going_backward(k, a_phyto[k+1], b_phyto[k+1], self.phyto[k+1], y_N=p_N)
            self.zoo[k] = Going2D.going_backward(k, a_zoo[k+1], b_zoo[k+1], self.zoo[k+1], y_N=z_N)
            self.nitro[k] = Going2D.going_backward(k, a_nitro[k+1], b_nitro[k+1], self.nitro[k+1], y_N=n_N)
            self.organic[k] = Going2D.going_backward(k, a_organic[k+1], b_organic[k+1], self.organic[k+1], y_N=y_N)
            self.I[k] = Going2D.going_backward(k, a_I[k+1], b_I[k+1], self.I[k+1])

            # print(k, k+1, self.N-1)
        
        self.I[-1] = self.I[self.N-1] + self.h * I_local(self.phyto[self.N-1],self.zoo[self.N-1],self.nitro[self.N-1],self.organic[self.N-1], self.I[k-1], self.T[k-1])

        return self.phyto, self.zoo, self.nitro, self.organic, self.I

class GoingTo:
    def __init__(self, A, B, C, N):
        self.A = A
        self.B = B
        self.C = C
        self.N = N

    def going_forward(self, k, a_cur, b_cur, F, y_0):
        a_for, b_for = None, None
        if k == 0:
            a_for = 1
            b_for = 0
        else:
            a_for = self.B / (self.C - a_cur * self.A)
            b_for = (self.A * b_cur  + F) / (self.C - a_cur * self.A)

        return a_for, b_for
    
    def going_backward(self, k, a, b, y_cur, y_N=None):
        y_back = a * y_cur + b
        if y_N is not None:
            if k == (self.N - 1):
                y_back = y_N

        return y_back