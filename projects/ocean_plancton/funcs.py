import numpy as np

from const import *

# Функции в уравнениях
def updateM_pz(P):
    return np.where(P < Up, 0, R_pzmax * P / (R_pz + P))

def updateEf(Z):
    return np.where(Z < Z_thr, 0, 0.1)

def updatemu_I(I):
    return np.where(I < I_thr, I * mu_maxI / (I0 + I), I * mu_maxI/ (I0 + I))

def T_local(z, T_0=3, T_h=2, H=70):
    b = T_0
    a = (T_h - T_0)/H
    T = a * z + b
    return T

def P_local(P, Z, N, Y, I, T):  
    mu_maxN = M_max / (1 + np.exp(-b * N * (1 - M_max / 2)))
    mu_N = N * mu_maxN / (N0 + N)
    mu_I = updatemu_I(I)
    mu = np.minimum(mu_N, mu_I)
    Ep = Ep_min * np.exp(A_cp * (T - T_min))
    M_yp = R_ypmax * kappa_yp * Y * np.exp(-rho * P) / (R_yp + kappa_yp * Y)
    M_pz = updateM_pz(P)
    p_local = (mu - Ep + M_yp)*P - M_pz*Z
    return p_local / (K)

def Z_local(P, Z, N, Y, I, T):
    M_pz = updateM_pz(P)
    Ez = Ez_min * np.exp(A_cz * (T - T_min))
    M_yz = 0
    Ef = updateEf(Z)
    z_local = M_pz * Z - Ez * Z + M_yz * Z - Ef * Z
    return z_local / (K)

def N_local(P, Z, N, Y, I, T):
    mu_maxN = M_max / (1 + np.exp(-b * N * (1 - M_max / 2)))
    mu_N = N * mu_maxN / (N0 + N)
    Ey = Ey_min * np.exp(A_cy * (T - T_min))
    F_N = Q1 / V * (X_ns - N) + Q2/V * (X_nb - X_ns) + W_F/V * (X_nd - X_ns)
    n_local = - D_np * mu_N * P + Ey*D_ny*Y + F_N
    return n_local / (K)

def Y_local(P, Z, N, Y, I, T):
    Ep = Ep_min * np.exp(A_cp * (T - T_min))
    M_yp = R_ypmax * kappa_yp * Y * np.exp(-rho * P) / (R_yp + kappa_yp * Y)
    Ez = Ez_min * np.exp(A_cz * (T - T_min))
    Ey = Ey_min * np.exp(A_cy * (T - T_min))
    M_pz = updateM_pz(P)
    M_yz = 0
    y_local = (Ep - M_yp) * P + ((1 - kappa_pz) * M_pz - kappa_yz * M_yz + Ez) * Z - Ey * Y
    return y_local / (K)

def I_local(P, Z, N, Y, I, T):
    i_locale = (alpha0 + alpha1 * P + alpha2 * Y) * I
    return i_locale