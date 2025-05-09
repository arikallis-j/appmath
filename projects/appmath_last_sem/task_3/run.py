import numpy as np
import matplotlib.pyplot as plt
from classes import WaveEquation
from classes import WaveEquationPML

# Пример использования

def main():
    # Параметры
    L, c = 1.0, 1.0
    a = L / 5
    Nx, Nt = 201, 200
    # Начальный профиль: гаусс
    phi = lambda z: np.exp(- z**2 / (2 * a**2))

    solver = WaveEquation(L, c, Nx, Nt, phi)
    E = solver.solve_numeric()
    for k in range(0,Nt+1, 10):
        solver.plot_solution(t_index=k, show=False, save=True, 
                             filename=f"wave_{k}|{Nt}")

    # GOOD PARAMETERS
    delta_layer = 0.3
    A0 = 4.0
    solver = WaveEquationPML(L=L, c=c, Nx=Nx, Nt=Nt, phi_func=phi,
                             delta_layer=delta_layer, A0=A0)
    # Без PML
    E, H = solver.solve_fdtd()
    for k in range(0,Nt+1, 10):
        solver.plot(t_index=k, show=False, save=True, 
                             filename=f"wave_nopml_{k}|{Nt}")
    # С PML
    E_pml, H_pml = solver.solve_fdtd_pml()
    for k in range(0,Nt+1, 10):
        solver.plot(ispml=True, t_index=k, show=False, save=True, 
                             filename=f"wave_pml_{k}|{Nt}")

if __name__ == '__main__':
    main()