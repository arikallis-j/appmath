import numpy as np
import matplotlib.pyplot as plt
import os

class WaveEquation:
    """
    Класс для численного и аналитического решения одномерного волнового уравнения
    с граничными условиями E=0 на z=±L и начальными условиями E(0,z)=phi(z), E_t(0,z)=0.
    """
    def __init__(self, L, c, Nx, Nt, phi_func, psi_func=None):
        self.L = L
        self.c = c
        self.Nx = Nx
        self.Nt = Nt
        # Пространственная сетка
        self.x = np.linspace(-L, L, Nx)
        self.dx = self.x[1] - self.x[0]
        # Начальные условия
        self.phi = phi_func(self.x)
        # По условию psi=0
        self.psi = np.zeros_like(self.x) if psi_func is None else psi_func(self.x)
        # Выбираем dt удовлетворяющий условию устойчивости
        self.dt = self.dx / self.c
        self.r = self.c * self.dt / self.dx  # число Куранта
        # Массив для численного решения: временная размерность Nt+1, пространственная Nx
        self.E = np.zeros((Nt+1, Nx))


    def solve_numeric(self):
        """
        Численное решение явной схемой второго порядка по времени и пространству.
        Формулы (8) и (9) в описании.
        """
        # Начальные условия в момент t=0
        self.E[0, :] = self.phi
        # Первый шаг по времени: формула (9)
        self.E[1, 1:-1] = (
            self.E[0, 1:-1]
            + 0.5 * self.r**2 * (
                self.E[0, 2:] - 2*self.E[0, 1:-1] + self.E[0, :-2]
            )
        )
        # Граничные условия E=0 на краях
        self.E[1, 0] = 0
        self.E[1, -1] = 0
        # Основной цикл по времени: формула (8)
        for n in range(1, self.Nt):
            self.E[n+1, 1:-1] = (
                2*self.E[n, 1:-1]
                - self.E[n-1, 1:-1]
                + self.r**2 * (
                    self.E[n, 2:] - 2*self.E[n, 1:-1] + self.E[n, :-2]
                )
            )
            self.E[n+1, 0] = 0
            self.E[n+1, -1] = 0
            
        return self.E

    def solve_analytic(self, t):
        """
        Аналитическое решение по формуле Даламбера при psi=0:
        E(t,z) = (phi(z-ct) + phi(z+ct)) / 2
        """
        xp = self.x + self.c * t
        xm = self.x - self.c * t
        E_a = 0.5 * (
            np.interp(xp, self.x, self.phi, left=0, right=0) +
            np.interp(xm, self.x, self.phi, left=0, right=0)
        )
        return E_a

    def plot_solution(self, t_index=None, save=False, show=True, filename='wave'):
        """
        Построение сравнения численного и аналитического решения.
        Аргументы:
        - t_index: индекс временного шага для численного решения.
        - t: время для аналитического решения (если t_index не указан).
        - filename: если указан, сохраняет график в файл, иначе отображает.
        """
        # Определяем момент времени
        if t_index is None:
            t_index = self.Nt

        t_val = t_index * self.dt
        # Получаем решения
        E_num = self.E[t_index]
        E_an = self.solve_analytic(t_val)
        # Строим график
        plt.figure(figsize=(8, 5))
        plt.plot(self.x, E_num, label=f"Численное E | base")
        plt.plot(self.x, E_an, '--', label="Аналитическое E")
        plt.xlabel('z')
        plt.ylabel('Поля')
        plt.title(f'Поля в момент t={t_val:.3f}')
        plt.legend()
        plt.grid(True)

        if save:
            if not os.path.isdir(f'graph'):
                os.mkdir(f'graph')
            if not os.path.isdir(f'graph/wave'):
                os.mkdir(f'graph/wave')
            plt.savefig(f"graph/wave/{filename}.png")
        if show:
            plt.show()


class WaveEquationPML:
    def __init__(self, L, c, Nx, Nt, phi_func, delta_layer=0.1, A0=1.0):
        """
        L: полупространство [-L, L]
        c: скорость волны
        Nx: число точек по z (включая концы)
        Nt: число временных шагов
        phi_func: функция начального профиля phi(z)
        delta_layer: толщина PML-слоя (в единицах координаты z)
        A0: амплитуда σ в слое
        """
        self.L, self.c = L, c
        self.Nx, self.Nt = Nx, Nt
        # сетка по z
        self.z = np.linspace(-L, L, Nx)
        self.h = self.z[1] - self.z[0]
        # шаг по времени, условие CFL: dt = h/c
        self.dt = self.h / self.c
        self.r = self.c * self.dt / self.h
        
        # начальный профиль E, H
        self.E = np.zeros((Nt+1, Nx))
        # H на полушаге пространства: узлы между E[j] и E[j+1], всего Nx-1
        self.H = np.zeros((Nt+1, Nx-1))
        
        # начальные условия
        self.phi = phi_func(self.z)
        self.E[0, :] = phi_func(self.z)
        # dE/dt(0) = 0, значит первый шаг E[1] через центральную формулу:
        # E1_j = E0_j + (r^2/2)*(E0_{j+1}-2E0_j+E0_{j-1})
        self.E[1,1:-1] = self.E[0,1:-1] + 0.5*self.r**2*(self.E[0,2:] - 2*self.E[0,1:-1] + self.E[0,:-2])
        self.E[1,0] = self.E[1,-1] = 0
        
        # PML-параметры
        self.delta_layer = delta_layer
        self.A0 = A0
        # строим массив σ_j для каждой узловой точки E[j]
        self.sigma_E = np.zeros(Nx)
        mask_left  = (self.z > -L) & (self.z < -L + delta_layer)
        mask_right = (self.z <  L) & (self.z >  L - delta_layer)
        self.sigma_E[mask_left]  = A0 * ((self.z[mask_left]  + L - delta_layer)/delta_layer)**2
        self.sigma_E[mask_right] = A0 * ((self.z[mask_right] - L + delta_layer)/delta_layer)**2
        
        # для H: узлы между, координаты z_half = z[j] + h/2
        z_half = self.z[:-1] + 0.5*self.h
        self.sigma_H = np.zeros(Nx-1)
        mask_left_h  = (z_half > -L) & (z_half < -L + delta_layer)
        mask_right_h = (z_half <  L) & (z_half >  L - delta_layer)
        self.sigma_H[mask_left_h]  = A0 * ((z_half[mask_left_h]  + L - delta_layer)/delta_layer)**2
        self.sigma_H[mask_right_h] = A0 * ((z_half[mask_right_h] - L + delta_layer)/delta_layer)**2

    def solve_fdtd(self):
        """Схема (10): без PML."""
        for n in range(1, self.Nt):
            # шаг H на полушаге времени (n+0.5)
            self.H[n, :] = self.H[n-1, :] + self.r*(self.E[n,1:] - self.E[n,:-1])
            # шаг E на целый шаг времени (n+1)
            self.E[n+1,1:-1] = ( self.E[n,1:-1]
                + self.r*( self.H[n,1:] - self.H[n,:-1] ) )
            # граничные условия E=0 на концах
            self.E[n+1,0] = 0
            self.E[n+1,-1] = 0
        return self.E, self.H

    def solve_fdtd_pml(self):
        """Схема (12): с PML-затуханием."""
        for n in range(1, self.Nt):
            # 1) Обновляем H на полушаге времени n+0.5
            H_prev = self.H[n-1, :]                           # shape (Nx-1,)
            dE = self.E[n, 1:] - self.E[n, :-1]                # shape (Nx-1,)
            num_H = H_prev + self.r * dE
            den_H = 1 + 2 * np.pi * self.sigma_H * self.dt
            self.H[n, :] = (num_H - 2*np.pi*self.sigma_H*self.dt * H_prev) / den_H

            # 2) Обновляем E на шаге времени n+1
            E_prev = self.E[n, :]                              # shape (Nx,)
            diff_H = np.zeros_like(E_prev)                     # shape (Nx,)
            # внутренние узлы j=1..Nx-2:
            diff_H[1:-1] = self.H[n, 1:] - self.H[n, :-1]      # оба среза shape (Nx-2,)
            num_E = E_prev + self.r * diff_H
            den_E = 1 + 2 * np.pi * self.sigma_E * self.dt
            self.E[n+1, :] = (num_E - 2*np.pi*self.sigma_E*self.dt * E_prev) / den_E

            # 3) Dirichlet-условия на границах:
            self.E[n+1, 0] = 0
            self.E[n+1, -1] = 0

        return self.E, self.H
    
    def solve_analytic(self, t):
        """
        Аналитическое решение по формуле Даламбера при psi=0:
        E(t,z) = (phi(z-ct) + phi(z+ct)) / 2
        """
        zp = self.z + self.c * t
        zm = self.z - self.c * t
        E_a = 0.5 * (
            np.interp(zp, self.z, self.phi, left=0, right=0) +
            np.interp(zm, self.z, self.phi, left=0, right=0)
        )
        return E_a


    def plot(self, ispml=False, t_index=None, with_H=False, save=False, show=True, filename='wave_pml'):
        """
        Построение графика E (и опционально H) в момент t_index.
        Если t_index=None, берётся последний шаг.
        """
        if t_index is None:
            t_index = self.Nt
        t = t_index * self.dt

        E_an = self.solve_analytic(t)

        plt.figure(figsize=(8,5))
        plt.plot(self.z, self.E[t_index,:], label=f"Численное E | {'pml' if ispml else 'no-pml'}")
        plt.plot(self.z, E_an, '--', label="Аналитическое E")

        if with_H:
            # отобразим H в точках z_half
            z_half = self.z[:-1] + 0.5*self.h
            plt.plot(z_half, self.H[t_index-1,:], label='H (числ.)')

        plt.xlabel('z')
        plt.ylabel('Поля')
        plt.title(f'Поля в момент t={t:.3f}')
        plt.legend()
        plt.grid(True)

        if ispml:
            dir_name = 'wave_pml'
        else:
            dir_name = 'wave_nopml'
        if save:
            if not os.path.isdir(f'graph'):
                os.mkdir(f'graph')
            if not os.path.isdir(f'graph/{dir_name}'):
                os.mkdir(f'graph/{dir_name}')
            plt.savefig(f"graph/{dir_name}/{filename}.png")
        if show:
            plt.show()
