import numpy as np
from scipy.fftpack import dct, idct, dst, idst
from scipy.linalg import solve_banded

import matplotlib.pyplot as plt
import os

class FourierTransform:
    """
    Класс для дискретных преобразований Фурье (косинусных и синусных) по оси y.
    Параметр M задает число отрезков по y (0..M).
    """
    def __init__(self, M):
        self.M = M

    def forward_cos_y(self, u):
        """
        Прямое косинусное преобразование вдоль оси y:
        U_im = (2/M) * sum_{j=0..M} u_{i,j} * cos(pi * m * j / M)
        Возвращает массив shape (N+1, M+1) -> (N+1, M+1)
        """
        # По умолчанию scipy.fftpack.dct типа II с нормализацией ``ortho`` соответствует нужному
        return dct(u, type=2, axis=1, norm='ortho') * 2 / self.M

    def inverse_cos_y(self, U):
        """
        Обратное косинусное преобразование вдоль оси y:
        u_{i,j} = U_{i,0}/2 + sum_{m=1..M-1} U_{i,m} * cos(pi * m * j / M)
        """
        # scipy.fftpack.idct типа III с норм='ortho' дает обратное по dct II
        return idct(U * (self.M/2), type=3, axis=1, norm='ortho')

    # Для общего случая можно добавить forward_sin_y и inverse_sin_y при необходимости


class HeatEquationVariantA:
    """
    Численное решение стационарного уравнения теплопроводности (вариант A)
    c Neumann-условиями на трех сторонах и Dirichlet на x=a.
    """
    def __init__(self, a, b, N, M, delta, f_func):
        self.a = a
        self.b = b
        self.N = N
        self.M = M
        self.delta = delta
        self.f_func = f_func
        self.hx = a / N
        self.hy = b / M
        # Сетки по x и y
        self.x = np.linspace(0, a, N+1)
        self.y = np.linspace(0, b, M+1)
        # Здесь будут храниться коэффициенты
        self.F = np.zeros((N+1, M+1))  # правая часть
        self.U = np.zeros((N+1, M+1))  # коэффициенты U_im
        self.u = None                  # решение u_ij
        self.ft = FourierTransform(M)

    def compute_rhs_coeffs(self):
        """
        Вычисление коэффициентов F_{i,m} по формулам (6)-(7)
        через дискретное косинус-преобразование.
        """
        # Формируем массив f_ij на сетке
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        f_grid = self.f_func(X, Y)
        # Прямое косинус-преобразование по y
        # F = (2/M) * sum f_ij cos(pi m j / M)
        self.F = self.ft.forward_cos_y(f_grid)

    def solve(self):
        """
        Решение трёхдиагональных систем (8) для каждого m методом прогонки.
        """
        # Предрасчёт rho_m = 4 sin^2(pi m / (2M))
        m = np.arange(0, self.M+1)
        rho = 4 * np.sin(np.pi * m / (2*self.M))**2

        # Формируем коэффициенты трёхдиагональной матрицы в форме banded для solve_banded
        # Диагонали: lower = 1/hx^2, main = -2/hx^2 + rho[m]/hy^2, upper = 1/hx^2
        N = self.N
        # Банда: (3, N+1, M+1) but мы решаем по каждому m отдельно
        for m_idx in range(self.M+1):
            # Диагонали для данного m
            lower = np.full(N+1, 1/self.hx**2)
            main  = np.full(N+1, -2/self.hx**2 + rho[m_idx]/self.hy**2)
            upper = np.full(N+1, 1/self.hx**2)
            # Граничные условия: U[0]=U[1], U[N]=0
            main[0] = 1; upper[0] = -1; lower[-1] = 0; main[-1] = 1; upper[-1] = 0
            # Собираем banded-матрицу: shape (3, N+1)
            ab = np.zeros((3, N+1))
            ab[0,1:] = upper[:-1]
            ab[1]   = main
            ab[2,:-1] = lower[1:]
            # Правая часть
            Fm = self.F[:, m_idx]
            # Вносим граничное условие U[0]=U[1] => RHS[0]=0
            Fm[0] = 0; Fm[-1] = 0
            # Решаем
            Ucol = solve_banded((1,1), ab, Fm)
            self.U[:, m_idx] = Ucol

    def reconstruct_solution(self):
        """
        Восстановление u_{i,j} по формуле (4) через обратное косинусное преобразование.
        """
        self.u = self.ft.inverse_cos_y(self.U)
        return self.u

    def plot_temperature(self, save=False, filename='temperature'):
        """
        Построение 2D-графика распределения температуры u(x,y).
        """
        if self.u is None:
            raise RuntimeError("Сначала вызовите reconstruct_solution().")
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        plt.figure(figsize=(8,6))
        cp = plt.contourf(X, Y, self.u, levels=50)
        plt.colorbar(cp)
        plt.xlabel('x'); plt.ylabel('y')
        plt.title('Распределение температуры u(x,y)')

        if save:
            if not os.path.isdir(f'graph'):
                os.mkdir(f'graph')
            if not os.path.isdir(f'graph/temp'):
                os.mkdir(f'graph/temp')
            plt.savefig(f"graph/temp/{filename}.png")

        plt.show()



class SpectralHeatSolver:
    """
    Общая задача стационарного уравнения теплопроводности
    с нулевыми (Дирихле) граничными условиями на всех сторонах.
    """

    def __init__(self, a, b, N, M, delta, f_func):
        self.a, self.b = a, b
        self.N, self.M = N, M
        self.delta = delta
        self.f_func = f_func

        self.hx = a / N
        self.hy = b / M
        # координатные сетки
        self.x = np.linspace(0, a, N+1)
        self.y = np.linspace(0, b, M+1)
        # поля
        self.u = np.zeros((N+1, M+1))

    def compute_rhs_spectral(self):
        """
        Вычислить Phi[l,m] — коэффициенты двойного синус-преобразования
        правой части f_ij.
        """
        # сетка f_ij
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        f = self.f_func(X, Y)

        # внутренняя область 1..N-1, 1..M-1
        f_int = f[1:-1, 1:-1]

        # DST-I по оси y (inner j), затем DST-I по оси x (inner i)
        Fm = dst(f_int, type=1, axis=1, norm=None)  # shape (N-1, M-1)
        Phi = dst(Fm, type=1, axis=0, norm=None)     # shape (N-1, M-1)

        # нормировка: множители 2/N и 2/M
        Phi *= (2/self.N) * (2/self.M)
        return Phi  # indexed [l-1, m-1] ~ l=1..N-1, m=1..M-1

    def solve(self):
        # спектр правой части
        Phi = self.compute_rhs_spectral()

        # предрасчет знаменателей
        l = np.arange(1, self.N)
        m = np.arange(1, self.M)
        gamma = 4 * np.sin(np.pi * l/(2*self.N))**2
        lam   = 4 * np.sin(np.pi * m/(2*self.M))**2

        # формируем V[l,m]
        # broadcasting: denom[l,m] = gamma[l]/hx^2 + lam[m]/hy^2
        denom = gamma[:,None]/self.hx**2 + lam[None,:]/self.hy**2
        V = Phi / denom  # размер (N-1, M-1)

        # обратное преобразование: сначала по x, потом по y
        U = idst(V, type=1, axis=0, norm=None)
        u_inner = idst(U, type=1, axis=1, norm=None)

        # нормирование обратное DST-I: умножить на 1/(2*(N)) и 1/(2*(M))
        u_inner *= 1/(2*self.N) * 1/(2*self.M)

        # вкладываем решение внутрь массива, граница нулевая
        self.u[1:-1, 1:-1] = u_inner

        return self.u

    def plot(self, save=False, filename='u_general'):
        """Сохранить контурный график решения."""
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        plt.figure(figsize=(6,5))
        cp = plt.contourf(X, Y, self.u, levels=50)
        plt.colorbar(cp)
        plt.xlabel('x'); plt.ylabel('y')
        plt.title('Общий случай: u(x,y)')

        if save:
            if not os.path.isdir(f'graph'):
                os.mkdir(f'graph')
            if not os.path.isdir(f'graph/spectra'):
                os.mkdir(f'graph/spectra')
            plt.savefig(f"graph/spectra/{filename}.png")

        plt.show()




