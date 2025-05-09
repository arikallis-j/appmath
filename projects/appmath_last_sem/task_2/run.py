import numpy as np

from classes import HeatEquationVariantA
from classes import SpectralHeatSolver

# Задаём параметры
a, b = 1.0, 1.0
N, M = 100, 100
delta = 1.0

# Источник f(x,y) для варианта A
def f_func(X, Y):
    return np.exp(-((X - a/2)**2 + (Y - b/2)**2) / delta**2)

def main():
    # # Вариант А
    # Инициализируем и решаем задачу
    heat = HeatEquationVariantA(a, b, N, M, delta, f_func)
    heat.compute_rhs_coeffs()        # DCT-II для правой части Фурье-преобразования :contentReference[oaicite:0]{index=0}
    heat.solve()                     # Решение параметрической трёхдиагональной СЛАУ методом solve_banded :contentReference[oaicite:1]{index=1}
    u = heat.reconstruct_solution()  # Обратное DCT-III для восстановления u(x,y) :contentReference[oaicite:2]{index=2}

    # Визуализация результата
    heat.plot_temperature(save=True)          # Контурный 2D-график через contourf :contentReference[oaicite:3]{index=3}

    # # Общая Задача
    solver = SpectralHeatSolver(a, b, N, M, delta, f_func)
    u = solver.solve()
    solver.plot(save=True)

if __name__ == "__main__":
    main()