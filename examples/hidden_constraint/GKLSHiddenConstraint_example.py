from problems.GKLS_hidden_constraint import GKLSHiddenConstraint
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции из GKLS генератора с добавленными областями невычислимости с номером 39
    """
    # создание объекта задачи
    problem = GKLSHiddenConstraint(dimension=2, functionNumber=39)

    # Формируем параметры решателя
    params = SolverParameters(r=3.5, eps=0.01, iters_limit=10000, refine_solution=True)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    #apl = StaticPainterNDListener("gkls_interp.png", mode='lines layers', calc='objective function')
    #solver.add_listener(apl)

    # Решение задачи
    solver.solve()
