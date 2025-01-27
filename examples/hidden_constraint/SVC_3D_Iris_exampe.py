from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from examples.Machine_learning.SVC._3D.Problem import SVC_3D_Iris

from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
import warnings

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    x, y = load_iris(return_X_y=True)

    regularization_value_bound = {'low': 1, 'up': 6}
    kernel_coefficient_bound = {'low': -7, 'up': -3}
    penalty_type = {'penalty': ['l1', 'l2']}
    dual_value = {'dual': [True, False]}
    loss_type = {'loss': ['hinge', 'squared_hinge']}

    problem = SVC_3D_Iris.SVC_3D_Iris(x, y, regularization_value_bound, kernel_coefficient_bound, penalty_type,
                                      dual_value, loss_type)

    # Формируем параметры решателя
    params = SolverParameters(r=3.5, iters_limit=300)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Решаем задачу
    solver_info = solver.solve()
