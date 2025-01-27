import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from typing import Dict, List

class SVC_3D_Iris(Problem):
    """
    Класс SVC_3D представляет возможность поиска оптимального набора гиперпараметров алгоритма
      C-Support Vector Classification.
      Найденные параметры являются оптимальными при варьировании параматра регуляризации
      (Regularization parameter С) значения коэфицента ядра (gamma) и типа ядра (kernel)
    """

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray,
                 regularization_bound: Dict[str, float],
                 kernel_coefficient_bound: Dict[str, float],
                 penalty_type: Dict[str, List[str]],
                 dual_value: Dict[str, List[bool]],
                 loss_type: Dict[str, List[str]]
                 ):
        """
        Конструктор класса SVC_3D

        :param x_dataset: входные данные обучающе выборки метода SVC
        :param y_dataset: выходные данные обучающе выборки метода SVC
        :param kernel_coefficient_bound: Значение параметра регуляризации
        :param regularization_bound: Границы изменения значений коэфицента ядра (low - нижняя граница, up - верхняя)
        :param penalty_type: Тип penalty
        """
        super(SVC_3D_Iris, self).__init__()
        self.dimension = 3
        self.number_of_float_variables = 1
        self.number_of_discrete_variables = 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        #self.float_variable_names = np.array(["Regularization parameter", "Kernel coefficient"], dtype=str)
        self.float_variable_names = np.array(["Regularization parameter"], dtype=str)
        self.lower_bound_of_float_variables = np.array([regularization_bound['low']],
                                                   dtype=np.double)
        self.upper_bound_of_float_variables = np.array([regularization_bound['up']],
                                                   dtype=np.double)
        # self.lower_bound_of_float_variables = np.array([regularization_bound['low'], kernel_coefficient_bound['low']],
        #                                            dtype=np.double)
        # self.upper_bound_of_float_variables = np.array([regularization_bound['up'], kernel_coefficient_bound['up']],
        #                                            dtype=np.double)

        # self.discrete_variable_names.append('penalty')
        # self.discrete_variable_values.append(penalty_type['penalty'])

        self.discrete_variable_names.append('loss')
        self.discrete_variable_values.append(loss_type['loss'])
        self.discrete_variable_names.append('dual')
        self.discrete_variable_values.append(dual_value['dual'])

        print(self.discrete_variable_values)


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param function_value: объект хранения значения целевой функции в точке
        """
        print(point.discrete_variables[0])
        #cs, gammas = point.float_variables[0], point.float_variables[1]
        cs = point.float_variables[0]

        loss_type = point.discrete_variables[0]
        dual_value = point.discrete_variables[1]

        #clf = SVC(C=10 ** cs, gamma=10 ** gammas, penalty=penalty_type, loss="hinge", dual=True)
        clf = LinearSVC(C=10 ** cs,  penalty = "l2", loss=loss_type, dual=dual_value)

        function_value.value = -cross_val_score(clf, self.x, self.y, scoring='f1_macro').mean()
        return function_value
