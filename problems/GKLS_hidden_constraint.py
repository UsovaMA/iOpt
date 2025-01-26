from iOpt.problem import Problem
from problems.GKLS_function.gkls_function import GKLSClass, GKLSFuncionType, GKLSFunction
from iOpt.trial import Point, FunctionValue, Trial

import random

class GKLSHiddenConstraint(Problem):
    """
    GKLS-generator, allows to generate multi-extremal optimization problems with known properties in advance:
    The number of local minima, the sizes of their regions of attraction, the point of global minimum,
    the value of function in it, etc.

    The functions are supplemented with randomly generated regions of non-computability in the form of ellipsoids,
    when entering which the method for calculating the functional returns an exception.
    """

    def __init__(self, dimension: int,
                 functionNumber: int = 1,
                 seed: int = 20) -> None:
        """
        Constructor of the GKLS generator class with hidden constraint

        :param dimension: Task dimensionality, :math:`2 <= dimension <= 5`
        :param functionNumber: set task number, :math:`1 <= functionNumber <= 100`
        """
        random.seed(seed)
        super(GKLSHiddenConstraint, self).__init__()
        self.dimension = dimension
        self.name = "GKLSHiddenConstraint"
        self.number_of_float_variables = dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = [str(x) for x in range(self.dimension)]

        self.lower_bound_of_float_variables = dimension * [-1]
        self.upper_bound_of_float_variables = dimension * [1]

        self.function: GKLSFunction = GKLSFunction()

        self.mMaxDimension: int = 5
        self.mMinDimension: int = 2

        self.function_number: int = functionNumber
        self.num_minima: int = 10

        self.problem_class: int = GKLSClass.Simple
        self.function_class: int = GKLSFuncionType.TD

        self.function.GKLS_global_value: float = -1.0
        self.function.NumberOfLocalMinima: int = self.num_minima
        self.function.SetDimension(self.dimension)
        self.function.mFunctionType: int = self.function_class

        self.function.SetFunctionClass(self.problem_class, self.dimension)

        self.global_dist: float = self.function.GKLS_global_dist
        self.global_radius: float = self.function.GKLS_global_radius

        self.function.GKLS_parameters_check()

        self.function.SetFunctionNumber(self.function_number)

        KOfunV = FunctionValue()
        KOfunV.value = self.function.GetOptimumValue()

        KOpoint = Point(self.function.GetOptimumPoint(), [])

        self.known_optimum = [Trial(KOpoint, [KOfunV])]

        self.non_computable_regions_count = 4
        self.non_computable_elipsoid_center = []
        self.non_computable_elipsoid_sizes = []

        self.generate_non_computable_regions()

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of a function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.

        :return: Calculated value of the function at the point.
        """

        isInf: bool = False
        for i in range(self.non_computable_regions_count):
            if self.is_point_in_elipsoid(
                    point.float_variables,
                    self.non_computable_elipsoid_center[i],
                    self.non_computable_elipsoid_sizes[i]
            ):
                isInf = True
                break

        if isInf:
            raise Exception("Infinity values")  # non-computable values

        function_value.value = self.function.Calculate(point.float_variables)
        return function_value

    def is_point_in_elipsoid(self, point, center, sizes):
        res = 0.0
        for i in range(0, len(point)):
            res += ((point[i] - center[i]) * (point[i] - center[i])) / (sizes[i] * sizes[i])
        return res <= 1

    def generate_non_computable_regions(self):
        elipsoid_parameters_count = self.non_computable_regions_count * self.dimension
        optimum = self.known_optimum[0].point.float_variables

        for i in range(0, elipsoid_parameters_count):
            elipsoid_center = []
            for j in range(self.dimension):
                elipsoid_center.append(-random.random() + random.random())

            elipsoid_sizes = []
            for j in range(self.dimension):
                elipsoid_sizes.append(random.uniform(5, 25) * 0.01)

            if self.is_point_in_elipsoid(optimum, elipsoid_center, elipsoid_sizes):
                i = i - 1
                continue

            self.non_computable_elipsoid_center.append(elipsoid_center)
            self.non_computable_elipsoid_sizes.append(elipsoid_sizes)

