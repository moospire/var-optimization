import pytest

from types import MethodType

from unittest.mock import Mock

from variational_optimization.core import Model
from variational_optimization.core import Objective
from variational_optimization.core import Optimizer
from variational_optimization.core import Parameters
from variational_optimization.core import SimpleGradientDescentUpdater
from variational_optimization.core import Solver
from variational_optimization.core import Tasker

# we keep normal sampler out of core for now, but include in tests
from variational_optimization.sampler import NormalSampler, DiagNormalSampler
from variational_optimization.core import SamplingDistributionDelta

from numpy import testing as np_testing

def test_optimizer():
    mock_objective = Mock()
    mock_updater = Mock()
    mock_updater.minimize.return_value = 10
    optim = Optimizer(mock_objective, mock_updater)
    return_avg = optim.run(10)
    assert return_avg == 10

def test_gradient_descent_updater():
    mock_objective = Mock()
    mock_objective.parameters = 2.
    objective_gradient = 1.
    gradient_updater = SimpleGradientDescentUpdater(0.001)
    assert gradient_updater.lr == 0.001
    gradient_updater.step(mock_objective, objective_gradient)
    assert mock_objective.parameters == 1.999

def test_objective():
    mock_model = Mock()
    mock_model.gradient.return_value = 1.
    mock_model.eps.side_effect = [[0.], 0., 0.]
    tasker = Tasker()
    objective = Objective(mock_model, tasker)

    # make sure can call the functions
    samples = [[1., 2., 5.]]
    costs = objective.costs_at(samples)

    samples = objective.get_samples(3)
    mock_model.sample.assert_called_with(None)

    samples = [[1., 2., 5.],[3., 4., 8.]]
    costs = [1., 2.]
    gradient = objective.gradient(costs, samples)
    mock_model.gradient.assert_called_with([3., 4., 8.])
    # we want to make sure we return the average costs
    # since gradient set to 1.
    assert gradient == 1.

def test_create_solver():
    mock_model = Mock()
    mock_updater = Mock()
    mock_cost_manager = Mock()
    solver = Solver(mock_model, mock_updater, num_iters=100, num_samples=10, logger=None, tasker=mock_cost_manager)

def test_proto_cost_function_manager():
    mock_function = Mock()
    mock_executor = Mock()
    cost_manager = Tasker(mock_function, executor=None)
    return_vals = cost_manager.submit([None]*10)
    assert len(return_vals) == 10.
    cost_manager = Tasker(mock_function, executor=mock_executor)
    return_vals = cost_manager.submit([None]*10)
    assert len(return_vals) == 10.

# note that log(1) == 0 for init
# for rough ideas about variance conversion:
# consider log(N)**2*2 is roughly == N for 3-20
# therefore, use sqrt(N/2) to estimate log val for init
def test_normal_distribution_conversions():
    init = [1., 1., 0., 0., 0.]
    normal = NormalSampler([1., 1.],[1., 1.])
    assert normal.parameters == init
    assert normal.mu == [1., 1.]
    assert (normal.Sigma == [[1., 0.],[0., 1.]]).all()
    assert normal.parameters_from_mu_Sigma(normal.mu, normal.Sigma) == normal.parameters

def test_diag_normal_distribution():
    normal = DiagNormalSampler(mus=[1., 1.],vars=[1., 1.])
    assert (normal.parameters == [1., 1., 0., 0.]).all()
    assert (normal.mu == [1., 1.]).all()
    assert (normal.cov == [1., 1.]).all()
    np_testing.assert_almost_equal(normal.d_E_dmu([0., 0.]), [-1.41421356, -1.41421356])
    np_testing.assert_almost_equal(normal.d_E_dS([0., 1.]),[ 0. , -0.5])

def test_normal_distribution_gradient_sampler():
    init = [1., 1., -.5, 2., 2.]
    normal = NormalSampler(scale_grad=False)
    normal.parameters = init
    np_testing.assert_almost_equal(normal.inv_Sigma, [[0.1353353, 0.       ],
                                                      [0.       , 0.1353353]])
    samples = normal.sample(3)
    assert len(samples) == 3
    assert len(samples[0]) == 2
    print(normal)
    assert normal.d_E_dmu([1., 1.]) == [0., 0.]
    # these are empirical results... should confirm...
    np_testing.assert_almost_equal(normal.d_E_dmu([0., 0.]), [-0.1353353, -0.1353353])
    np_testing.assert_almost_equal(normal.d_E_dS([0., 0.]),[[-0.0585098,  0.0091578],
                                                            [ 0.0091578, -0.0585098]])
    np_testing.assert_almost_equal(normal.d_E_dS([1., 1.]), [[-0.0676676,  0.       ],
                                                             [ 0.       , -0.0676676]])

    # now try with gradient scaling
    normal = NormalSampler(scale_grad=True)
    normal.parameters = init
    np_testing.assert_almost_equal(normal.inv_Sigma, [[0.1353353, 0.       ],
                                                      [0.       , 0.1353353]])
    samples = normal.sample(3)
    assert len(samples) == 3
    assert len(samples[0]) == 2
    print(normal)
    assert normal.d_E_dmu([1., 1.]) == [0., 0.]
    # these are empirical results... should confirm...
    np_testing.assert_almost_equal(normal.d_E_dmu([0., 0.]), [-0.191393, -0.191393])
    np_testing.assert_almost_equal(normal.d_E_dS([0., 0.]),[[-0.0585098,  0.0091578],
                                                            [ 0.0091578, -0.0585098]])
    np_testing.assert_almost_equal(normal.d_E_dS([1., 1.]), [[-0.0676676,  0.       ],
                                                             [ 0.       , -0.0676676]])

def test_iteration_hook_wrapper():
    model = Model()
    model.counter = 0
    # Python 3.8 allows assignment in lambdas but keeping this in a named function for compatibility
    def iteration_hook(self, p, s, c):
        self.counter += 1
    model.iteration_hook = MethodType(iteration_hook, model)
    model.iteration_hook_with_return_values([Parameters()], [], [])
    assert model.counter == 1
