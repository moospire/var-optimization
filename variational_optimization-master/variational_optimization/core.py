import numpy as np
from variational_optimization.ops import ops


class SamplingDistributionDelta:
    def __init__(self, parameters_list):
        self.parameters = parameters_list

    @property
    def mode(self):
        raise NotImplementedError

    @staticmethod
    def params_from_means_vars(means, variances):
        raise NotImplementedError

    def calc_mul(self, x):
        scaled_params = []
        for param in self.parameters:
            scaled_params.append(x * param)
        return SamplingDistributionDelta(scaled_params)

    def __mul__(self, x):
        return self.calc_mul(x)

    def __rmul__(self, x):
        return self.calc_mul(x)

    def add(self, x):
        if isinstance(x, int):
            return self
        inc_params = []
        for param, x_i in zip(self.parameters, x.parameters):
            inc_params.append(param + x_i)

        return type(self)(inc_params)

    def __radd__(self, x):
        return self.add(x)

    def __add__(self, x):
        return self.add(x)

    def __str__(self):
        return str(self.parameters)


class SamplingDistribution:
    def __init__(self, parameters_list=None):
        # create a name for serialization/type checking
        self.name = "Abstract"
        self._parameters = parameters_list
        self.gradient_cls = SamplingDistributionDelta
        # define the sample output dimension
        self.dim = None

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self.validate_parameters(value)
        self.update_properties_from_parameters(value)
        self._parameters = value

    def update_properties_from_parameters(self, parameters=None):
        pass

    def validate_parameters(self, parameters):
        if np.isscalar(parameters): raise TypeError('parameters is not array-like')

    # generate m self.dim dimension samples
    def sample(self, m, seed):
        if seed is None:
            raise NotImplementedError
        else:
            raise NotImplementedError

    # return a sampling distribution delta
    def gradient(self, sample):
        raise NotImplementedError

    def eps(self, sample):
        return np.zeros(len(self.parameters))

    def calc_parameters_updated(self, inc_parameters):
        if self.parameters is None:
            raise AttributeError("parameters must first be initialized!")
        inc_params = []
        for param, x_i in zip(self.parameters, inc_parameters):
            inc_params.append(param + x_i)

        return inc_params

    def __add__(self, x):
        if self.parameters is None:
            raise AttributeError("parameters must first be initialized!")
        if not isinstance(x, self.gradient_cls):
            raise Exception(
                "Types of sampling distribution and delta are not compatible"
            )
        # this will trigger a projection so it all works out
        # this is syntax is also kinda weird so watch out!
        # probably would do something weird with python2
        return type(self)(self.calc_parameters_updated(x.parameters))

    def __iadd__(self, x):
        return self + x

    def __str__(self):
        if self.parameters is None:
            raise AttributeError("parameters must first be initialized!")
        return str(self.parameters)


class Parameters:
    # container class for passing state
    # proto conversion helpers
    def as_proto(self, proto_message_cls):
        proto_message = proto_message_cls()
        proto_keys = proto_message.DESCRIPTOR.fields_by_name.keys()
        for k, v in self.__dict__.items():
            if k in proto_keys:
                if type(getattr(proto_message, k)) == int:
                    setattr(proto_message, k, int(v))
                elif type(getattr(proto_message, k)) == float:
                    setattr(proto_message, k, float(v))
                elif type(getattr(proto_message, k)) == str:
                    setattr(proto_message, k, str(v))
                else:
                    # try to cast as Numpy
                    self.np_to_proto(v, getattr(proto_message, k))
        return proto_message

    @staticmethod
    def np_to_proto(np_arr, proto):
        proto.dtype = str(np_arr.dtype)
        proto.shape[:] = np_arr.shape
        proto.data = np_arr.tobytes()

    @staticmethod
    def proto_to_np(proto):
        b = np.frombuffer(proto.data, dtype=proto.dtype).reshape(list(proto.shape))
        return b

    def __add__(self, x):
        """
            overload addition to combine parameters
        """
        assert issubclass(x.__class__, Parameters)
        p = Parameters()
        for k, v in self.__dict__.items():
            setattr(p, k, v + getattr(x, k))
        return p

    def __str__(self):
        return str(self.__dict__)


class Variable:
    def __init__(self, sampler, indices=[]):
        self.sampler = sampler
        self.out_indices = indices


class Model:
    def __init__(self):
        self._trainables = None
        self.blacklist = ["_trainables"]


    def apply_filters_self_dict_items(self, filter_list):
        """The filters defined in the class are applied by the following function"""
        items = self.__dict__.items()
        return filter( lambda x: all(f(x) for f in filter_list), items)

    @staticmethod
    def contains_sampling_distribution(item):
        """Only get sampling distributions"""
        k,v = item
        return issubclass(v.__class__, SamplingDistribution)

    def not_in_blacklist(self, item):
        """Only get class attributes not in the blacklist"""
        k,v = item
        return k not in self.blacklist

    @staticmethod
    def contains_variable(item):
        """Makes sure items are in the blacklist"""
        k,v = item
        return issubclass(v.__class__, Variable)

    @staticmethod
    def contains_key(keys):
        """Only get class attributes not in the blacklist"""
        return lambda item: item[0] in keys

    def sample(self, seed=None):
        # get all the variables
        p = Parameters()
        # first fill in sampler values
        sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
        raw_samples = map(lambda item: (item[1], item[1].sample(1)[0]), sampler_items)

        # then construct the sample_dict from other attributes
        allowed_items = self.apply_filters_self_dict_items(
                      [lambda x: not self.contains_sampling_distribution(x),
                       self.not_in_blacklist])

        # get the corresponding samples from the variables
        variable_items = filter(self.contains_variable, allowed_items)
        for k, v in variable_items:
            for sampler, samples in raw_samples:
                if sampler == v.sampler:
                    value = np.array(samples)[v.out_indices]
                    setattr(p, k, value)

        # set all the attributes that aren't variables
        [setattr(p, k, v) for k, v in filter(lambda x: not self.contains_variable(x), allowed_items)]
        return p

    # TODO: The following methods have very similar structure... how to reduce repitition?
    def gradient(self, p):
        # use to fill in vals for gradients
        # first initialize in sampler output arrays
        sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
        sampler_output = list(map(lambda item: (item[1], np.zeros(item[1].dim)), sampler_items))
        # lookup matches for sampler outputs
        # then construct the sample_dict from other attributes
        allowed_items = self.apply_filters_self_dict_items(
                      [lambda x: not self.contains_sampling_distribution(x),
                       self.not_in_blacklist,
                       self.contains_variable,
                       self.contains_key(p.__dict__.keys())])

        for k, v in allowed_items:
            for sampler, output_list in sampler_output:
                if sampler == v.sampler:
                    output_list[v.out_indices] = getattr(p, k)

        # calculate gradient with respect to the sample message
        # first get the parameters for each sampler
        grads = [np.array(sampler.gradient(output).parameters) for sampler, output in sampler_output]

        grad = np.concatenate(grads)
        return grad

    def eps(self, p):
        # use to fill in vals for eps
        # first initialize in sampler output arrays
        sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
        sampler_output = list(map(lambda item: (item[1], np.zeros(item[1].dim)), sampler_items))
        # lookup matches for sampler outputs
        # then construct the sample_dict from other attributes
        allowed_items = self.apply_filters_self_dict_items(
                      [lambda x: not self.contains_sampling_distribution(x),
                       self.not_in_blacklist,
                       self.contains_variable,
                       self.contains_key(p.__dict__.keys())])

        for k, v in allowed_items:
            for sampler, output_list in sampler_output:
                if sampler == v.sampler:
                    output_list[v.out_indices] = getattr(p, k)

        # calculate gradient with respect to the sample message
        # first get the parameters for each sampler
        eps = [np.array(sampler.eps(output)) for sampler, output in sampler_output]

        eps = np.concatenate(eps)
        return eps

    def prob_at(self, p):
        # first initialize in sampler output arrays
        sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
        sampler_output = list(map(lambda item: (item[1], np.zeros(item[1].dim)), sampler_items))
        # iterate through model attributes and lookup matches for sampler outputs
        allowed_items = self.apply_filters_self_dict_items(
                      [lambda x: not self.contains_sampling_distribution(x),
                       self.not_in_blacklist,
                       self.contains_variable,
                       self.contains_key(p.__dict__.keys())])
        for k, v in allowed_items:
            for sampler, output_list in sampler_output:
                if sampler == v.sampler:
                    output_list[v.out_indices] = getattr(p, k)

        # calculate gradient with respect to the sample message
        # first get the parameters for each sampler
        probs = [np.array(sampler.prob_at(output)) for sampler, output in sampler_output]
        return np.prod(np.array(probs), axis=0)

    @property
    def trainables(self):
        if self._trainables is None:
            sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
            params = list(map(lambda item: np.array(item[1].parameters), sampler_items))
            self._trainables = np.concatenate(params)
        return self._trainables

    @trainables.setter
    def trainables(self, parameters):
        ind=0
        sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
        for k,v in sampler_items:
            # check if Model contains sampler classes
            # update parameters based on expected length
            v.parameters = parameters[ind : ind + len(v.parameters)]
            ind += len(v.parameters)
        self._trainables = None


    def _get_params(self, return_value):
        if hasattr(return_value, 'DESCRIPTOR'):
            parameters = Parameters()
            proto_keys = return_value.DESCRIPTOR.fields_by_name.keys()
            for key in proto_keys:
                if type(getattr(return_value, key)) in [int, float, str]:
                    setattr(parameters, key, getattr(return_value, key))
                else:
                    setattr(parameters, key, parameters.proto_to_np(getattr(return_value, key)))
        else:
            return return_value


    def loss_with_return_values(self, return_values):
        return self.loss(self._get_params(return_values))

    def loss(self, r):
        raise NotImplementedError

    def iteration_hook_with_return_values(self, return_values, samples, costs):
        parameters = list(map(self._get_params, return_values))
        self.iteration_hook(parameters, samples, costs)

    def iteration_hook(self, parameters, samples, costs):
        pass

    @property
    def mode(self):
        # get all the variables
        p = Parameters()
        # first fill in sampler values
        sampler_items = self.apply_filters_self_dict_items([self.contains_sampling_distribution])
        raw_modes = map(lambda item: (item[1], item[1].mode), sampler_items)

        # then construct the sample_dict from other attributes
        allowed_items = self.apply_filters_self_dict_items(
                      [lambda x: not self.contains_sampling_distribution(x),
                       self.not_in_blacklist])

        # get the corresponding samples from the variables
        variable_items = filter(self.contains_variable, allowed_items)
        for k, v in variable_items:
            for sampler, mode in raw_modes:
                if sampler == v.sampler:
                    value = np.array(mode)[v.out_indices]
                    setattr(p, k, value)

        # set all the attributes that aren't variables
        [setattr(p, k, v) for k, v in filter(lambda x: not self.contains_variable(x), allowed_items)]
        return p

    def __str__(self):
        variable_str = ""
        sampler_str = ""
        other_attr = ""
        for k, v in self.__dict__.items():
            if issubclass(v.__class__, SamplingDistribution):
                sampler_str += "    {}:\n    {}\n".format(k, v)
            elif issubclass(v.__class__, Variable):
                variable_str += "    {}:\n    {}\n".format(
                    k, np.array(v.sampler.mode)[v.out_indices]
                )
            else:
                if type(v) in [float, int, str, np.float64]:
                    other_attr += "    {}:\n    {}\n".format(k, v)
                elif type(v) is type(np.array([])):
                    other_attr += "    {}:\n    {}\n".format(k, v.shape)
        return "Samplers:\n  {}Variables:\n  {}Other:\n   {}".format(
            sampler_str, variable_str, other_attr
        )


class Tasker:
    def __init__(self, task_function=None, executor=None):
        # task_function is a function that can convert a message into a response
        self.task_function = task_function
        # with no cost_function, return identity
        if self.task_function is None:
            self.task_function = lambda x: x
        self.executor = executor

    def submit(self, m):
        # get value from stub, use thread executor if available
        if self.executor is None:
            vals = [self.task_function(message) for message in m]
        else:
            # get futures in a list
            future_vals = [
                self.executor.submit(self.task_function, message) for message in m
            ]
            # get result of the futures in a list (TODO: keep as iter?)
            vals = list(map(lambda x: x.result(), future_vals))
        return vals


class Objective:
    def __init__(
        self, model, tasker, importance_sample_buffer_size=0
    ):
        self.model = model
        self.tasker = tasker
        self.importance_sample_buffer = []
        self.importance_sample_buffer_size = importance_sample_buffer_size

    @property
    def parameters(self):
        return self.model.trainables

    @parameters.setter
    def parameters(self, x):
        self.model.trainables = x

    # consider interface -> send out job requests -> sync (which handles sending updated costs)
    def costs_at(self, samples):
        return_values = self.tasker.submit(samples)
        costs = list(map(self.model.loss_with_return_values, return_values))
        self.model.iteration_hook_with_return_values(return_values, samples, costs)
        return costs

    def get_samples(self, m, seed=None):
        samples = list(map(self.model.sample, [None] * m))
        return samples

    def update_gradient(self, diff_gradient):
        return -self.model.gradient(self.model.mode + diff_gradient)

    def gradient(self, costs, samples):
        # get mode cost if model.eps has relevant values
        # TODO: a better system to scale the gradients
        if None in self.model.eps(self.model.mode):
            mode_cost = self.costs_at([self.model.mode])[0]
            N_g = 1.
        else:
            mode_cost = 0.
            N_g = sum(costs)
        sum_g = sum([c * self.model.gradient(s) - mode_cost * self.model.eps(s) for c, s in zip(costs, samples)])
        # unbiased, but less stable?
        # N_g = len(costs)
        # importance sample
        # for importance sampling we use weight: N(s)/q(s)
        # N(s) is current probability of sample (probs)
        # we keep a running tab on grads_weighted, so that becomes q(s)
        # we need to keep (s, c), because we care about the new gradient
        if (
            self.importance_sample_buffer_size > 0
            and len(self.importance_sample_buffer) >= self.importance_sample_buffer_size
        ):
            sample_weights = [self.model.prob_at(s) / p for c, s, p in self.importance_sample_buffer]
            # filter out nan values and p < small probability threshold
            sample_weights = [w if w > 1e-8 else 0. for w in sample_weights]
            f = [
                c * self.model.gradient(s) for c, s, p in self.importance_sample_buffer
            ]
            sum_i = sum([f * w for f, w in zip(f, sample_weights)])
            N_i = sum(sample_weights)
            # unbiased , but less stable?
            # N_i = len(self.importance_sample_buffer)
            self.importance_sample_buffer = self.importance_sample_buffer[len(costs) :]
        else:
            sum_i = 0.0
            N_i = 0.0
        probs = [self.model.prob_at(s) for s in samples]

        if self.importance_sample_buffer_size > 0:
            [
                self.importance_sample_buffer.append((c, s, p))
                for c, s, p in zip(costs, samples, probs)
            ]
        return (1.0 / (N_g + N_i)) * (sum_g + sum_i)


class Optimizer:
    def __init__(self, objective, updater, num_samples=10):
        self.objective = objective
        self.updater = updater
        self.num_samples = num_samples

    def run(self, iters):
        fun, x0, jac, hess = self.problem_factory()
        result = self.updater.minimize(
            fun, x0, iters, args=(self.objective, self.num_samples), jac=jac, hess=hess
        )
        return result

    def problem_factory(self):
        x0 = self.objective.parameters
        # hash x to determine sampling params?
        def fun(x, objective, num_samples):
            seed = str(x)
            objective.parameters = x
            samples = objective.get_samples(num_samples, seed=seed)
            cost = objective.costs_at(samples)
            N = float(len(cost))
            total_cost = (1 / N) * sum(cost)
            return total_cost

        def jac(x, objective, num_samples):
            seed = str(x)
            objective.parameters = x
            samples = objective.get_samples(num_samples, seed=seed)
            cost = objective.costs_at(samples)
            objective_gradient = objective.gradient(cost, samples)
            return np.array(objective_gradient)

        return fun, x0, jac, None


class LoggingOptimizer:
    def __init__(self, optimizer, logger):
        self.optimizer = optimizer
        self.optimizer.updater.logger = logger
        self.optimizer.updater.num_print_steps = logger.num_print_steps

    def run(self, iters):
        result = self.optimizer.run(iters)
        # generate movie?
        if (
            hasattr(self.optimizer.updater.logger, "save_vis")
            and self.optimizer.updater.logger.save_vis
        ):
            ops.gen_vis_mp4(self.optimizer.updater.logger.save_key)
        return result


# helper class for usability
class Solver:
    def __init__(
        self, model, updater, num_iters=100, num_samples=100, logger=None, tasker=None
    ):
        self.model = model
        self.updater = updater
        self.logger = logger
        self.num_iters = num_iters
        if tasker is None:
            tasker = Tasker()
        self.objective = Objective(model, tasker)
        self.optimizer = Optimizer(
            self.objective, self.updater, num_samples=num_samples
        )
        if self.logger is not None:
            self.optimizer = LoggingOptimizer(self.optimizer, self.logger)

    def run(self, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iters
        return self.optimizer.run(num_iters)


# simple gradient descent
class Updater:
    def __init__(self, logger=None):
        self.step_index = 0
        self.logger = logger
        self.num_print_steps = 1

    def minimize(self, fun, x0, iters, args=(None, None), jac=None, hess=None):
        raise NotImplementedError


class SimpleGradientDescentUpdater(Updater):
    def __init__(self, lr):
        Updater.__init__(self)
        self.lr = lr

    def minimize(self, fun, x0, iters, args=(None, None), jac=None, hess=None):
        x = x0
        cost = None
        objective = args[0]
        if iters < self.num_print_steps:
            self.num_print_steps = iters
        for n in range(iters):
            objective_gradient = jac(x, *args)
            objective = self.step(objective, objective_gradient)
            x = objective.parameters

            if self.logger is not None and n % int(iters / self.num_print_steps) == 0:
                cost = fun(x, *args)
                self.logger.process_iter(n, cost, objective, iters=iters)
        if self.logger is not None:
            cost = fun(x, *args)
            self.logger.process_iter(iters, cost, objective)
        return cost

    def step(self, objective, objective_gradient):
        # maybe want to add a small amount of noise to objective_gradient,
        # making the updates more stochastic (since deterministic in x)
        sampling_distribution_delta = -self.lr * objective_gradient
        # order of addition matters here (since overriding +)
        objective.parameters = objective.parameters + sampling_distribution_delta
        self.step_index += 1
        return objective


class AdamUpdater(SimpleGradientDescentUpdater):
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        SimpleGradientDescentUpdater.__init__(self, lr)
        self.m_weight = beta1
        self.v_weight = beta2
        self.m = None
        self.v = None

    def step(self, objective, objective_gradient):
        # make sure objective_gradient is numpy array
        # maybe want to add a small amount of noise to objective_gradient,
        # making the updates more stochastic (since deterministic in x)
        if self.m is None:
            self.m = np.zeros(objective_gradient.shape)
        if self.v is None:
            self.v = np.zeros(objective_gradient.shape)
        self.m = self.m_weight * self.m + (1 - self.m_weight) * objective_gradient
        self.v = self.v_weight * self.v + (1 - self.v_weight) * np.power(
            objective_gradient, 2
        )
        sampling_distribution_delta = -self.lr * self.m / (np.sqrt(self.v) + 1e-8)
        # order of addition matters here (since overriding +)
        objective.parameters = objective.parameters + sampling_distribution_delta
        self.step_index += 1
        return objective


class ExternalUpdater(Updater):
    def __init__(self, minimize_interface, method):
        Updater.__init__(self)
        self.minimize_interface = minimize_interface
        self.method = method

    def minimize(self, fun, x0, iters, args=(None, None), jac=None, hess=None):
        result = self.minimize_interface(
            fun,
            x0,
            method=self.method,
            args=args,
            jac=jac,
            hess=hess,
            options={"maxiter": iters},
        )
        self.step_index += iters
        objective = args[0]
        if self.logger is not None:
            print(result)
            self.logger.process_iter(iters, result.fun, objective)
        objective.parameters = result.x
        return result
