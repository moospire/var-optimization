import numpy as np

from variational_optimization.core import *
from variational_optimization.ops import ops
from variational_optimization.sampler import NormalSampler

sampler_init = [[5., 5.],[4.47, 4.09]]

sampling_distribution = NormalSampler(sampler_init[0],sampler_init[1], scale_grad=False)

Sigma = sampling_distribution.Sigma
mu = sampling_distribution.mu
print('First Sigma: \n{}\n'.format(Sigma))
print('First params: \n{}\n'.format(sampling_distribution.parameters))

J = sampling_distribution.d_S_dgamma()
print('First Jacobian: \n{}\n'.format(J))

d_E_dS = np.array(sampling_distribution.d_E_dS([1., 0]))
print('dE_dS: \n{}\n'.format(d_E_dS))

grad = J @ d_E_dS.flatten()
print('grad sigma params: \n{}\n'.format(grad))

updated_params = sampling_distribution.parameters[sampling_distribution.dim:] - 0.1 * grad
print('updated parameters: \n{}\n'.format(updated_params))

# still need to make sure parameters are updated correctly...
new_Sigma = sampling_distribution.Sigma_from_manifold_params(sampling_distribution.dim, updated_params)
print('new Sigma: \n{}\n'.format(new_Sigma))

new_params = sampling_distribution.parameters_from_mu_Sigma(mu, new_Sigma)
print('new params: \n{}\n'.format(new_params))

sampling_distribution.parameters = new_params
print('new Sigma calc: \n{}\n'.format(sampling_distribution.Sigma))
print('new mu calc (note we actually have not updated here: \n{}\n'.format(sampling_distribution.mu))

# reset to original state and check
sampling_distribution = NormalSampler(sampler_init[0],sampler_init[1], scale_grad=False)
print('reset params: \n{}\n'.format(sampling_distribution.parameters))
print('reset Sigma calc: \n{}\n'.format(sampling_distribution.Sigma))
gradient = sampling_distribution.gradient([1., 0])
print('reset grad: \n{}\n'.format(gradient))

sampling_distribution_delta = -0.1 * gradient
print('samp dist delta: \n{}\n'.format(sampling_distribution_delta))
sampling_distribution.parameters = np.array(sampling_distribution.parameters) + np.array(sampling_distribution_delta.parameters)

print('reset new params: \n{}\n'.format(sampling_distribution.parameters))
print('reset new Sigma calc: \n{}\n'.format(sampling_distribution.Sigma))
print('reset new mu calc: \n{}\n'.format(sampling_distribution.mu))
