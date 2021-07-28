import time
from scipy import optimize

from variational_optimization.fit import DefaultRemoteConfig, Fit
from variational_optimization.logger import SaveResultsLogger
from variational_optimization.ops import ops
from variational_optimization.cost_definitions import docker_blender_iccp
from variational_optimization.core import SimpleGradientDescentUpdater, CovarianceGradientDescentUpdater
from variational_optimization.core import ExternalUpdater
from variational_optimization.parameters import LCornerConfig, LCornerParameters
from variational_optimization.rpc.messages import LCornerWorkerRequest 


lcornerconfig = LCornerConfig()
lcornerconfig.front_corner.x.trainable = False
lcornerconfig.front_corner.y.trainable = False
lcornerconfig.back_corner.x.trainable = True
lcornerconfig.back_corner.y.trainable = True
lcornerconfig.height.trainable = False
lcornerconfig.target_location.x.trainable = True
lcornerconfig.target_location.y.trainable = True

lcornerconfig.front_corner.x.init_variance = .1
lcornerconfig.front_corner.y.init_variance = .1
lcornerconfig.back_corner.x.init_variance = .1
lcornerconfig.back_corner.y.init_variance = .1
lcornerconfig.height.init_variance = .1
lcornerconfig.target_location.x.init_variance = 3.
lcornerconfig.target_location.y.init_variance = 3.

lcornerconfig.target_location.x.value = 8.
lcornerconfig.target_location.y.value = 8.

lcornerconfig.target = ops.load_blender_im('tests/iccp_6-9.png')

class BlenderICCPRemoteConfig(DefaultRemoteConfig):
    request_cls = LCornerWorkerRequest
    parameter_cls = LCornerParameters
    logger = SaveResultsLogger('results', save_key=int(time.time()), save_vis=True)
    updater = CovarianceGradientDescentUpdater(25.)
    #sampler_init = [8., 8., 0., 1.6, 1.6]
    # L-BFGS-B
    #updater = ExternalUpdater(optimize.minimize, method='BFGS')
    #sampler_init = [0., 0., 0., 1.1, 1.1]
    worker_f = docker_blender_iccp
    # beta, alpha, x, y, --- a, b, c, d, e, f --- covbeta, covalpha, covx, covy
    #sampler_init = [0., .2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.2, 1.2]
    # fixed
    param_config = lcornerconfig
    num_workers = 20
    num_samples = 20
    num_print_steps = 30
            
def fit(config, iters=100):
    config.logger.set_log_id('iccp_example')
    blender_fit = Fit(config)
    result, distribution = blender_fit.fit(iters)
    return result, distribution

if __name__== "__main__":
    config = BlenderICCPRemoteConfig
    result, distribution = fit(config, iters=100)
    print(result)
    print(distribution)
    