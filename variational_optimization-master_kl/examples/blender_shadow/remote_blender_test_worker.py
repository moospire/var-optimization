import redis
import sys
import json
import numpy as np
import os, sys

# if in the future we don't want to have to load a provided blender file
# bpy.ops.wm.read_homefile(use_empty=True)

# this is hardcoded for testing... (only if you request to render test target)
GT_TEST_XY = (1, 3)

argv = sys.argv
try:
    index = argv.index("--") + 1
except ValueError:
    index = len(argv)

argv = argv[index:]

print(argv)

# startup.blend --background --python {}/generate-random-scene.py --
# '/home/tswedish/nlos/blender-git/build_linux/bin/blender'
# add paths relative to the location of the python file
# there must be a better solution here?
file_dir = os.path.dirname(__file__)
# add below when breaking out into another file
if not file_dir in sys.path:
    sys.path.append(file_dir)
from blender_ops import Scene
blender_scene = Scene()
    
variational_optim_module_path = os.path.join('/',*file_dir.split('/')[:-4])
if not variational_optim_module_path in sys.path:
    sys.path.append(variational_optim_module_path)

from variational_optimization.rpc.pubsub import RedisInterface

remote_interface = RedisInterface()

worker_channel = argv[0]
worker_uid = argv[1]
worker_timeout = float(argv[2])
render_test_target = int(argv[3])

print('render test target is : {}'.format(render_test_target))

print(worker_uid)
print('worker created on channel {}'.format(worker_channel))

res_channel = '{}-response'.format(worker_channel)

target = None

if render_test_target == 1:
    print('Rendering test target')
    target = blender_scene.render(*GT_TEST_XY, quality=1)
    print('done.')

def cost(sample, return_vis=False):
    x, y = sample
    result = blender_scene.render(x, y, quality=0.)
    assert target is not None, 'target not defined!'
    cost_val = np.mean(np.abs(result - target))
    if return_vis:
        vis_arr = np.zeros((2, target.shape[0], target.shape[1], target.shape[2]))
        vis_arr[0] = target
        vis_arr[1] = result
        return cost_val, vis_arr
    return cost_val

if __name__== "__main__":
    remote_interface.subscribe(worker_channel)
    remote_interface.res_worker_ready(res_channel, worker_uid)
    while True:
        # res: len(2) -> update config, res[0] == target, res[1] == params_list
        #      len(3) -> compute cost, res[0] == params, res[1] == uid, res[2] == return activations
        # TODO: return objects, not tuples
        res = remote_interface.get_worker_job(timeout=worker_timeout)
        if res is not None:
            if len(res) == 2:
                target = np.array(res[0])
            elif target is not None:
                params = res[0]
                uid = res[1]
                print(res)
                if res[2]:
                    cost_val, vis = cost(params, return_vis=True)
                    remote_interface.res_worker_cost_and_vis(res_channel,  
                                                             cost_val, 
                                                             uid,
                                                             vis)
                else:
                    remote_interface.res_worker_cost(res_channel, 
                                                     cost(params), 
                                                     uid)
        
            
