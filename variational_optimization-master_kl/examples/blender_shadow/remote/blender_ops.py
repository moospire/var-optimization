import bpy
import bmesh

import numpy as np

from PIL import Image

seed = 0

bpy.context.scene.render.filepath = '/dev/shm/render.png'

# setup devices (default to CPU/small kernel size)
class Scene:
    def __init__(self, seed=0):
        self.seed = seed
        
    def render(self, x, y, quality=0, target_brightness=1.):
        # use quality [0-1] to represent how much noise is in the final render... 1 is low noise.
        bpy.data.objects["Target"].location[0] = x
        bpy.data.objects["Target"].location[1] = y
        bpy.data.scenes["Scene"].cycles.aa_samples = int(quality*15+1)
        bpy.data.scenes["Scene"].cycles.seed = self.seed
        self.seed += 1
        #bpy.data.objects["Target"].location[2] = 6.5
        bpy.ops.render.render(layer='RenderLayer', write_still=True)
        render_result = (np.array(Image.open(bpy.context.scene.render.filepath))[:,:,:3])/255.
        print('Target location: ')
        print(bpy.data.objects["Target"].location)
        return render_result