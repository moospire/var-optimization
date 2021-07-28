import bpy
import bmesh

import numpy as np

from PIL import Image

class Scene:
    def __init__(self, params, seed=0):
        self.params = params
        self.seed = seed
        self.scene = bpy.context.scene
        self.scene.render.filepath = '/dev/shm/render.png'
        self.filepath = '/dev/shm/'
        tree = self.scene.node_tree
        tree.nodes["File Output"].base_path = self.filepath
        self.obj_dict = {}
        self.mat_dict = {}
        self.init = True
        self.setup()
        
    # pass in a param value, and recurse while checking scene object structure
    @staticmethod
    def walk_obj(obj, params):
        for attrkey in params.keys():
            if type(params[attrkey]) == dict:
                if hasattr(obj, attrkey):
                    walk_obj(getattr(obj, attrkey), params[attrkey])
            else:
                if hasattr(obj, attrkey):
                    setattr(obj, attrkey, params[attrkey])
        return True        
    
    def update_for_render(self, params):
        run_setup = False
        for obj_key in params.keys():
            if obj_key in ['front_corner','back_corner', 'height']:
                # check if anything in params that needs to get re-generated changed
                if not (self.params[obj_key] == params[obj_key]):
                    run_setup = True
                    self.params[obj_key] = params[obj_key]
            else:
                if obj_key in self.obj_dict:
                    self.walk_obj(self.obj_dict[obj_key], params[obj_key])
        if run_setup:
            self.setup()
        
        
    def render(self, params, quality=0, return_result=True, return_depth=False):
        # use quality [0-1] to represent how much noise is in the final render... 1 is low noise.
        self.update_for_render(params)
        self.scene.render.resolution_x = self.params['render_size'][0]
        self.scene.render.resolution_y = self.params['render_size'][1]
        self.scene.render.resolution_percentage = 100
        bpy.data.scenes["Scene"].cycles.aa_samples = int(quality*15+1)
        bpy.data.scenes["Scene"].cycles.seed = self.seed
        bpy.ops.render.render(layer='RenderLayer', write_still=False)
        
        self.seed += 1
        if return_result:
            render_result = (np.array(Image.open('{}Image0001.png'.format(self.filepath)))[:,:,:3])/255.
            if return_depth:
                depth = (np.array(Image.open('{}Depth0001.png'.format(self.filepath)))[:,:,:3])/255.
                return render_result, depth
            return render_result
        
        else:
            return True
    
    def setup(self, target_name='Target', use_gpu_num=None):
        ## SCENE      
        if self.init:
            if use_gpu_num is not None:
                scene.cycles.device = "GPU"
                cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences

                C = bpy.context
                C.scene.render.use_overwrite = True
                C.scene.render.use_placeholder = False
                cycles_prefs.compute_device_type = "CUDA"
                
            self.mat_dict['floor_material'] = self.create_new_bsdf_material(roughness=0.8, 
                                                                            color=(0.8, 0.8, 0.8, 1))
            self.mat_dict['wall_material'] = self.create_new_bsdf_material(roughness=0.05, 
                                                                           color=(0.8, 0.8, 0.8, 1))
            self.mat_dict['target_material'] = self.create_new_emission_material(brightness=100.)
            
            target_loc = self.params['target']['location']
            self.obj_dict['target'] = self.create_cube_obj(target_name, location=target_loc, scale=(0.3, 0.3, 1.))
            self.obj_dict['target'].data.materials.append(self.mat_dict['target_material'])
            
            self.obj_dict['camera'] = bpy.data.objects['Camera']
            self.obj_dict['camera'].rotation_mode = 'AXIS_ANGLE'
            
            self.init = False
        else:
            for obj_key in ['floor', 'right_wall', 'hidden_wall', 'back_wall',
                        'left_wall']:
                if obj_key in self.obj_dict.keys():
                    bpy.data.objects.remove(self.obj_dict[obj_key], do_unlink=True)
        
        floorplan_verts = self.make_floorplan_verts(self.params['front_corner'], self.params['back_corner'])
        right_wall_verts = floorplan_verts[0:2]
        hidden_wall_verts = floorplan_verts[1:3]
        back_wall_verts = floorplan_verts[3:5]
        left_wall_verts = floorplan_verts[4:6]
        
        self.obj_dict['floor'] = self.make_floor('floor', floorplan_verts)
        #ceiling = make_floor('ceiling', floorplan_verts, scene)
        #ceiling.location.z = params['height']
        self.obj_dict['right_wall'] = self.make_wall('right_wall', right_wall_verts, self.params['height'])
        self.obj_dict['hidden_wall'] = self.make_wall('hidden_wall', hidden_wall_verts, self.params['height'])
        self.obj_dict['back_wall'] = self.make_wall('back_wall', back_wall_verts, self.params['height'])
        self.obj_dict['left_wall'] = self.make_wall('left_wall', left_wall_verts, self.params['height'])
        
        self.obj_dict['floor'].data.materials.append(self.mat_dict['floor_material'])
        
        
        # for now
        self.obj_dict['right_wall'].data.materials.append(self.mat_dict['wall_material'])
        self.obj_dict['hidden_wall'].data.materials.append(self.mat_dict['wall_material'])
        self.obj_dict['back_wall'].data.materials.append(self.mat_dict['wall_material'])
        self.obj_dict['left_wall'].data.materials.append(self.mat_dict['wall_material'])
        
        

    ## MATERIALS
    @staticmethod
    def create_new_emission_material(brightness=180.):
        target_material = bpy.data.materials.new(name="target_material")
        target_material.use_nodes = True
        target_material_tree = target_material.node_tree
        target_material_tree.nodes.new(type='ShaderNodeEmission')
        target_material_tree.nodes['Emission'].inputs['Strength'].default_value = brightness
        inp = target_material_tree.nodes['Material Output'].inputs['Surface']
        outp = target_material_tree.nodes['Emission'].outputs['Emission']

        target_material_tree.links.new(inp, outp)
        return target_material

    @staticmethod
    def create_new_bsdf_material(roughness=0.0, color=(0.5, 0.5, 0.5, 1), use_diffuse=False):
        target_material = bpy.data.materials.new(name="bsdf_material")
        target_material.use_nodes = True
        target_material_tree = target_material.node_tree
        if use_diffuse:
            target_material_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
            target_material_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = color
            inp = target_material_tree.nodes['Material Output'].inputs['Surface']
            outp = target_material_tree.nodes['Diffuse BSDF'].outputs['BSDF']
        else:
            target_material_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
            target_material_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
            inp = target_material_tree.nodes['Material Output'].inputs['Surface']
            outp = target_material_tree.nodes['Principled BSDF'].outputs['BSDF']
            target_material_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = roughness
        target_material_tree.links.new(inp, outp)
        return target_material

    ## GEOMETRY
    @staticmethod
    def set_smooth(obj):
        for face in obj.data.polygons:
            face.use_smooth = True

    def object_from_data(self, data, name):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(data['verts'], data['edges'], data['faces'])

        obj = bpy.data.objects.new(name, mesh)
        self.scene.objects.link(obj)

        # scene.objects.active = obj
        # obj.select = True
        mesh.validate(verbose=True)
        return obj
    
    def create_cube_obj(self, name, location=(0,0,0), rotation=(0,0,0), scale=(1,1,1)):
        mesh = bpy.data.meshes.new('{}_mesh'.format(name))
        obj = bpy.data.objects.new('{}_obj'.format(name), mesh)
        self.scene.objects.link(obj)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
        obj.location.x = location[0]
        obj.location.y = location[1]
        obj.location.z = location[2]
        obj.rotation_euler.x = rotation[0]
        obj.rotation_euler.y = rotation[1]
        obj.rotation_euler.x = rotation[2]
        obj.scale.x = scale[0]
        obj.scale.y = scale[1]
        obj.scale.z = scale[2]
        return obj
    
    @staticmethod
    def make_floorplan_verts(front_corner, back_corner):
        verts = []
        frontx, fronty = front_corner
        backx, backy = back_corner

        verts.append((frontx, 0., 0.))
        verts.append((frontx, fronty, 0.))
        verts.append((20., fronty, 0.))
        verts.append((20., backy, 0.))
        verts.append((backx, backy, 0.))
        verts.append((backx, 0., 0.))
        return verts
    
    def make_floor(self, name, floorplan_verts):
        floor_face = list(range(len(floorplan_verts)))

        data = {'verts': [], 'edges': [], 'faces': []}
        data['verts'].extend(floorplan_verts)
        data['faces'].append(floor_face)
        obj = self.object_from_data(data, name)
        # set_smooth(obj)
        return obj
    
    def make_wall(self, name, base_verts, height):
        top_verts = []
        for v in base_verts:
            top_verts.append((v[0],v[1],v[2]+height))
        base_verts.extend(top_verts)
        top_verts_ind = [i+len(top_verts) for i in range(len(top_verts))]
        base_verts_ind = list(range(len(top_verts)))
        base_verts_ind.reverse()
        face = []
        for ind in top_verts_ind:
            face.append(ind)
        for ind in base_verts_ind:
            face.append(ind)
        data = {'verts': [], 'edges': [], 'faces': []}
        data['verts'].extend(base_verts)
        data['faces'].append(face)
        obj = self.object_from_data(data, name)
        return obj
    
if __name__== "__main__":
    params = {'front_corner': (2., 5.),
          'back_corner': (-2, 10.),
          'height': 3.,
          'target': {'location': (10, 7, 1.5)}}
    blender_scene = Scene(params)
    pi = 3.1415926
    render_params = {'target': {'location': (10, 8, 1.5)}, 
                     'camera': {'location': (2, 0, 1.5), 
                                'rotation_axis_angle': (pi/2.+0.1, 1.,0.,0)}
                    }
    blender_scene.filepath = 'test_blender.png'
    blender_scene.render(render_params, quality=1., return_result=False)
