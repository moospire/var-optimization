import grpc
import time
import os
import sys
from concurrent import futures

# add paths relative to the location of the python file
# there must be a better solution here?
file_dir = os.path.dirname(__file__)
# add below when breaking out into another file
if not file_dir in sys.path:
    sys.path.append(file_dir)
    print('adding {} to sys path'.format(file_dir))

import blender_shadow_pb2_grpc as blender_shadow_rpc
import blender_shadow_pb2 as blender_shadow_pb
from blender_ops import Scene

scene = Scene()

# typically would be defined in docker/remote
class BlenderShadowServicer(blender_shadow_rpc.BlenderShadowServicer):
    def GetVal(self, request, context):
        render_result = scene.render(request.target_x, request.target_y)
        res = blender_shadow_pb.Response(id=request.id)
        self.np_to_proto(render_result, res.render)
        return res

    @staticmethod
    def np_to_proto(np_arr, proto):
        proto.dtype = str(np_arr.dtype)
        proto.shape[:] = np_arr.shape
        proto.data = np_arr.tobytes()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    blender_shadow_rpc.add_BlenderShadowServicer_to_server(BlenderShadowServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Server started on port.')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
