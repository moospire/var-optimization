import grpc
import time
import os
import sys
from concurrent import futures

import blender_shadow_pb2_grpc as blender_shadow_rpc
import blender_shadow_pb2 as blender_shadow_pb

grpc_opts = [("grpc.lb_policy_name", "round_robin",)]
channel = grpc.insecure_channel('shadow_blendershadow:50051', grpc_opts)
stub = blender_shadow_rpc.BlenderShadowStub(channel)

# relay requests to multiple nodes using round robin on the overlay network
class BlenderShadowServicer(blender_shadow_rpc.BlenderShadowServicer):
    def GetVal(self, request, context):
        result = stub.GetVal(request)
        return result

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
