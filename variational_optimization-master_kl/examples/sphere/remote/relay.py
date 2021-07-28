import grpc
import time
import os
import sys
from concurrent import futures

import sphere_pb2_grpc as sphere_rpc
import sphere_pb2 as sphere_pb

grpc_opts = [("grpc.lb_policy_name", "round_robin",)]
channel = grpc.insecure_channel('sphere_sphere:50051', grpc_opts)
stub = sphere_rpc.SphereStub(channel)

# relay requests to multiple nodes using round robin on the overlay network
class SphereServicer(sphere_rpc.SphereServicer):
    def GetVal(self, request, context):
        result = stub.GetVal(request)
        return result

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sphere_rpc.add_SphereServicer_to_server(SphereServicer(), server)
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
