import grpc
import time
from concurrent import futures
import sphere_pb2_grpc as sphere_rpc
import sphere_pb2 as sphere_pb

# typically would be defined in docker/remote
class SphereServicer(sphere_rpc.SphereServicer):
    def GetVal(self, request, context):
        return sphere_pb.Response(id=request.id, x=request.x)

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
