import os
import array
import logging
import argparse
import socket
import socketserver
import struct
import threading

import ray
from easydist.torch.tensorfield.mem_pool import IPCMemoryPool

logger = logging.getLogger(__name__)


class ParamGroups:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ParamGroups, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'param_groups'):
            self.param_groups = {}
            self._lock = threading.Lock()

    def set(self, param_group_id, alloc_return):
        with self._lock:
            if param_group_id in self.param_groups:
                raise ValueError(f"param_group_id({param_group_id}) already exists")
            self.param_groups[param_group_id] = alloc_return

    def get(self, param_group_id):
        with self._lock:
            if param_group_id not in self.param_groups:
                raise ValueError(f"param_group_id({param_group_id}) does not exist")
            return self.param_groups[param_group_id]

    def free(self, param_group_id):
        with self._lock:
            if param_group_id not in self.param_groups:
                raise ValueError(f"param_group_id({param_group_id}) does not exist")
            del self.param_groups[param_group_id]


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    pass


class TFeildRequestHandler(socketserver.BaseRequestHandler):

    def setup(self):
        # setup the connection and get device_id
        self.mem_pool = self.server.mem_pool
        self.param_groups = ParamGroups()

    def send_fd(self, sock, fd):
        """Send a file descriptor to the client."""
        fds = array.array("i", fd)
        sock.sendmsg([b'*'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])

    def handle(self):
        # setup the connection
        data = self.request.recv(256).strip()
        data = data.decode()
        assert data.startswith("pid") and len(data.split(" ")) == 2
        self.pid = int(data.split(" ")[1])
        self.request.sendall(b"PID received")
        logger.info(f"client connected {self.pid}")

        while True:
            # Receive data from the client
            data = self.request.recv(256).strip()
            if not data:
                logger.info(f"client disconnected {self.pid}")
                statistics = self.mem_pool.usage_statistics()
                logger.info(f"Memory pool statistics: {statistics}")
                break

            logger.debug(f"received: {data} from client {self.client_address}")

            if data.startswith(b"alloc"):
                data = data.decode()
                size = int(data.split(" ")[1])
                stream = int(data.split(" ")[2])
                ptr, handle, offset = self.mem_pool.malloc(size, stream)
                self.request.sendall(struct.pack('Q', ptr) + struct.pack('Q', offset) + handle)
            elif data.startswith(b"free"):
                data = data.decode()
                ptr = int(data.split(" ")[1])
                stream = int(data.split(" ")[2])
                self.mem_pool.free(ptr, stream)
                self.request.sendall(b"free done")
            elif data.startswith(b"param_group_alloc"):
                data = data.decode()
                size = int(data.split(" ")[1])
                param_group_id = data.split(" ")[2]
                stream = 0
                ptr, handle, offset = self.mem_pool.malloc(size, stream)
                self.param_groups.set(param_group_id, (size, ptr, handle, offset))
                self.request.sendall(struct.pack('Q', ptr) + struct.pack('Q', offset) + handle)
            elif data.startswith(b"param_group_get"):
                data = data.decode()
                param_group_id = data.split(" ")[1]
                size, ptr, handle, offset = self.param_groups.get(param_group_id)
                self.request.sendall(
                    struct.pack('Q', size) + struct.pack('Q', ptr) + struct.pack('Q', offset) +
                    handle)
            elif data.startswith(b"param_group_free"):
                data = data.decode()
                param_group_id = data.split(" ")[1]
                size, ptr, handle, offset = self.param_groups.get(param_group_id)
                self.mem_pool.free(ptr, 0)
                self.param_groups.free(param_group_id)
                self.request.sendall(b"free_param_group done")
            else:
                # Send a response
                self.request.sendall(b"Hello from server")

            statistics = self.mem_pool.usage_statistics()
            logger.debug(f"Memory pool statistics: {statistics}")


@ray.remote
class TFeildActor:

    def __init__(self, verbose=False):

        self.device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.socket_file = f"/tmp/tensorfield.{self.device_id}.sock"

        if verbose:
            logger.setLevel(logging.DEBUG)

        # Ensure the old socket file is removed
        if os.path.exists(self.socket_file):
            os.remove(self.socket_file)

        local_device_id = 0
        self.mem_pool = IPCMemoryPool(local_device_id)

    def start(self):
        # start on a separate thread
        self.thread = threading.Thread(target=self._start)
        self.thread.daemon = True
        self.thread.start()
        return True

    def _start(self):
        # Create and start the Unix domain socket server
        with ThreadingUnixServer(self.socket_file, TFeildRequestHandler) as server:
            server.mem_pool = self.mem_pool
            logger.info(f"Server listening on {self.socket_file} with device_id({self.device_id})")
            server.serve_forever()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--socket-file', type=str, default='/tmp/tensorfield.sock')
    argparser.add_argument('--device-id', type=int, default=0)
    argparser.add_argument('--verbose', action='store_true')
    args = argparser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        level=logging.INFO)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Define the path for the Unix socket file
    socket_file = args.socket_file

    # Ensure the old socket file is removed
    if os.path.exists(socket_file):
        os.remove(socket_file)

    mem_pool = IPCMemoryPool(args.device_id)

    # Create and start the Unix domain socket server
    with ThreadingUnixServer(socket_file, TFeildRequestHandler) as server:
        server.mem_pool = mem_pool
        logger.info(f"Server listening on {socket_file} with device_id({args.device_id})")
        server.serve_forever()


if __name__ == '__main__':
    main()
