#include <cuda_runtime_api.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <torch/extension.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

int create_socket() {
    int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd == -1) {
        throw std::runtime_error("Failed to create socket");
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    const char* tensorfield_socket_path = std::getenv("TENSORFIELD_SOCKET_PATH");
    if (tensorfield_socket_path != NULL) {
        strncpy(addr.sun_path, tensorfield_socket_path, sizeof(addr.sun_path) - 1);
    } else {
        strncpy(addr.sun_path, "/tmp/tensorfield.sock", sizeof(addr.sun_path) - 1);
    }

    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        throw std::runtime_error(
            "Connection to the socket failed. Please start the tfield server using "
            "`tfield-server` command first.");
    }

    return sockfd;
}

void close_socket(int sockfd) {
    if (sockfd != -1) {
        close(sockfd);
    }
}

int sockfd = -1;

void send_to_server(int sockfd, const std::string& message) {
    if (send(sockfd, message.c_str(), message.size(), 0) == -1) {
        throw std::runtime_error("Failed to send message");
    }
}

void recv_from_server(int sockfd) {
    char buffer[256];
    ssize_t bytes_received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received == -1) {
        std::cerr << "Failed to receive message" << std::endl;
        close_socket(sockfd);
        sockfd = -1;
        return;
    }
    buffer[bytes_received] = '\0';
}

extern "C" {

static std::unordered_map<uintptr_t, uintptr_t> allocate_mem;

void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
    static bool initialized = false;

    if (!initialized) {
        initialized = true;
        sockfd = create_socket();
        if (sockfd == -1) {
            throw std::runtime_error("Failed to create socket");
        }
        // send pid to server
        int pid = getpid();
        std::string message = "pid " + std::to_string(pid);
        send_to_server(sockfd, message);
        recv_from_server(sockfd);
    }

    if (size == 0) {
        return 0;
    }

    std::string message =
        "alloc " + std::to_string(size) + " " + std::to_string(reinterpret_cast<uintptr_t>(stream));
    send_to_server(sockfd, message);

    char buffer[256];
    ssize_t bytes_received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);

    if (bytes_received != 2 * sizeof(uintptr_t) + sizeof(cudaIpcMemHandle_t)) {
        throw std::runtime_error("alloc failed: invalid response size");
    }

    uintptr_t received_ptr = 0;
    memcpy(&received_ptr, buffer, sizeof(uintptr_t));

    uintptr_t offset = 0;
    memcpy(&offset, buffer + sizeof(uintptr_t), sizeof(uintptr_t));

    cudaIpcMemHandle_t ipcMemHandle;
    memcpy(&ipcMemHandle, buffer + 2 * sizeof(uintptr_t), sizeof(cudaIpcMemHandle_t));

    void* ptr;
    cudaError_t ret = cudaIpcOpenMemHandle(&ptr, ipcMemHandle, cudaIpcMemLazyEnablePeerAccess);
    if (ret != cudaSuccess) {
        throw std::runtime_error("Failed to import pointer");
    }
    ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + offset);

    if (allocate_mem.find(reinterpret_cast<uintptr_t>(ptr)) != allocate_mem.end()) {
        std::cout << "Allocated memory at " << std::to_string(reinterpret_cast<uintptr_t>(ptr))
                  << " " << std::to_string(received_ptr) << std::endl;
        throw std::runtime_error("Pointer already exists");
    }

    allocate_mem[reinterpret_cast<uintptr_t>(ptr)] = received_ptr;

    return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    uintptr_t received_ptr = allocate_mem[reinterpret_cast<uintptr_t>(ptr)];

    std::string message = "free " + std::to_string(received_ptr) + " " +
                          std::to_string(reinterpret_cast<uintptr_t>(stream));
    allocate_mem.erase(reinterpret_cast<uintptr_t>(ptr));

    send_to_server(sockfd, message);
    recv_from_server(sockfd);
}
}

PYBIND11_MODULE(tensorfield, m) {}
