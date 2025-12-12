#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function pointer type for the original connect function
typedef int (*connect_func_t)(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    // Only intercept IPv4 connections
    if (addr->sa_family == AF_INET) {
        const struct sockaddr_in *sin = (const struct sockaddr_in *)addr;
        
        // Check if connecting to 127.0.0.1:1080 (Our SOCKS Target)
        // 1080 in Network Byte Order (htons not stricly needed if we know constant, but safer)
        if (sin->sin_port == htons(1080) &&
            sin->sin_addr.s_addr == inet_addr("127.0.0.1")) {
            
            // Redirect to Unix Domain Socket
            struct sockaddr_un sun;
            memset(&sun, 0, sizeof(sun));
            sun.sun_family = AF_UNIX;
            strncpy(sun.sun_path, "/tmp/socks.sock", sizeof(sun.sun_path) - 1);
            
            // Get the NEXT connect function (libc or next in chain)
            connect_func_t orig_connect = (connect_func_t)dlsym(RTLD_NEXT, "connect");
            if (!orig_connect) {
                fprintf(stderr, "[Shim] Error: connect symbol not found\n");
                return -1;
            }
            
            return orig_connect(sockfd, (struct sockaddr *)&sun, sizeof(sun));
        }
    }
    
    // Pass through all other connections
    connect_func_t orig_connect = (connect_func_t)dlsym(RTLD_NEXT, "connect");
    if (!orig_connect) return -1;
    return orig_connect(sockfd, addr, addrlen);
}
