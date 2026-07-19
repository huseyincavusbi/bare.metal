#ifndef BMT_METAL_DEVICE_H
#define BMT_METAL_DEVICE_H

struct backend_ctx_s {
    void* device;
    void* queue;
    void* library;
};

struct backend_buffer_s {
    void* buffer;
    size_t size;
};

struct backend_kernel_s {
    void* pipeline_state;
};

struct backend_encoder_s {
    void* command_buffer;
    void* encoder;
};

#endif
