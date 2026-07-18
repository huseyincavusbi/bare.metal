#include "baremetal/checkpoint.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int bmt_checkpoint_load(bm_model_t* model, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        BMT_LOG_ERROR("Cannot open checkpoint: %s", path);
        return -1;
    }

    int32_t header[256];
    if (fread(header, sizeof(int32_t), 256, file) != 256) {
        BMT_LOG_ERROR("Failed to read header");
        fclose(file);
        return -1;
    }

    if (header[0] != BMT_CHECKPOINT_MAGIC) {
        BMT_LOG_ERROR("Bad magic: %d (expected %d)", header[0], BMT_CHECKPOINT_MAGIC);
        fclose(file);
        return -1;
    }

    bm_arch_t arch;
    memset(&arch, 0, sizeof(arch));
    arch.precision     = (bm_precision_t)header[2];
    arch.dim           = header[3];
    arch.hidden_dim    = header[4];
    arch.n_layers      = header[5];
    arch.n_heads       = header[6];
    arch.n_kv_heads    = header[7];
    arch.vocab_size    = header[8];
    arch.max_seq_len   = header[9];
    arch.norm          = (bm_norm_t)header[10];
    arch.activation    = (bm_act_t)header[11];
    arch.pos_enc       = (bm_pos_t)header[12];
    arch.attention     = (bm_attn_t)header[13];
    arch.bias          = header[14];
    arch.weight_tie    = header[15];
    arch.padded_vocab_size = arch.vocab_size;
    arch.rope_theta    = 10000.0f;

    if (bmt_model_alloc_buffers(model, &arch) != 0) {
        fclose(file);
        return -1;
    }

    float* w = (float*)model->weight_buffer;
    size_t total_floats = model->n_parameters;
    size_t read_count = fread(w, sizeof(float), total_floats, file);
    fclose(file);

    if (read_count != total_floats) {
        BMT_LOG_ERROR("Read %zu floats, expected %d", read_count, model->n_parameters);
        bmt_model_free_buffers(model);
        return -1;
    }

    BMT_LOG_INFO("Loaded checkpoint: %s (%d params)", path, model->n_parameters);
    return 0;
}

int bmt_checkpoint_load_legacy_llama2c(bm_model_t* model, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        BMT_LOG_ERROR("Cannot open legacy checkpoint: %s", path);
        return -1;
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (data == MAP_FAILED) {
        BMT_LOG_ERROR("mmap failed for %s", path);
        return -1;
    }

    int32_t* config = (int32_t*)data;
    bm_arch_t arch;
    memset(&arch, 0, sizeof(arch));
    arch.dim         = config[0];
    arch.hidden_dim  = config[1];
    arch.n_layers    = config[2];
    arch.n_heads     = config[3];
    arch.n_kv_heads  = config[4];
    arch.vocab_size  = abs(config[5]);
    arch.weight_tie  = config[5] > 0;
    arch.max_seq_len = config[6];
    arch.padded_vocab_size = arch.vocab_size;
    arch.norm        = BM_NORM_RMSNORM;
    arch.activation  = BM_ACT_SWIGLU;
    arch.pos_enc     = BM_POS_ROPE;
    arch.attention   = (arch.n_kv_heads < arch.n_heads) ? BM_ATTN_GQA : BM_ATTN_MHA;
    arch.bias        = 0;
    arch.rope_theta  = 10000.0f;
    arch.precision   = BM_PRECISION_FP32;

    if (bmt_model_alloc_buffers(model, &arch) != 0) {
        munmap(data, file_size);
        return -1;
    }

    float* dst = (float*)model->weight_buffer;
    float* src = (float*)(config + 7);
    memcpy(dst, src, model->n_parameters * sizeof(float));

    munmap(data, file_size);
    BMT_LOG_INFO("Loaded legacy llama2.c checkpoint: %s (%d params)", path, model->n_parameters);
    return 0;
}

int bmt_checkpoint_save(bm_model_t* model, const char* path) {
    (void)model; (void)path;
    BMT_LOG_WARN("bmt_checkpoint_save: not yet implemented");
    return -1;
}
