#include <string.h>
#include <nova/errors.h>

#define NOVA_ERRMSG_MAX 512

static char g_nova_last_error[NOVA_ERRMSG_MAX];

static const char* nova_error_table[] = {
    [NOVA_SUCCESS] = "success",
    [NOVA_ERROR_INVALID_ARGUMENT] = "invalid argument",
    [NOVA_ERROR_NOT_INITIALIZED] = "not initialized",
    [NOVA_ERROR_PROCESSING_FAILED] = "processing failed",
    [NOVA_ERROR_OUT_OF_MEMORY] = "out of memory",
    [NOVA_ERROR_IO] = "I/O error",
    [NOVA_ERROR_FORMAT] = "invalid or unsupported format",
    [NOVA_ERROR_UNSUPPORTED] = "unsupported operation",
    [NOVA_ERROR_NOT_FINALIZED] = "model not finalized",
    [NOVA_ERROR_ALREADY_FINALIZED] = "model already finalized",
    [NOVA_ERROR_LAYER_MISMATCH] = "layer configuration mismatch",
    [NOVA_ERROR_CORRUPTED_MODEL] = "corrupted model file",
    [NOVA_ERROR_CHECKSUM_MISMATCH] = "checksum mismatch",
    [NOVA_ERROR_INCOMPLETE_MODEL] = "incomplete model files",
    [NOVA_ERROR_BACKEND_UNAVAILABLE] = "backend not available",
    [NOVA_ERROR_QUANTIZATION] = "quantization error",
    [NOVA_ERROR_BUFFER_TOO_SMALL] = "buffer too small",
    [NOVA_ERROR_UNKNOWN] = "unknown error",
};

const char* nova_strerror(NOVA_Status status) {
    if (status >= 0 && status < (int)(sizeof(nova_error_table) / sizeof(nova_error_table[0]))) {
        const char* msg = nova_error_table[status];
        if (msg) return msg;
    }
    return nova_error_table[NOVA_ERROR_UNKNOWN];
}

const char* nova_get_last_error(void) {
    return g_nova_last_error;
}

void nova_clear_last_error(void) {
    g_nova_last_error[0] = '\0';
}

void nova_set_last_error(const char* message) {
    if (!message) {
        nova_clear_last_error();
        return;
    }
    strncpy(g_nova_last_error, message, NOVA_ERRMSG_MAX - 1);
    g_nova_last_error[NOVA_ERRMSG_MAX - 1] = '\0';
}
