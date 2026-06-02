#include <string.h>

#include <fahren/errors.h>

#define FAHREN_ERRMSG_MAX 512

static char g_fahren_last_error[FAHREN_ERRMSG_MAX];

static const char* fahren_error_table[] = {
    [FAHREN_SUCCESS] = "success",
    [FAHREN_ERROR_INVALID_ARGUMENT] = "invalid argument",
    [FAHREN_ERROR_NOT_INITIALIZED] = "library or model not initialized",
    [FAHREN_ERROR_PROCESSING_FAILED] = "processing failed",
    [FAHREN_ERROR_OUT_OF_MEMORY] = "out of memory",
    [FAHREN_ERROR_IO] = "I/O error",
    [FAHREN_ERROR_FORMAT] = "invalid or unsupported file format",
    [FAHREN_ERROR_UNSUPPORTED] = "unsupported operation in this build",
    [FAHREN_ERROR_NOT_FINALIZED] = "model not finalized to a weights file",
    [FAHREN_ERROR_ALREADY_FINALIZED] = "model already finalized",
    [FAHREN_ERROR_LAYER_MISMATCH] = "layer configuration does not match weights file",
    [FAHREN_ERROR_UNKNOWN] = "unknown error",
};

const char* fahren_strerror(int code) {
    if (code >= 0 && code < (int)(sizeof(fahren_error_table) / sizeof(fahren_error_table[0]))) {
        const char* msg = fahren_error_table[code];
        if (msg) {
            return msg;
        }
    }
    return fahren_error_table[FAHREN_ERROR_UNKNOWN];
}

const char* fahren_last_error_message(void) {
    return g_fahren_last_error;
}

void fahren_clear_last_error(void) {
    g_fahren_last_error[0] = '\0';
}

void fahren_set_last_error(const char* message) {
    if (!message) {
        fahren_clear_last_error();
        return;
    }
    strncpy(g_fahren_last_error, message, FAHREN_ERRMSG_MAX - 1);
    g_fahren_last_error[FAHREN_ERRMSG_MAX - 1] = '\0';
}
