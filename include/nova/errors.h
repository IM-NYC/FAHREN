#ifndef NOVA_ERRORS_H
#define NOVA_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

#define NOVA_SUCCESS                      0
#define NOVA_ERROR_INVALID_ARGUMENT       1
#define NOVA_ERROR_NOT_INITIALIZED        2
#define NOVA_ERROR_PROCESSING_FAILED      3
#define NOVA_ERROR_OUT_OF_MEMORY          4
#define NOVA_ERROR_IO                     5
#define NOVA_ERROR_FORMAT                 6
#define NOVA_ERROR_UNSUPPORTED            7
#define NOVA_ERROR_NOT_FINALIZED          8
#define NOVA_ERROR_ALREADY_FINALIZED      9
#define NOVA_ERROR_LAYER_MISMATCH         10
#define NOVA_ERROR_CORRUPTED_MODEL        11
#define NOVA_ERROR_CHECKSUM_MISMATCH      12
#define NOVA_ERROR_INCOMPLETE_MODEL       13
#define NOVA_ERROR_BACKEND_UNAVAILABLE    14
#define NOVA_ERROR_QUANTIZATION           15
#define NOVA_ERROR_BUFFER_TOO_SMALL       16
#define NOVA_ERROR_UNKNOWN                127

typedef int NOVA_Status;

const char* nova_strerror(NOVA_Status status);
const char* nova_last_error_message(void);
void nova_clear_last_error(void);
void nova_set_last_error(const char* message);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_ERRORS_H */
