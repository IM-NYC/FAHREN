/*
 * FAHREN error codes and diagnostics.
 *
 * Public APIs return FAHREN_SUCCESS (0) or a negative-style positive code
 * from this header. Use fahren_strerror() for human-readable messages and
 * fahren_last_error_message() after failures that record context.
 */
#ifndef FAHREN_ERRORS_H
#define FAHREN_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Core status codes (stable ABI) */
#define FAHREN_SUCCESS                      0
#define FAHREN_ERROR_INVALID_ARGUMENT       1
#define FAHREN_ERROR_NOT_INITIALIZED        2
#define FAHREN_ERROR_PROCESSING_FAILED      3

/* Extended codes (room for growth) */
#define FAHREN_ERROR_OUT_OF_MEMORY          4
#define FAHREN_ERROR_IO                     5
#define FAHREN_ERROR_FORMAT                 6
#define FAHREN_ERROR_UNSUPPORTED            7
#define FAHREN_ERROR_NOT_FINALIZED          8
#define FAHREN_ERROR_ALREADY_FINALIZED      9
#define FAHREN_ERROR_LAYER_MISMATCH         10
#define FAHREN_ERROR_UNKNOWN                127

/** Return a static English description for `code`. Never NULL. */
const char* fahren_strerror(int code);

/**
 * Optional detail set by the library on some failures (e.g. I/O).
 * Empty string when none was recorded.
 */
const char* fahren_last_error_message(void);

/** Clear the last detail message. */
void fahren_clear_last_error(void);

/**
 * Record a detail message (internal use; exposed for add-ons and tests).
 * Truncates safely to an internal buffer.
 */
void fahren_set_last_error(const char* message);

#ifdef __cplusplus
}
#endif

#endif /* FAHREN_ERRORS_H */
