#ifndef NOVA_TEST_HARNESS_H
#define NOVA_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>

static int g_nova_tests_run = 0;
static int g_nova_tests_failed = 0;

#define NOVA_ASSERT(cond, msg) do { \
    ++g_nova_tests_run; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", (msg)); \
        ++g_nova_tests_failed; \
    } else { \
        printf("ok  %s\n", (msg)); \
    } \
} while (0)

#endif
