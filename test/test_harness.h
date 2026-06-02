#ifndef FAHREN_TEST_HARNESS_H
#define FAHREN_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>

static int g_fahren_tests_run = 0;
static int g_fahren_tests_failed = 0;

#define FAHREN_ASSERT(cond, msg) do { \
    ++g_fahren_tests_run; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", (msg)); \
        ++g_fahren_tests_failed; \
    } else { \
        printf("ok  %s\n", (msg)); \
    } \
} while (0)

#define FAHREN_TEST_MAIN() \
    int main(void) { \
        printf("\n%d passed, %d failed\n\n", \
               g_fahren_tests_run - g_fahren_tests_failed, g_fahren_tests_failed); \
        return g_fahren_tests_failed > 0 ? 1 : 0; \
    }

#endif
