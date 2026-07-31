// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "log.h"
#include <stdarg.h>
#include <stdlib.h>

static bmt_log_level_t g_log_level = BMT_LOG_INFO;

void bmt_log_set_level(bmt_log_level_t level) {
    g_log_level = level;
}

void bmt_log(bmt_log_level_t level, const char* file, int line,
             const char* func, const char* fmt, ...) {
    if (level < g_log_level) return;

    const char* prefix;
    switch (level) {
        case BMT_LOG_DEBUG: prefix = "DEBUG"; break;
        case BMT_LOG_INFO:  prefix = "INFO";  break;
        case BMT_LOG_WARN:  prefix = "WARN";  break;
        case BMT_LOG_ERROR: prefix = "ERROR"; break;
        default:            prefix = "?";     break;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    fprintf(stderr, "[%s] %ld.%09ld %s:%d %s(): ",
            prefix, ts.tv_sec, ts.tv_nsec,
            file, line, func);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}
