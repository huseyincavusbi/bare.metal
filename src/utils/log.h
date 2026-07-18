#ifndef BMT_LOG_H
#define BMT_LOG_H

#include <stdio.h>
#include <time.h>

typedef enum {
    BMT_LOG_DEBUG,
    BMT_LOG_INFO,
    BMT_LOG_WARN,
    BMT_LOG_ERROR,
} bmt_log_level_t;

void bmt_log_set_level(bmt_log_level_t level);
void bmt_log(bmt_log_level_t level, const char* file, int line,
             const char* func, const char* fmt, ...);

#define BMT_LOG_DEBUG(fmt, ...) \
    bmt_log(BMT_LOG_DEBUG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define BMT_LOG_INFO(fmt, ...) \
    bmt_log(BMT_LOG_INFO, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define BMT_LOG_WARN(fmt, ...) \
    bmt_log(BMT_LOG_WARN, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define BMT_LOG_ERROR(fmt, ...) \
    bmt_log(BMT_LOG_ERROR, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#ifdef BMT_DEBUG
#define BMT_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        BMT_LOG_ERROR("ASSERTION FAILED: %s (%s)", msg, #cond); \
        abort(); \
    } \
} while(0)
#else
#define BMT_ASSERT(cond, msg) ((void)0)
#endif

#endif
