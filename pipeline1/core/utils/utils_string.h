#ifndef _UTILS_STRING
#define _UTILS_STRING

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Make uint32_t hash identificator using C-string
/// @param str string to be converted to a numeric value
/// @return unsigned 32-bit integer hash
uint32_t str_to_uint32_hash(const char *str);

#ifdef __cplusplus
}
#endif

#endif // _UTILS_STRING
