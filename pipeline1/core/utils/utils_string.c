#include "utils_string.h"

uint32_t str_to_uint32_hash(const char *str)
{
  if (str == NULL) {
    return 0;
  }

  uint32_t hash = 2166136261u; // seed
  while (*str) {
    hash ^= (uint8_t)(*str);
    hash *= 16777619u;
    str++;
  }
  return hash;
}
