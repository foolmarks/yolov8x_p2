#ifndef SIMAMM
#define SIMAMM

#include <gst/gst.h>
#include <simaai/simaai_memory.h>

#ifdef __cplusplus
extern "C"
{
#endif
// Exported APIs
simaai_memory_t *allocate_memory(unsigned int size, int target);
void deallocate_memory(simaai_memory_t *memory);
simaai_memory_t *attach_to_memory(unsigned int buffer_id);
uint64_t buffer_id_to_paddr(unsigned int id);
void* buffer_id_to_vaddr(unsigned int id);
#ifdef __cplusplus
}
#endif /* extern "C" { */
#endif //SIMAMM
