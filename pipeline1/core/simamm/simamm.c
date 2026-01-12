#include <pthread.h>

#include "simamm.h"

typedef struct {
    unsigned int id;
    uint64_t paddr;
    void* vaddr;
} buffer_mapping;

static buffer_mapping mappings[sizeof(uint64_t) * 8] = {{ .id = -1, .paddr = 0, .vaddr = NULL }};
static int64_t current_mappings = 0;
static pthread_mutex_t mapping_mutex = PTHREAD_MUTEX_INITIALIZER;

simaai_memory_t *allocate_memory(unsigned int size, int target) {

	simaai_memory_t *memory = NULL;
	void *vaddr = NULL;

	memory = simaai_memory_alloc(size, target);
	if(!memory) {
		GST_ERROR("ERROR: allocating contiguous memory");
		return NULL;
	}

	vaddr = simaai_memory_map(memory);
	if(!vaddr) {
		GST_ERROR("ERROR: mapping allocated contiguouse memory %#x", simaai_memory_get_phys(memory));
		simaai_memory_free(memory);
		return NULL;
	}

	return memory;
}

void deallocate_memory(simaai_memory_t *memory) {

	if(!memory) {
		return;
	}
	
	simaai_memory_unmap(memory);
	simaai_memory_free(memory);

}

simaai_memory_t *attach_to_memory(unsigned int buffer_id) {

	simaai_memory_t *memory = NULL;
	void *vaddr = NULL;

	memory = simaai_memory_attach(buffer_id);
	if(!memory) {
		GST_ERROR("ERROR: attaching to memory of buffer id %d", buffer_id);
		return NULL;
	}

	vaddr = simaai_memory_map(memory);
	if(!vaddr) {
		GST_ERROR("ERROR: mapping memory of buffer id %d", buffer_id);
		return NULL;
	}	

	return memory;
}

static int32_t find_buffer_by_id(unsigned int id)
{
    int32_t i;
    simaai_memory_t * m;

    pthread_mutex_lock(&mapping_mutex);
    for(i = 0; i < (sizeof(mappings) / sizeof(mappings[0])); i++) {
        if((id == mappings[i].id) && (current_mappings & (1 << i))) {
            pthread_mutex_unlock(&mapping_mutex);
            return i;
        }
    }

    for(i = 0; i < (sizeof(mappings) / sizeof(mappings[0])); i++) {
        if(!(current_mappings & (1 << i))) {
            m = simaai_memory_attach(id);
            mappings[i].paddr = simaai_memory_get_phys(m);
            mappings[i].vaddr = simaai_memory_map(m);
            mappings[i].id = id;
            current_mappings |= 1 << i;
            GST_INFO("Storing new mapping of %#x to %#lx(%#lx) at index %d", mappings[i].id,
                      mappings[i].paddr, (uint64_t)(mappings[i].vaddr), i);
            pthread_mutex_unlock(&mapping_mutex);
            return i;
        }
    }

    pthread_mutex_unlock(&mapping_mutex);
    return -1;
}
uint64_t buffer_id_to_paddr(unsigned int id)
{
    simaai_memory_t * m;
    uint64_t res;
    int32_t i = find_buffer_by_id(id);

    if(i >= 0)
        res = mappings[i].paddr;
    else {
        GST_WARNING("Ran out of space to store buffer address, this may have impact on performance or stability");
        m = simaai_memory_attach(id);
        res = simaai_memory_get_phys(m);
    }

    return res;
}

void* buffer_id_to_vaddr(unsigned int id)
{
    simaai_memory_t * m;
    void *res;
    int32_t i = find_buffer_by_id(id);

    if(i >= 0)
        res = mappings[i].vaddr;
    else {
        GST_WARNING("Ran out of space to store buffer address, this may have impact on performance or stability");
        m = simaai_memory_attach(id);
        res = simaai_memory_get_virt(m);
    }

    return res;
}
