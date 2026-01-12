#include <pthread.h>

#include "simamm.h"

typedef struct {
    unsigned int id;
    uint64_t addr;
} buffer_mapping;

static buffer_mapping mappings[sizeof(uint64_t) * 8] = {{ .id = -1, .addr = 0 }};
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

uint64_t buffer_id_to_paddr(unsigned int id)
{
    uint32_t i;
    simaai_memory_t * m;
    uint64_t res = 0;

    pthread_mutex_lock(&mapping_mutex);

    for(i = 0; i < (sizeof(mappings) / sizeof(mappings[0])); i++) {
        if((id == mappings[i].id) && (current_mappings & (1 << i))) {
            res = mappings[i].addr;
            break;
        }
    }

    if(res == 0) {
        for(i = 0; i < (sizeof(mappings) / sizeof(mappings[0])); i++) {
            if(!(current_mappings & (1 << i))) {
                m = simaai_memory_attach(id);
                mappings[i].addr = simaai_memory_get_phys(m);
                mappings[i].id = id;
                current_mappings |= 1 << i;
                GST_INFO("Storing new mapping of %#x to %#lx at index %d", mappings[i].id, mappings[i].addr, i);
                res = mappings[i].addr;
                simaai_memory_free(m);
                break;
            }
        }
    }
    if(res == 0) {
        GST_WARNING("Ran out of space to store buffer address, this may have impact on performance or stability");
        m = simaai_memory_attach(id);
        res = simaai_memory_get_phys(m);
        simaai_memory_free(m);
    }

    pthread_mutex_unlock(&mapping_mutex);

    return res;
}
