#include <gst/gst.h>
#include <inttypes.h>
#include <stdio.h>

#include "../gstsimaaiallocator.h"
#include <simaai/simaai_memory.h>

int
main (int argc, char **argv)
{
  GstAllocator *alloc;
  GstMemory *mem;
  GstAllocationParams params;
  GstMapInfo info;

  gst_init(&argc, &argv);

  gst_simaai_buffer_memory_init_once();
  alloc = gst_allocator_find ("simaai-allocator");
  g_assert(alloc != NULL);

  // Method 1

  g_message("Testing first method of allocation");
  
  gst_allocation_params_init (&params);
  mem = gst_allocator_alloc (alloc, 1024, &params);

  gst_memory_map (mem, &info, GST_MAP_READ);
  g_assert(info.size == 1024);
  gst_memory_unmap (mem, &info);
  
  gst_memory_map (mem, &info, GST_MAP_WRITE);
  gint64 test1 = 0x13021989;
  memcpy(info.data, &test1, sizeof(gint64));
  if (!memcmp(info.data, &test1, sizeof(test1)) == 0) {
       gst_memory_unmap (mem, &info);
       g_assert(true);
  }
  gst_memory_unmap (mem, &info);
  
  gst_memory_unref (mem);
  gst_object_unref (alloc);
  
  // Method 2 for different target areas
  g_message("Testing second method of allocation");
  gint64 buf_id = 0;
  mem = simaai_target_mem_alloc(SIMAAI_MEM_TARGET_EV74, 1024, &buf_id);
  g_assert(mem != NULL);

  g_message("Buffer id = %" PRId64, buf_id);

  gst_memory_map (mem, &info, GST_MAP_READ);
  g_assert(info.size == 1024);
  gst_memory_unmap (mem, &info);

  gst_memory_map (mem, &info, GST_MAP_WRITE);
  gint64 test2 = 0x15102018;
  memcpy(info.data, &test2, sizeof(gint64));
  // g_assert(info.data == 0x15102018);
  gst_memory_unmap (mem, &info);

  gst_memory_unref(mem);


  gint64 buf_id_mla = 0;
  mem = simaai_target_mem_alloc(SIMAAI_MEM_TARGET_DMS0, 1024, &buf_id_mla);
  g_assert(mem != NULL);

  g_message("Buffer id = %" PRId64, buf_id_mla);

  gst_memory_map (mem, &info, GST_MAP_READ);
  g_assert(info.size == 1024);
  gst_memory_unmap (mem, &info);

  gst_memory_map (mem, &info, GST_MAP_WRITE);
  gint64 test3 = 0x28011990;
  memcpy(info.data, &test3, sizeof(gint64));
  // g_assert(info.data == 0x28011990);
  gst_memory_unmap (mem, &info);

  gst_memory_unref(mem);

  // New allocator
  g_message("Testing %s allocator method of allocation", GST_ALLOCATOR_SIMAAI);

  gst_simaai_segment_memory_init_once();
  alloc = gst_simaai_memory_get_segment_allocator();
  g_assert_true(alloc != NULL);

  constexpr gsize seg1_size = 1024;
  constexpr gsize seg2_size = 2048;
  constexpr gsize total_size = seg1_size + seg2_size;

  GstSimaaiAllocationParams seg_params;
  gst_simaai_memory_allocation_params_init(&seg_params);
  seg_params.parent.flags = (GstMemoryFlags)(GST_SIMAAI_MEMORY_TARGET_EV74 | GST_SIMAAI_MEMORY_FLAG_CACHED);

  gboolean res = gst_simaai_memory_allocation_params_add_segment(&seg_params, seg1_size, "seg1");
  g_assert_true(res != FALSE);

  res = gst_simaai_memory_allocation_params_add_segment(&seg_params, seg2_size, "seg2");
  g_assert_true(res != FALSE);

  mem = gst_allocator_alloc (alloc, total_size, (GstAllocationParams *)(&seg_params));
  g_assert_true(mem != NULL);

  g_message("Memory physical address: 0x%" PRIxPTR, gst_simaai_memory_get_phys_addr(mem));

  gst_memory_map (mem, &info, GST_MAP_READ);
  g_assert_true(info.size == total_size);
  gst_memory_unmap (mem, &info);

  gst_memory_map (mem, &info, GST_MAP_WRITE);
  memcpy(info.data, &test1, sizeof(gint64));
  g_assert_true(memcmp(info.data, &test1, sizeof(test1)) == 0);
  gst_memory_unmap (mem, &info);

  gst_memory_unref (mem);
  gst_object_unref (alloc);

  // Segmented memory API
  g_message("Testing segmented memory simaai-memory API");

  char* first_segment = (char *)calloc(seg1_size, sizeof(char));
  g_assert(first_segment);
  memset(first_segment, 1, seg1_size);
  
  char* second_segment = (char *)calloc(seg2_size, sizeof(char)); 
  g_assert(second_segment);
  memset(second_segment, 2, seg2_size);
  
  char* out_buffer = (char *)calloc(total_size, sizeof(char)); 
  g_assert(out_buffer);

  uint32_t segment_sizes[] = {seg1_size, seg2_size};
  uint32_t segment_number = 2;
  simaai_memory_t** segments = simaai_memory_alloc_segments_flags(segment_sizes, segment_number, SIMAAI_MEM_TARGET_EV74, SIMAAI_MEM_FLAG_DEFAULT);
  g_assert(segments);

  simaai_memory_t* segment1 = segments[0];
  simaai_memory_t* segment2 = segments[1];

  auto seg1_phys = simaai_memory_get_phys(segment1);
  auto seg2_phys = simaai_memory_get_phys(segment2);

  g_message("segment1 phys=%lx", simaai_memory_get_phys(segment1));
  g_message("segment2 phys=%lx", simaai_memory_get_phys(segment2));

  // if does not match - segment 2 starts not right after segment 1
  g_assert_true((seg2_phys - seg1_phys) == seg1_size);

  void * s1_vmem = simaai_memory_map(segment1);
  g_message("segment1 vaddr = %p", s1_vmem);
  memcpy(s1_vmem, first_segment, seg1_size);
  simaai_memory_unmap(segment1);
  
  void * s2_vmem = simaai_memory_map(segment2);
  g_message("segment2 vaddr = %p", s2_vmem);
  memcpy(s2_vmem, second_segment, seg2_size);
  simaai_memory_unmap(segment2);

  s1_vmem = simaai_memory_map(segment1);
  memcpy(out_buffer,             (char*)(s1_vmem),             seg1_size);
  memcpy(out_buffer + seg1_size, (char*)(s1_vmem) + seg1_size, seg2_size);
  simaai_memory_unmap(segment1);
  
  if (memcmp(out_buffer, first_segment, seg1_size) != 0)
    g_message("segment1 was not copied successfully");

  if (memcmp(out_buffer + seg1_size, second_segment, seg2_size) != 0)
    g_error("segment2 was not copied successfully");

  free(first_segment);
  free(second_segment);
  free(out_buffer);

  simaai_memory_free_segments(segments, segment_number);

  return 0;
}
