# Gst SiMa-Memory allocator

Some details about how the sima-custom-allocator works and example of using the same. There are couple of methods of using the sima-custom allocator.

## First method

* We register the allocator within the plugin context by either finding the pre-registered ones, or re-register them on plugin initialization as below

  To register a custom memory allocator,
  GstAllocator *allocator;
  allocator = g_object_new (simaai_allocator_get_type (), NULL);
  gst_allocator_register ("simaai-allocator", allocator);


  To find the allocator
  GstAllocator * alloc = gst_allocator_find($ALLOCATOR_NAME);

  Ex:

  GstAllocator * alloc = gst_allocator_find("simaai-allocator");

  To create a memory chunk/object using the above we use these

  GstAllocationParams params;
  gst_allocation_params_init (&params);
  // Initializes the memory block by using the allocator
  mem = gst_allocator_alloc (alloc, 1024, &params);

  // To read the memory
  gst_memory_map (mem, &info, GST_MAP_READ);
  gst_memory_unmap (mem, &info);


  // To write to the memory
  gst_memory_map (mem, &info, GST_MAP_WRITE);
  gst_memory_unmap (mem, &info);

## Second method, if we need specific CPU target based allocation, we use the allocator like below

   /// To initialize the allocator, call
   gst_simaai_buffer_memory_init_once ();

   // After init, use the below api to allocate
   GstMemory * mem = simaai_target_mem_alloc($TARGET, $SIZE, $BUFFER_ID_RETURNED);

   // To use the memory,
   GstMapInfo meminfo;
   gst_memory_map(self->priv->mem, &meminfo, GST_MAP_WRITE);
   // Write data
   gst_memory_unmap(self->priv->mem, &meminfo);

   // To wrap the memory object with GstBuffer
   GstBuffer * outbuf = gst_buffer_new();

   gst_buffer_insert_memory (outbuf, -1,
   			    gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
						   (gpointer) meminfo.data,
						   meminfo.size,
						   offset,
						   maxsize,
						   NULL, NULL));

   // push the above buffer to downstream using pad push.

# NOTE:

For more details check the sample test application
