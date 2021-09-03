/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *
 *  \brief
 *  Implements HwlocTopology wrapper around hwloc library
 */
#pragma once
#include "config.h"
#ifdef HAVE_HWLOC

#include <vector>

#include <hwloc.h>
#ifdef HAVE_NVCC
#include <cuda_runtime_api.h>
#include <hwloc/cudart.h>
#endif /* HAVE_NVCC */

namespace spral { namespace hw_topology {

/**
 * \brief Object orientated wrapper around hwloc topology.
 */
class HwlocTopology {
public:
   // \{
   // Not copyable
   HwlocTopology(HwlocTopology const&) =delete;
   HwlocTopology operator=(HwlocTopology const&) =delete;
   // \}
   /** \brief Constructor */
   HwlocTopology() {
      hwloc_topology_init(&topology_);
#if HWLOC_API_VERSION >= 0x20000
      hwloc_topology_set_type_filter(topology_, HWLOC_OBJ_OS_DEVICE,
            HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
      hwloc_topology_set_type_filter(topology_, HWLOC_OBJ_PCI_DEVICE,
            HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
#else /* HWLOC_API_VERSION */
      hwloc_topology_set_flags(topology_, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
#endif /* HWLOC_API_VERSION */
      hwloc_topology_load(topology_);
   }
   /** \brief Destructor */
   ~HwlocTopology() {
      hwloc_topology_destroy(topology_);
   }

   /** \brief Return vector of Numa nodes or just machine object */
   std::vector<hwloc_obj_t> get_numa_nodes() const {
      std::vector<hwloc_obj_t> regions;
      int nregions = hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_NODE);
      if(nregions==0) {
         // No regions, just give machine
         regions.push_back(
               hwloc_get_obj_by_type(topology_, HWLOC_OBJ_MACHINE, 0)
               );
         return regions;
      } else {
         // Iterate over regions, adding them
         regions.reserve(nregions);
         for(int i=0; i<nregions; ++i)
            regions.push_back(
                  hwloc_get_obj_by_type(topology_, HWLOC_OBJ_NODE, i)
                  );
         return regions;
      }
   }

   /** \brief Return number of cores associated to object. */
   int count_cores(hwloc_obj_t const& obj) const {
      return count_type(obj, HWLOC_OBJ_CORE);
   }

   /** \brief Return list of gpu indices associated to object. */
   std::vector<int> get_gpus(hwloc_obj_t const& obj) const {
      std::vector<int> gpus;
#ifdef HAVE_NVCC
      int ngpu;
      cudaError_t cuda_error = cudaGetDeviceCount(&ngpu);
      if(cuda_error != cudaSuccess) {
         //printf("Error using CUDA. Assuming no GPUs.\n");
         return gpus; // empty
      }
      /* Now for each device search up its topology tree and see if we
       * encounter obj. */
      for(int i=0; i<ngpu; ++i) {
         // TODO: Switch between the two calls below as appropriate.
         // hwloc_obj_t p = hwloc_cudart_get_device_osdev_by_index(topology_, i);
         hwloc_obj_t p = hwloc_cudart_get_device_pcidev(topology_, i);
         for(; p; p=p->parent) {
            if(p==obj) {
               gpus.push_back(i);
               break;
            }
         }
      }
#endif
      return gpus; // will be empty ifndef HAVE_NVCC
   }

private:
   int count_type(hwloc_obj_t const& obj, hwloc_obj_type_t type) const {
      if(obj->type == type) return 1;
      int count = 0;
      for(unsigned int i=0; i<obj->arity; ++i)
         count += count_type(obj->children[i], type);
      return count;
   }

   hwloc_topology_t topology_; ///< Underlying topology object
};

}} /* namespace spral::hw_topology */

#endif /* HAVE_HWLOC */
