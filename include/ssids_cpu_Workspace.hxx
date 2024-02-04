/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-04 AT 08:30 GMT
 */

#pragma once

#include <memory>

#include "spral_compat.hxx" // in case std::align not defined
#include "ssids_rip.hxx"

namespace spral { namespace ssids { namespace cpu {

/** A Workspace is a chunk of memory that can be reused. The get_ptr<T>(len)
 * function provides a pointer to it after ensuring it is of at least the
 * given size. */
class Workspace {
#if defined(__AVX512F__)
  static ipc_ const align = 64;
#elif defined(__AVX__)
  static ipc_ const align = 32;
#else
  static ipc_ const align = 16;
#endif
public:
   Workspace(size_t sz)
   {
      alloc_and_align(sz);
   }
   ~Workspace() {
      ::operator delete(mem_);
   }
   void alloc_and_align(size_t sz) {
      sz_ = sz+align;
      mem_ = ::operator new(sz_);
      mem_aligned_ = mem_;
      if(!std::align(align, sz, mem_aligned_, sz_)) throw std::bad_alloc();
   }
   template <typename T>
   T* get_ptr(size_t len) {
      if(sz_ < len*sizeof(T)) {
         // Need to resize
         ::operator delete(mem_);
         alloc_and_align(len*sizeof(T));
      }
      return static_cast<T*>(mem_aligned_);
   }
private:
   void* mem_;
   void* mem_aligned_;
   size_t sz_;
};

}}} /* end of namespace spral::ssids::cpu */
