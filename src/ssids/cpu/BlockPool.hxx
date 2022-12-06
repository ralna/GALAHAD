/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "omp.hxx"

namespace spral { namespace ssids { namespace cpu {

/** Pool of blocks that can be aquired/released, providing a way to cap
 *  memory usage whilst avoiding fragmentation.
 *  Further, guaruntees blocks are aligned to 32-byte boundaries so are
 *  suitable for AVX usage. */
template <typename T, typename Allocator>
class BlockPool {
   typedef typename std::allocator_traits<Allocator>::template rebind_traits<char> CharAllocTraits;
#if defined(__AVX512F__)
  static const std::size_t align_ = 64; //< Alignment for AVX512 is 64 bytes
#elif defined(__AVX__)
  static const std::size_t align_ = 32; //< Alignment for AVX(2) is 32 bytes
#else
  static const std::size_t align_ = 16; //< Alignment for SSE(2,3,4.1,4.2) or Power's VSX is 16 bytes
#endif
public:
   /* Not copyable */
   BlockPool(BlockPool const&) =delete;
   BlockPool& operator=(BlockPool const&) =delete;
   /** Constructor allocates memory pool */
   BlockPool(std::size_t num_blocks, std::size_t block_dimn, Allocator const& alloc=Allocator())
   : alloc_(alloc), num_blocks_(num_blocks), block_dimn_(block_dimn)
   {
      // Calculate size in elements
      std::size_t sz = block_dimn_*block_dimn_*sizeof(T);
      block_size_ = align_*((sz-1)/align_ + 1);
      mem_ = CharAllocTraits::allocate(alloc_, num_blocks*block_size_);
      // Set up stack of free blocks such that we issue them in order
      pool_.reserve(num_blocks);
      for(int i=num_blocks-1; i>=0; --i) {
         pool_.push_back(reinterpret_cast<T*>(mem_ + i*block_size_));
      }
   }
   ~BlockPool() {
      // FIXME: Throw an exception if we've not had all memory returned?
      CharAllocTraits::deallocate(alloc_, mem_, num_blocks_*block_size_);
   }

   /** Get next free block in a thread-safe fashion.
    *  Return nullptr if it can't find a free block.
    */
   T *get_nowait() {
      T *ptr = nullptr;
      spral::omp::AcquiredLock scopeLock(lock_);
      if(pool_.size() > 0) {
         ptr = pool_.back();
         pool_.pop_back();
      }
      return ptr;
   }
   /** Get next free block in a thread-safe fashion.
    *  Keep trying until it suceeds, use taskyield after each try.
    *  NB: This may deadlock if there are no other tasks releasing blocks.
    */
   T *get_wait() {
      while(true) {
         T *ptr = get_nowait();
         if(ptr) {
            return ptr;
         }
         #pragma omp taskyield
      }
   }
   /** Release block acquired using get_*() function for reuse */
   void release(T *const ptr) {
      spral::omp::AcquiredLock scopeLock(lock_);
      pool_.push_back(ptr);
   }
private:
   typename CharAllocTraits::allocator_type alloc_;
   std::size_t num_blocks_; //< Number of blocks
   std::size_t block_dimn_; //< Blocks are block_size_ x block_size_ elements
   std::size_t block_size_; //< Size of an aligned block in bytes
   char* mem_; //< pointer to allocated memory
   std::vector<T*> pool_; //< stack of free blocks
   spral::omp::Lock lock_; //< lock used for safe access to pool_
};

}}} /* namespace spral::ssids::cpu */
