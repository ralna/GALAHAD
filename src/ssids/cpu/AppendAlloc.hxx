/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

//#define MEM_STATS

#include <memory>

#include "compat.hxx" // for std::align if required

namespace spral { namespace ssids { namespace cpu {

namespace append_alloc_internal {

/** A single fixed size page of memory with allocate function.
 * We are required to guaruntee it is zero'd, so use calloc rather than anything
 * else for the allocation.
 * Deallocation is not supported.
 */
class Page {
#if defined(__AVX512F__)
  static const int align = 64; // 64 byte alignment
#elif defined(__AVX__)
  static const int align = 32; // 32 byte alignment
#else
  static const int align = 16; // 16 byte alignment
#endif
public:
   Page(size_t sz, Page* next=nullptr)
   : next(next), mem_(calloc(sz+align, 1)), ptr_(mem_), space_(sz+align)
   {
      if(!mem_) throw std::bad_alloc();
   }
   ~Page() {
#ifdef MEM_STATS
      uintptr_t used =
         reinterpret_cast<uintptr_t>(ptr_) - reinterpret_cast<uintptr_t>(mem_);
      uintptr_t total = used + space_;
      printf("AppendAlloc: Allocated %16ld (%.2e GB)\n",
            total, 1e-9*double(used));
      printf("AppendAlloc: Used      %16ld (%.2e GB)\n",
            used, 1e-9*double(used));
#endif /* MEM_STATS */
      free(mem_);
   }
   void* allocate(size_t sz) {
      if(!std::align(align, sz, ptr_, space_)) return nullptr;
      void* ret = ptr_;
      ptr_ = (char*)ptr_ + sz;
      space_ -= sz;
      return ret;
   }
public:
   Page* const next;
private:
   void *const mem_; // Pointer to memory so we can free it
   void *ptr_; // Next address to return
   size_t space_; // Amount of free memory
};

/** A memory allocation pool consisting of one or more pages.
 * Deallocation is not supported.
 */
class Pool {
   const size_t PAGE_SIZE = 8*1024*1024; // 8MB
public:
   Pool(size_t initial_size)
   : top_page_(new Page(std::max(PAGE_SIZE, initial_size)))
   {}
   Pool(const Pool&) =delete; // Not copyable
   Pool& operator=(const Pool&) =delete; // Not copyable
   ~Pool() {
      /* Iterate over linked list deleting pages */
      for(Page* page=top_page_; page; ) {
         Page* next = page->next;
         delete page;
         page = next;
      }
   }
   void* allocate(size_t sz) {
      void* ptr;
      #pragma omp critical
      {
         ptr = top_page_->allocate(sz);
         if(!ptr) { // Insufficient space on current top page, make a new one
            top_page_ = new Page(std::max(PAGE_SIZE, sz), top_page_);
            ptr = top_page_->allocate(sz);
         }
      }
      return ptr;
   }
private:
   Page* top_page_;
};

} /* namespace spral::ssids::cpu::append_alloc_internal */

/** An allocator built on top of a pool of pages, with expectation of
 * sequential allocation, and then everything deallocated at the end.
 * Deallocation is not supported.
 */
template <typename T>
class AppendAlloc {
public :
   typedef T               value_type;

   AppendAlloc(size_t initial_size)
   : pool_(new append_alloc_internal::Pool(initial_size))
   {}

   /** Rebind a type T to a type U AppendAlloc */
   template <typename U>
   AppendAlloc(AppendAlloc<U> &other)
   : pool_(other.pool_)
   {}

   T* allocate(std::size_t n) {
      return static_cast<T*>(pool_->allocate(n*sizeof(T)));
   }
   void deallocate(T* p, std::size_t n) {
      throw std::runtime_error("Deallocation not supported on AppendAlloc");
   }
   template<class U>
   bool operator==(AppendAlloc<U> const& rhs) {
      return true;
   }
   template<class U>
   bool operator!=(AppendAlloc<U> const& rhs) {
      return !(*this==rhs);
   }
protected:
   std::shared_ptr<append_alloc_internal::Pool> pool_;
   template <typename U> friend class AppendAlloc;
};

}}} /* namepsace spral::ssids::cpu */
