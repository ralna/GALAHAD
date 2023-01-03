/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *
 *  \brief
 *  Additional support functions and wrappers for OpenMP.
 */
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

/* This file wraps the C interface for OpenMP in C++ for style/safety */
namespace spral { namespace omp {

/**
 * \brief Safe wrapper around omp_lock_t ensuring init/cleanup.
 *        See AcquiredLock for locking functionality.
 *
 * This acts as an underlying resource that may be aquired by instantiating
 * an AcquiredLock with this as an argument.
 *
 * \sa AcquiredLock
 */
class Lock {
public:
   Lock(Lock const&) =delete;
   Lock& operator=(Lock const&) =delete;
   Lock() {
#ifdef _OPENMP
      omp_init_lock(&lock_);
#endif /* _OPENMP */
   }
   ~Lock() {
#ifdef _OPENMP
      omp_destroy_lock(&lock_);
#endif /* _OPENMP */
   }
private:
   inline
   void set() {
#ifdef _OPENMP
      omp_set_lock(&lock_);
#endif /* _OPENMP */
   }
   inline
   void unset() {
#ifdef _OPENMP
      omp_unset_lock(&lock_);
#endif /* _OPENMP */
   }
   inline
   bool test() {
#ifdef _OPENMP
      return omp_test_lock(&lock_);
#else
      return true;
#endif /* _OPENMP */
   }

#ifdef _OPENMP
   omp_lock_t lock_;
#endif /* _OPENMP */

   friend class AcquiredLock;
};

/**
 * \brief RAII lock. Acquires lock on construction, releases on destruction.
 */
class AcquiredLock {
public:
   AcquiredLock(Lock& lock)
   : lock_(lock)
   {
      lock_.set();
   }
   ~AcquiredLock() {
      lock_.unset();
   }
private:
   Lock& lock_; ///< Underlying lock.
};

/// Return global thread number (=thread number if not nested)
int get_global_thread_num();

}} /* end of namespace spral::omp */
