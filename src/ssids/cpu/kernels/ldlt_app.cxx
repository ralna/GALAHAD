/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#include "ssids/cpu/kernels/ldlt_app.hxx"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#include "compat.hxx"
#include "ssids/profile.hxx"
#include "ssids/cpu/BlockPool.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/kernels/block_ldlt.hxx"
#include "ssids/cpu/kernels/calc_ld.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/common.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

namespace spral { namespace ssids { namespace cpu {

namespace ldlt_app_internal {

static const int INNER_BLOCK_SIZE = 32;

/** \return number of blocks for given n */
inline int calc_nblk(int n, int block_size) {
   return (n-1) / block_size + 1;
}

/** \return block size of block blk if maximum in dimension is n */
inline int calc_blkn(int blk, int n, int block_size) {
   return std::min(block_size, n-blk*block_size);
}

/** \brief Data about block column of factorization; handles operations
 *         concerning number of eliminated variables and stores D.
 *  \tparam T underlying data type, e.g. double
 */
template<typename T>
class Column {
public:
   bool first_elim; ///< True if first column with eliminations
   int nelim; ///< Number of eliminated entries in this column
   T *d; ///< Pointer to local d

   // \{
   Column(Column const&) =delete; // must be unique
   Column& operator=(Column const&) =delete; // must be unique
   Column() =default;
   // \}

   /** \brief Initialize number of passed columns ready for reduction
    *  \param passed number of variables passing a posteori pivot test in block
    */
   void init_passed(int passed) {
      spral::omp::AcquiredLock scopeLock(lock_);
      npass_ = passed;
   }
   /** \brief Update number of passed columns.
    *  \details Aquires a lock before doing a minimum reduction across blocks
    *  \param passed number of variables passing a posteori pivot test in block
    */
   void update_passed(int passed) {
      spral::omp::AcquiredLock scopeLock(lock_);
      npass_ = std::min(npass_, passed);
   }
   /** \brief Test if column has failed (in unpivoted case), recording number of
    *         blocks in column that have passed. To be called once per block
    *         in the column.
    *  \details Whilst this check could easily be done without calling this
    *           routine, the atomic recording of number that have passed would
    *           not be done, and this is essential for calculating number of
    *           sucessful columns in the case of a global cancellation.
    *  \param passed number of pivots that succeeded for a block
    *  \returns true if passed < nelim */
   bool test_fail(int passed) {
      bool fail = (passed < nelim);
      if(!fail) {
         // Record number of blocks in column passing this test
         spral::omp::AcquiredLock scopeLock(lock_);
         ++npass_;
      }
      return fail;
   }

   /** \brief Adjust nelim after all blocks of row/column have completed to
    *         avoid split 2x2 pivots. Also updates next_elim.
    *  \details If a split 2x2 pivot is detected, the number of eliminated
    *           variables is reduced by one. This routine also sets first_elim
    *           to true if this is the first column to successfully eliminated
    *           a variable, and sets nelim for this column.
    *  \param next_elim global number of eliminated pivots to be updated based
    *         on number eliminated in this column. */
   void adjust(int& next_elim) {
      // Test if last passed column was first part of a 2x2: if so,
      // decrement npass
      spral::omp::AcquiredLock scopeLock(lock_);
      if(npass_>0) {
         T d11 = d[2*(npass_-1)+0];
         T d21 = d[2*(npass_-1)+1];
         if(std::isfinite(d11) && // not second half of 2x2
               d21 != 0.0)        // not a 1x1 or zero pivot
            npass_--;              // so must be first half 2x2
      }
      // Update elimination progress
      first_elim = (next_elim==0 && npass_>0);
      next_elim += npass_;
      nelim = npass_;
   }

   /** \brief Move entries of permutation for eliminated entries backwards to
    *         close up space from failed columns, whilst extracting failed
    *         entries.
    *  \details n entries of perm are moved to elim_perm (that may overlap
    *           with perm). Uneliminated variables are placed into failed_perm.
    *  \param n number of entries in block to be moved to elim_perm or failed.
    *  \param perm[n] source pointer
    *  \param elim_perm destination pointer for eliminated columns
    *         from perm, first nelim entries are filled on output.
    *  \param failed_perm destination pointer for failed columns from
    *         perm first (n-nelim) entries are filled on output.
    *  \internal Note that there is no need to consider a similar operation for
    *            d[] as it is only used for eliminated variables.
    */
   void move_back(int n, int const* perm, int* elim_perm, int* failed_perm) {
      if(perm != elim_perm) { // Don't move if memory is identical
         for(int i=0; i<nelim; ++i)
            *(elim_perm++) = perm[i];
      }
      // Copy failed perm
      for(int i=nelim; i<n; ++i)
         *(failed_perm++) = perm[i];
   }

   /** \brief return number of passed columns */
   int get_npass() const {
     spral::omp::AcquiredLock scopeLock(lock_);
     return npass_;
   }

private:
   mutable spral::omp::Lock lock_; ///< lock for altering npass
   int npass_=0; ///< reduction variable for nelim
};

/** \brief Stores data about block columns
 *  \details A wrapper around a vector of Column, also handles local permutation
 *           vector and calculation of nelim in unpivoted factorization
 *  \tparam T underlying datatype e.g. double
 *  \tparam IntAlloc Allocator specialising in int used for internal memory
 *          allocation.
 * */
template<typename T, typename IntAlloc>
class ColumnData {
   // \{
   typedef typename std::allocator_traits<IntAlloc>::template rebind_traits<Column<T>> ColAllocTraits;
   typedef typename std::allocator_traits<IntAlloc> IntAllocTraits;
   // \}
public:
   // \{
   ColumnData(ColumnData const&) =delete; //not copyable
   ColumnData& operator=(ColumnData const&) =delete; //not copyable
   // \}
   /** \brief Constructor
    *  \param n number of columns
    *  \param block_size block size
    *  \param alloc allocator instance to use for allocation
    */
   ColumnData(int n, int block_size, IntAlloc const& alloc)
   : n_(n), block_size_(block_size), alloc_(alloc)
   {
      int nblk = calc_nblk(n_, block_size_);
      typename ColAllocTraits::allocator_type colAlloc(alloc_);
      cdata_ = ColAllocTraits::allocate(colAlloc, nblk);
      for(int i=0; i<nblk; ++i)
         ColAllocTraits::construct(colAlloc, &cdata_[i]);
      lperm_ = IntAllocTraits::allocate(alloc_, nblk*block_size_);
   }
   ~ColumnData() {
      int nblk = calc_nblk(n_, block_size_);
      IntAllocTraits::deallocate(alloc_, lperm_, nblk*block_size_);
      typename ColAllocTraits::allocator_type colAlloc(alloc_);
      ColAllocTraits::deallocate(colAlloc, cdata_, nblk);
   }

   /** \brief Returns Column instance for given column
    *  \param idx block column
    */
   Column<T>& operator[](int idx) { return cdata_[idx]; }

   /** \brief Return local permutation pointer for given column
    *  \param blk block column
    *  \return pointer to local permutation
    */
   int* get_lperm(int blk) { return &lperm_[blk*block_size_]; }

   /** \brief Calculate number of eliminated columns in unpivoted case
    *  \param m number of rows in matrix
    *  \return number of sucesfully eliminated columns
    */
   int calc_nelim(int m) const {
      int mblk = calc_nblk(m, block_size_);
      int nblk = calc_nblk(n_, block_size_);
      int nelim = 0;
      for(int j=0; j<nblk; ++j) {
         if(cdata_[j].get_npass() == mblk-j) {
            nelim += cdata_[j].nelim;
         } else {
            break; // After first failure, no later pivots are valid
         }
      }
      return nelim;
   };

private:
   int const n_; ///< number of columns in matrix
   int const block_size_; ///< block size for matrix
   IntAlloc alloc_; ///< internal copy of allocator to be used in destructor
   Column<T> *cdata_; ///< underlying array of columns
   int* lperm_; ///< underlying local permutation
};


/** Returns true if ptr is suitably aligned for AVX, false if not */
bool is_aligned(void* ptr) {
#if defined(__AVX512F__)
  const int align = 64;
#elif defined(__AVX__)
  const int align = 32;
#else
  const int align = 16;
#endif
   return (reinterpret_cast<uintptr_t>(ptr) % align == 0);
}

/** Move up eliminated entries to fill any gaps left by failed pivots
 *  within diagonal block.
 *  Note that out and aval may overlap. */
template<typename T, typename Column>
void move_up_diag(Column const& idata, Column const& jdata, T* out, T const* aval, int lda) {
   if(out == aval) return; // don't bother moving if memory is the same
   for(int j=0; j<jdata.nelim; ++j)
   for(int i=0; i<idata.nelim; ++i)
      out[j*lda+i] = aval[j*lda+i];
}

/** Move up eliminated entries to fill any gaps left by failed pivots
 *  within rectangular block of matrix.
 *  Note that out and aval may overlap. */
template<typename T, typename Column>
void move_up_rect(int m, int rfrom, Column const& jdata, T* out, T const* aval, int lda) {
   if(out == aval) return; // don't bother moving if memory is the same
   for(int j=0; j<jdata.nelim; ++j)
   for(int i=rfrom; i<m; ++i)
      out[j*lda+i] = aval[j*lda+i];
}

/** Copies failed rows and columns^T to specified locations */
template<typename T, typename Column>
void copy_failed_diag(int m, int n, Column const& idata, Column const& jdata, T* rout, T* cout, T* dout, int ldout, T const* aval, int lda) {
   /* copy rows */
   for(int j=0; j<jdata.nelim; ++j)
   for(int i=idata.nelim, iout=0; i<m; ++i, ++iout)
      rout[j*ldout+iout] = aval[j*lda+i];
   /* copy cols in transpose (not for diagonal block) */
   if(&idata != &jdata) {
      for(int j=jdata.nelim, jout=0; j<n; ++j, ++jout)
      for(int i=0; i<idata.nelim; ++i)
         cout[i*ldout+jout] = aval[j*lda+i];
   }
   /* copy intersection of failed rows and cols */
   for(int j=jdata.nelim, jout=0; j<n; j++, ++jout)
   for(int i=idata.nelim, iout=0; i<m; ++i, ++iout)
      dout[jout*ldout+iout] = aval[j*lda+i];
}

/** Copies failed columns to specified location */
template<typename T, typename Column>
void copy_failed_rect(int m, int n, int rfrom, Column const& jdata, T* cout, int ldout, T const* aval, int lda) {
   for(int j=jdata.nelim, jout=0; j<n; ++j, ++jout)
      for(int i=rfrom; i<m; ++i)
         cout[jout*ldout+i] = aval[j*lda+i];
}

/** Check if a block satisifies pivot threshold (colwise version) */
template <enum operation op, typename T>
int check_threshold(int rfrom, int rto, int cfrom, int cto, T u, T* aval, int lda) {
   // Perform threshold test for each uneliminated row/column
   int least_fail = (op==OP_N) ? cto : rto;
   for(int j=cfrom; j<cto; j++)
   for(int i=rfrom; i<rto; i++)
      if(fabs(aval[j*lda+i]) > 1.0/u) {
         if(op==OP_N) {
            // must be least failed col
            return j;
         } else {
            // may be an earlier failed row
            least_fail = std::min(least_fail, i);
            break;
         }
      }
   // If we get this far, everything is good
   return least_fail;
}

/** Performs solve with diagonal block \f$L_{21} = A_{21} L_{11}^{-T} D_1^{-1}\f$. Designed for below diagonal. */
/* NB: d stores (inverted) pivots as follows:
 * 2x2 ( a b ) stored as d = [ a b Inf c ]
 *     ( b c )
 * 1x1  ( a )  stored as d = [ a 0.0 ]
 * 1x1  ( 0 ) stored as d = [ 0.0 0.0 ]
 */
template <enum operation op, typename T>
void apply_pivot(int m, int n, int from, const T *diag, const T *d, const T small, T* aval, int lda) {
   if(op==OP_N && from > m) return; // no-op
   if(op==OP_T && from > n) return; // no-op

   if(op==OP_N) {
      // Perform solve L_11^-T
      host_trsm<T>(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_UNIT,
            m, n, 1.0, diag, lda, aval, lda);
      // Perform solve L_21 D^-1
      for(int i=0; i<n; ) {
         if(i+1==n || std::isfinite(d[2*i+2])) {
            // 1x1 pivot
            T d11 = d[2*i];
            if(d11 == 0.0) {
               // Handle zero pivots carefully
               for(int j=0; j<m; j++) {
                  T v = aval[i*lda+j];
                  aval[i*lda+j] = 
                     (fabs(v)<small) ? 0.0
                                     : std::numeric_limits<T>::infinity()*v;
                  // NB: *v above handles NaNs correctly
               }
            } else {
               // Non-zero pivot, apply in normal fashion
               for(int j=0; j<m; j++)
                  aval[i*lda+j] *= d11;
            }
            i++;
         } else {
            // 2x2 pivot
            T d11 = d[2*i];
            T d21 = d[2*i+1];
            T d22 = d[2*i+3];
            for(int j=0; j<m; j++) {
               T a1 = aval[i*lda+j];
               T a2 = aval[(i+1)*lda+j];
               aval[i*lda+j]     = d11*a1 + d21*a2;
               aval[(i+1)*lda+j] = d21*a1 + d22*a2;
            }
            i += 2;
         }
      }
   } else { /* op==OP_T */
      // Perform solve L_11^-1
      host_trsm<T>(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT,
            m, n-from, 1.0, diag, lda, &aval[from*lda], lda);
      // Perform solve D^-T L_21^T
      for(int i=0; i<m; ) {
         if(i+1==m || std::isfinite(d[2*i+2])) {
            // 1x1 pivot
            T d11 = d[2*i];
            if(d11 == 0.0) {
               // Handle zero pivots carefully
               for(int j=from; j<n; j++) {
                  T v = aval[j*lda+i];
                  aval[j*lda+i] = 
                     (fabs(v)<small) ? 0.0 // *v handles NaNs
                                     : std::numeric_limits<T>::infinity()*v;
                  // NB: *v above handles NaNs correctly
               }
            } else {
               // Non-zero pivot, apply in normal fashion
               for(int j=from; j<n; j++) {
                  aval[j*lda+i] *= d11;
               }
            }
            i++;
         } else {
            // 2x2 pivot
            T d11 = d[2*i];
            T d21 = d[2*i+1];
            T d22 = d[2*i+3];
            for(int j=from; j<n; j++) {
               T a1 = aval[j*lda+i];
               T a2 = aval[j*lda+(i+1)];
               aval[j*lda+i]     = d11*a1 + d21*a2;
               aval[j*lda+(i+1)] = d21*a1 + d22*a2;
            }
            i += 2;
         }
      }
   }
}

/** \brief Stores backups of matrix blocks using a complete copy of matrix.
 *  \details Note that whilst a complete copy of matrix is allocated, copies
 *           of blocks are still stored individually to facilitate cache
 *           locality.
 *  \tparam T underlying data type, e.g. double
 *  \tparam Allocator allocator to use when allocating memory
 */
template <typename T, typename Allocator=std::allocator<T>>
class CopyBackup {
   // \{
   typedef typename std::allocator_traits<Allocator>::template rebind_traits<bool> BATraits;
   // \}
public:
   // \{
   CopyBackup(CopyBackup const&) =delete;
   CopyBackup& operator=(CopyBackup const&) =delete;
   // \}
   /** \brief constructor
    *  \param m number of rows in matrix
    *  \param n number of blocks in matrix
    *  \param block_size dimension of a block in rows or columns
    *  \param alloc allocator instance to use when allocating memory
    */
   CopyBackup(int m, int n, int block_size, Allocator const& alloc=Allocator())
   : alloc_(alloc), m_(m), n_(n), mblk_(calc_nblk(m,block_size)),
     block_size_(block_size), ldcopy_(align_lda<T>(m_)),
     acopy_(alloc_.allocate(n_*ldcopy_))
   {
      typename BATraits::allocator_type boolAlloc(alloc_);
   }
   ~CopyBackup() {
      release_all_memory();
   }

   /** \brief release all associated memory; no further operations permitted.
    *  \details Storing a complete copy of the matrix is memory intensive, this
    *           routine is provided to free that storage whilst the instance is
    *           still in scope, for cases where it cannot otherwise easily be
    *           reclaimed as soon as required.
    */
   void release_all_memory() {
      if(acopy_) {
         alloc_.deallocate(acopy_, n_*ldcopy_);
         acopy_ = nullptr;
      }
   }

   /** \brief Release memory associated with backup of given block.
    *  \details Provided for compatability with PoolBackup, this
    *           routine is a no-op for CopyBackup.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    */
   void release(int iblk, int jblk) { /* no-op */ }

   /** \brief Create a restore point for the given block.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point(int iblk, int jblk, T const* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=0; j<get_ncol(jblk); j++)
      for(int i=0; i<get_nrow(iblk); i++)
         lwork[j*ldcopy_+i] = aval[j*lda+i];
   }

   /** \brief Apply row permutation to block and create a restore point.
    *  \details The row permutation is applied before taking the copy. This
    *           routine is provided as the row permutation requires taking a
    *           copy anyway, so they can be profitably combined.
    *  \param iblk row index of block
    *  \param jblk column index of block
    *  \param nperm number of rows to permute (allows for rectangular blocks)
    *  \param perm the permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point_with_row_perm(int iblk, int jblk, int nperm,
         int const* perm, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=0; j<get_ncol(jblk); j++) {
         for(int i=0; i<nperm; i++) {
            int r = perm[i];
            lwork[j*ldcopy_+i] = aval[j*lda+r];
         }
         for(int i=nperm; i<get_nrow(iblk); i++) {
            lwork[j*ldcopy_+i] = aval[j*lda+i];
         }
      }
      for(int j=0; j<get_ncol(jblk); j++)
      for(int i=0; i<nperm; i++)
         aval[j*lda+i] = lwork[j*ldcopy_+i];
   }

   /** \brief Apply column permutation to block and create a restore point.
    *  \details The column permutation is applied before taking the copy. This
    *           routine is provided as the permutation requires taking a
    *           copy anyway, so they can be profitably combined.
    *  \param iblk row index of block
    *  \param jblk column index of block
    *  \param perm the permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point_with_col_perm(int iblk, int jblk, const int *perm, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=0; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=0; i<get_nrow(iblk); i++)
            lwork[j*ldcopy_+i] = aval[c*lda+i];
      }
      for(int j=0; j<get_ncol(jblk); j++)
      for(int i=0; i<get_nrow(iblk); i++)
         aval[j*lda+i] = lwork[j*ldcopy_+i];
   }

   /** \brief Restore submatrix (rfrom:, cfrom:) of block from backup.
    *  \param iblk row of block
    *  \param jblk column of block
    *  \param rfrom row from which to start restoration
    *  \param cfrom column from which to start restoration
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void restore_part(int iblk, int jblk, int rfrom, int cfrom, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=cfrom; j<get_ncol(jblk); j++)
      for(int i=rfrom; i<get_nrow(iblk); i++)
         aval[j*lda+i] = lwork[j*ldcopy_+i];
   }

   /** \brief Restore submatrix (from:, from:) from a symmetric permutation of
    *         backup.
    *  \details The backup will have been stored pritor to a symmetric
    *           permutation associated with the factorization of a diagonal
    *           block. This routine restores any failed columns, taking into
    *           account the supplied permutation.
    *  \param iblk row of block
    *  \param jblk column of block
    *  \param from row and column from which to start restoration
    *  \param perm permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void restore_part_with_sym_perm(int iblk, int jblk, int from, const int *perm, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=from; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=from; i<get_ncol(jblk); i++) {
            int r = perm[i];
            aval[j*lda+i] = (r>c) ? lwork[c*ldcopy_+r]
                                  : lwork[r*ldcopy_+c];
         }
         for(int i=get_ncol(jblk); i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[c*ldcopy_+i];
      }
   }

private:
   /** \brief returns pointer to internal backup of given block */
   inline T* get_lwork(int iblk, int jblk) {
      return &acopy_[jblk*block_size_*ldcopy_+iblk*block_size_];
   }
   /** \brief return number of columns in given block column */
   inline int get_ncol(int blk) const {
      return calc_blkn(blk, n_, block_size_);
   }
   /** \brief return number of rows in given block row */
   inline int get_nrow(int blk) const {
      return calc_blkn(blk, m_, block_size_);
   }

   Allocator alloc_; ///< internal copy of allocator needed for destructor
   int const m_; ///< number of rows in matrix
   int const n_; ///< number of columns in matrix
   int const mblk_; ///< number of block rows in matrix
   int const block_size_; ///< block size
   size_t const ldcopy_; ///< leading dimension of acopy_
   T* acopy_; ///< internal storage for copy of matrix
};

/** \brief Stores backups of matrix blocks using a pool of memory.
 *  \details The pool is not necessarily as large as the full matrix, so in
 *  some cases allocation will have to wait until a block is released by
 *  another task. In this case, OpenMP taskyield is used.
 *  \tparam T underlying data type, e.g. double
 *  \tparam Allocator allocator to use when allocating memory
 */
template <typename T, typename Allocator=std::allocator<T*>>
class PoolBackup {
   //! \{
   typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T*> TptrAlloc;
   //! \}
public:
   /** \brief Constructor
    *  \param m number of rows in matrix
    *  \param n number of blocks in matrix
    *  \param block_size dimension of a block in rows or columns
    *  \param alloc allocator instance to use when allocating memory
    */
   // FIXME: reduce pool size
   PoolBackup(int m, int n, int block_size, Allocator const& alloc=Allocator())
   : m_(m), n_(n), block_size_(block_size), mblk_(calc_nblk(m,block_size)),
     pool_(calc_nblk(n,block_size)*((calc_nblk(n,block_size)+1)/2+mblk_), block_size, alloc),
     ptr_(mblk_*calc_nblk(n,block_size), alloc)
   {}

   /** \brief Release memory associated with backup of given block.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    */
   void release(int iblk, int jblk) {
      pool_.release(ptr_[jblk*mblk_+iblk]);
      ptr_[jblk*mblk_+iblk] = nullptr;
   }

   /** \brief Create a restore point for the given block.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point(int iblk, int jblk, T const* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      lwork = pool_.get_wait();
      for(int j=0; j<get_ncol(jblk); j++)
      for(int i=0; i<get_nrow(iblk); i++)
         lwork[j*block_size_+i] = aval[j*lda+i];
   }

   /** \brief Apply row permutation to block and create a restore point.
    *  \details The row permutation is applied before taking the copy. This
    *           routine is provided as the row permutation requires taking a
    *           copy anyway, so they can be profitably combined.
    *  \param iblk row index of block
    *  \param jblk column index of block
    *  \param nperm number of rows to permute (allows for rectangular blocks)
    *  \param perm the permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point_with_row_perm(int iblk, int jblk, int nperm,
         int const* perm, T* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      lwork = pool_.get_wait();
      for(int j=0; j<get_ncol(jblk); j++) {
         for(int i=0; i<nperm; i++) {
            int r = perm[i];
            lwork[j*block_size_+i] = aval[j*lda+r];
         }
         for(int i=nperm; i<get_nrow(iblk); i++) {
            lwork[j*block_size_+i] = aval[j*lda+i];
         }
      }
      for(int j=0; j<get_ncol(jblk); j++)
      for(int i=0; i<nperm; i++)
         aval[j*lda+i] = lwork[j*block_size_+i];
   }

   /** \brief Apply column permutation to block and create a restore point.
    *  \details The column permutation is applied before taking the copy. This
    *           routine is provided as the permutation requires taking a
    *           copy anyway, so they can be profitably combined.
    *  \param iblk row index of block
    *  \param jblk column index of block
    *  \param perm the permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point_with_col_perm(int iblk, int jblk,
         int const* perm, T* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      lwork = pool_.get_wait();
      for(int j=0; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=0; i<get_nrow(iblk); i++)
            lwork[j*block_size_+i] = aval[c*lda+i];
      }
      for(int j=0; j<get_ncol(jblk); j++)
      for(int i=0; i<get_nrow(iblk); i++)
         aval[j*lda+i] = lwork[j*block_size_+i];
   }

   /** \brief Restore submatrix (rfrom:, cfrom:) of block from backup.
    *  \param iblk row of block
    *  \param jblk column of block
    *  \param rfrom row from which to start restoration
    *  \param cfrom column from which to start restoration
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void restore_part(int iblk, int jblk, int rfrom, int cfrom, T* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      for(int j=cfrom; j<get_ncol(jblk); j++)
      for(int i=rfrom; i<get_nrow(iblk); i++)
         aval[j*lda+i] = lwork[j*block_size_+i];
   }

   /** \brief Restore submatrix (from:, from:) from a symmetric permutation of
    *         backup.
    *  \details The backup will have been stored pritor to a symmetric
    *           permutation associated with the factorization of a diagonal
    *           block. This routine restores any failed columns, taking into
    *           account the supplied permutation.
    *  \param iblk row of block
    *  \param jblk column of block
    *  \param from row and column from which to start restoration
    *  \param perm permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void restore_part_with_sym_perm(int iblk, int jblk, int from,
         int const* perm, T* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      for(int j=from; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=from; i<get_ncol(jblk); i++) {
            int r = perm[i];
            aval[j*lda+i] = (r>c) ? lwork[c*block_size_+r]
                                  : lwork[r*block_size_+c];
         }
         for(int i=get_ncol(jblk); i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[c*block_size_+i];
      }
   }

private:
   /** \brief return number of columns in given block column */
   inline int get_ncol(int blk) {
      return calc_blkn(blk, n_, block_size_);
   }
   /** \brief return number of rows in given block row */
   inline int get_nrow(int blk) {
      return calc_blkn(blk, m_, block_size_);
   }

   int const m_; ///< number of rows in main matrix
   int const n_; ///< number of columns in main matrix
   int const block_size_; ///< block size of main matrix
   int const mblk_; ///< number of block rows in main matrix
   BlockPool<T, Allocator> pool_; ///< pool of blocks
   std::vector<T*, TptrAlloc> ptr_; ///< map from pointer matrix entry to block
};

template<typename T,
         int BLOCK_SIZE,
         typename Backup,
         bool use_tasks, // Use tasks, so we can disable on one or more levels
         bool debug=false,
         typename Allocator=std::allocator<T>
         >
class LDLT;

/** \brief Functional wrapper around a block of the underlying matrix.
 *  \details Provides a light-weight wrapper around blocks of the matrix
 *           to provide location-aware functionality and thus safety.
 *  \tparam T Underlying datatype, e.g. double.
 *  \tparam INNER_BLOCK_SIZE The inner block size to be used for recursion
 *          decisions in factor().
 *  \tparam IntAlloc an allocator for type int used in specification of
 *          ColumnData type.
 */
template<typename T, int INNER_BLOCK_SIZE, typename IntAlloc>
class Block {
public:
   /** \brief Constuctor.
    *  \param i Block's row index.
    *  \param j Block's column index.
    *  \param m Number of rows in matrix.
    *  \param n Number of columns in matrix.
    *  \param cdata ColumnData for factorization.
    *  \param a Pointer to underlying storage of matrix.
    *  \param lda Leading dimension of a.
    *  \param block_size The block size.
    */
   Block(int i, int j, int m, int n, ColumnData<T,IntAlloc>& cdata, T* a,
         int lda, int block_size)
   : i_(i), j_(j), m_(m), n_(n), lda_(lda), block_size_(block_size),
     cdata_(cdata), aval_(&a[j*block_size*lda+i*block_size])
   {}

   /** \brief Create backup of this block.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void backup(Backup& backup) {
      backup.create_restore_point(i_, j_, aval_, lda_);
   }

   /** \brief Apply column permutation to block and create a backup.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void apply_rperm_and_backup(Backup& backup) {
      backup.create_restore_point_with_row_perm(
            i_, j_, get_ncol(i_), cdata_.get_lperm(i_), aval_, lda_
            );
   }

   /** \brief Apply row permutation to block.
    *  \param work Thread-specific workspace.
    */
   void apply_rperm(Workspace& work) {
      int ldl = align_lda<T>(block_size_);
      T* lwork = work.get_ptr<T>(ncol()*ldl);
      int* lperm = cdata_.get_lperm(i_);
      // Copy into lwork with permutation
      for(int j=0; j<ncol(); ++j) {
         for(int i=0; i<get_ncol(i_); ++i) {
            int r = lperm[i];
            lwork[j*ldl+i] = aval_[j*lda_+r];
         }
      }
      // Copy back again
      for(int j=0; j<ncol(); ++j)
      for(int i=0; i<get_ncol(i_); ++i)
         aval_[j*lda_+i] = lwork[j*ldl+i];
   }

   /** \brief Apply inverse of row permutation to block.
    *  \details Intended for recovery from failed Cholesky-like factorization.
    *  \param work Thread-specific workspace.
    */
   void apply_inv_rperm(Workspace& work) {
      int ldl = align_lda<T>(block_size_);
      T* lwork = work.get_ptr<T>(ncol()*ldl);
      int* lperm = cdata_.get_lperm(i_);
      // Copy into lwork with permutation
      for(int j=0; j<ncol(); ++j) {
         for(int i=0; i<get_ncol(i_); ++i) {
            int r = lperm[i];
            lwork[j*ldl+r] = aval_[j*lda_+i];
         }
      }
      // Copy back again
      for(int j=0; j<ncol(); ++j)
      for(int i=0; i<get_ncol(i_); ++i)
         aval_[j*lda_+i] = lwork[j*ldl+i];
   }

   /** \brief Apply column permutation to block and create a backup.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void apply_cperm_and_backup(Backup& backup) {
      backup.create_restore_point_with_col_perm(
            i_, j_, cdata_.get_lperm(j_), aval_, lda_
            );
   }

   /** \brief Apply column permutation to block.
    *  \param work Thread-specific workspace.
    */
   void apply_cperm(Workspace& work) {
      int ldl = align_lda<T>(block_size_);
      T* lwork = work.get_ptr<T>(ncol()*ldl);
      int* lperm = cdata_.get_lperm(j_);
      // Copy into lwork with permutation
      for(int j=0; j<ncol(); ++j) {
         int c = lperm[j];
         for(int i=0; i<nrow(); ++i)
            lwork[j*ldl+i] = aval_[c*lda_+i];
      }
      // Copy back again
      for(int j=0; j<ncol(); ++j)
      for(int i=0; i<nrow(); ++i)
         aval_[j*lda_+i] = lwork[j*ldl+i];
   }

   /** \brief Restore the entire block from backup.
    *  \details Intended for recovery from failed Cholesky-like factorization.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void full_restore(Backup& backup) {
      backup.restore_part(i_, j_, 0, 0, aval_, lda_);
   }

   /** \brief Restore any failed columns from backup.
    *  \details Storage associated with backup is released by this routine
    *           once we are done with it. This routine should only be called
    *           for blocks in the eliminated row/column.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    *  \param elim_col The block column we've just finished eliminating and
    *         wish to perform restores associated with.
    */
   template <typename Backup>
   void restore_if_required(Backup& backup, int elim_col) {
      if(i_ == elim_col && j_ == elim_col) { // In eliminated diagonal block
         if(cdata_[i_].nelim < ncol()) { // If there are failed pivots
            backup.restore_part_with_sym_perm(
                  i_, j_, cdata_[i_].nelim, cdata_.get_lperm(i_), aval_, lda_
                  );
         }
         // Release resources regardless, no longer required
         backup.release(i_, j_);
      }
      else if(i_ == elim_col) { // In eliminated row
         if(cdata_[i_].nelim < nrow()) // If there are failed pivots
            backup.restore_part(
                  i_, j_, cdata_[i_].nelim, cdata_[j_].nelim, aval_, lda_
                  );
         // Release resources regardless, no longer required
         backup.release(i_, j_);
      }
      else if(j_ == elim_col) { // In eliminated col
         if(cdata_[j_].nelim < ncol()) { // If there are failed pivots
            int rfrom = (i_ <= elim_col) ? cdata_[i_].nelim : 0;
            backup.restore_part(i_, j_, rfrom, cdata_[j_].nelim, aval_, lda_);
         }
         // Release resources regardless, no longer required
         backup.release(i_, j_);
      }
   }

   /** \brief Factorize diagonal block.
    *  \details Performs the in-place factorization
    *           \f[ A_{ii} = P L_{ii} D_i L_{ii}^T P^T. \f]
    *           The mechanism to do so varies:
    *           - If block_size != BLOCK_SIZE then recurse with a call to
    *             LDLT::factor() using BLOCK_SIZE as the new block size.
    *           - Otherwise, if the block is a full block of size BLOCK_SIZE,
    *             call block_ldlt().
    *           - Otherwise, if the block is not full, call ldlt_tpp_factor().
    *           Note that two permutations are maintained, the user permutation
    *           perm, and the local permutation lperm obtained from
    *           ColumnData::get_lperm() that represents P above.
    *  \tparam Allocator allocator type to be used on recursion to
    *          LDLT::factor().
    *  \param next_elim Next variable to be eliminated, used to determine
    *         location in d to be used.
    *  \param perm User permutation: entries are permuted in same way as
    *         matrix columns.
    *  \param d pointer to global array for D.
    *  \param options user-supplied options
    *  \param work vector of thread-specific workspaces
    *  \param alloc allocator instance to be used on recursion to
    *         LDLT::factor().
    */
   template <typename Allocator>
   int factor(int next_elim, int* perm, T* d,
         struct cpu_factor_options const &options,
         std::vector<Workspace>& work, Allocator const& alloc) {
      if(i_ != j_)
         throw std::runtime_error("factor called on non-diagonal block!");
      int* lperm = cdata_.get_lperm(i_);
      for(int i=0; i<ncol(); i++)
         lperm[i] = i;
      cdata_[i_].d = &d[2*next_elim];
      if(block_size_ != INNER_BLOCK_SIZE) {
         // Recurse
         CopyBackup<T, Allocator> inner_backup(
               nrow(), ncol(), INNER_BLOCK_SIZE, alloc
               );
         bool const use_tasks = false; // Don't run in parallel at lower level
         bool const debug = false; // Don't print debug info for inner call
         cdata_[i_].nelim =
            LDLT<T, INNER_BLOCK_SIZE, CopyBackup<T,Allocator>,
                 use_tasks, debug, Allocator>
                ::factor(
                      nrow(), ncol(), lperm, aval_, lda_,
                      cdata_[i_].d, inner_backup, options, options.pivot_method,
                      INNER_BLOCK_SIZE, 0, nullptr, 0, work, alloc
                      );
         if(cdata_[i_].nelim < 0) return cdata_[i_].nelim;
         int* temp = work[omp_get_thread_num()].get_ptr<int>(ncol());
         int* blkperm = &perm[i_*block_size_];
         for(int i=0; i<ncol(); ++i)
            temp[i] = blkperm[lperm[i]];
         for(int i=0; i<ncol(); ++i)
            blkperm[i] = temp[i];
      } else { /* block_size == INNER_BLOCK_SIZE */
         // Call another routine for small block factorization
         if(ncol() < INNER_BLOCK_SIZE || !is_aligned(aval_)) {
            T* ld = work[omp_get_thread_num()].get_ptr<T>(2*INNER_BLOCK_SIZE);
            cdata_[i_].nelim = ldlt_tpp_factor(
                  nrow(), ncol(), lperm, aval_, lda_,
                  cdata_[i_].d, ld, INNER_BLOCK_SIZE, options.action,
                  options.u, options.small
                  );
            if(cdata_[i_].nelim < 0) return cdata_[i_].nelim;
            int* temp = work[omp_get_thread_num()].get_ptr<int>(ncol());
            int* blkperm = &perm[i_*INNER_BLOCK_SIZE];
            for(int i=0; i<ncol(); ++i)
               temp[i] = blkperm[lperm[i]];
            for(int i=0; i<ncol(); ++i)
               blkperm[i] = temp[i];
         } else {
            int* blkperm = &perm[i_*INNER_BLOCK_SIZE];
            T* ld = work[omp_get_thread_num()].get_ptr<T>(
                  INNER_BLOCK_SIZE*INNER_BLOCK_SIZE
                  );
            block_ldlt<T, INNER_BLOCK_SIZE>(
                  0, blkperm, aval_, lda_, cdata_[i_].d, ld, options.action,
                  options.u, options.small, lperm
                  );
            cdata_[i_].nelim = INNER_BLOCK_SIZE;
         }
      }
      return cdata_[i_].nelim;
   }

   /** \brief Apply pivots to this block and return number of pivots passing
    *         a posteori pivot test.
    *  \details If this block is below dblk, perform the operation
    *           \f[ L_{ij} = A_{ij} (D_j L_{jj})^{-T} \f]
    *           otherwise, if this block is to left of dblk, perform the
    *           operation
    *           \f[ L_{ij} = (D_i L_{ii})^{-1} A_{ij} \f]
    *           but only to uneliminated columns.
    *           After operation has completed, check a posteori pivoting
    *           condition \f$ l_{ij} < u^{-1} \f$ and return first column
    *           (block below dblk) or row (block left of dblk) in which
    *           it fails, or the total number of rows/columns otherwise.
    *  \param dblk The diagonal block to apply.
    *  \param u The pivot threshold for threshold test.
    *  \param small The drop tolerance for zero testing.
    *  \returns Number of successful pivots in this block.
    */
   int apply_pivot_app(Block const& dblk, T u, T small) {
      if(i_ == j_)
         throw std::runtime_error("apply_pivot called on diagonal block!");
      if(i_ == dblk.i_) { // Apply within row (ApplyT)
         apply_pivot<OP_T>(
               cdata_[i_].nelim, ncol(), cdata_[j_].nelim, dblk.aval_,
               cdata_[i_].d, small, aval_, lda_
               );
         return check_threshold<OP_T>(
               0, cdata_[i_].nelim, cdata_[j_].nelim, ncol(), u, aval_, lda_
               );
      } else if(j_ == dblk.j_) { // Apply within column (ApplyN)
         apply_pivot<OP_N>(
               nrow(), cdata_[j_].nelim, 0, dblk.aval_,
               cdata_[j_].d, small, aval_, lda_
               );
         return check_threshold<OP_N>(
               0, nrow(), 0, cdata_[j_].nelim, u, aval_, lda_
               );
      } else {
         throw std::runtime_error("apply_pivot called on block outside eliminated column");
      }
   }

   /** \brief Perform update of this block.
    *  \details Perform an update using the outer product of the supplied
    *           blocks:
    *           \f[ A_{ij} = A_{ij} - L_{ik} D_k L_{jk}^T \f]
    *           If this block is in the last "real" block column, optionally
    *           apply the same update to the supplied part of the contribution
    *           block that maps on to the "missing" part of this block.
    *  \param isrc The Block L_{ik}.
    *  \param jsrc The Block L_{jk}.
    *  \param work Thread-specific workspace.
    *  \param beta Global coefficient of original \f$ U_{ij} \f$ value.
    *         See form_contrib() for details.
    *  \param upd Optional pointer to \f$ U_{ij} \f$ values to be updated.
    *         If this is null, no such update is performed.
    *  \param ldupd Leading dimension of upd.
    */
   void update(Block const& isrc, Block const& jsrc, Workspace& work,
         double beta=1.0, T* upd=nullptr, int ldupd=0) {
      if(isrc.i_ == i_ && isrc.j_ == jsrc.j_) {
         // Update to right of elim column (UpdateN)
         int elim_col = isrc.j_;
         if(cdata_[elim_col].nelim == 0) return; // nothing to do
         int rfrom = (i_ <= elim_col) ? cdata_[i_].nelim : 0;
         int cfrom = (j_ <= elim_col) ? cdata_[j_].nelim : 0;
         int ldld = align_lda<T>(block_size_);
         T* ld = work.get_ptr<T>(block_size_*ldld);
         // NB: we use ld[rfrom] below so alignment matches that of aval[rfrom]
         calcLD<OP_N>(
               nrow()-rfrom, cdata_[elim_col].nelim, &isrc.aval_[rfrom],
               lda_, cdata_[elim_col].d, &ld[rfrom], ldld
               );
         host_gemm(
               OP_N, OP_T, nrow()-rfrom, ncol()-cfrom, cdata_[elim_col].nelim,
               -1.0, &ld[rfrom], ldld, &jsrc.aval_[cfrom], lda_,
               1.0, &aval_[cfrom*lda_+rfrom], lda_
               );
         if(upd && j_==calc_nblk(n_,block_size_)-1) {
            // Handle fractional part of upd that "belongs" to this block
            int u_ncol = std::min(block_size_-ncol(), m_-n_); // ncol for upd
            beta = (cdata_[elim_col].first_elim) ? beta : 1.0; // user beta only on first update
            if(i_ == j_) {
               // diagonal block
               host_gemm(
                     OP_N, OP_T, u_ncol, u_ncol, cdata_[elim_col].nelim,
                     -1.0, &ld[ncol()], ldld,
                     &jsrc.aval_[ncol()], lda_,
                     beta, upd, ldupd
                     );
            } else {
               // off-diagonal block
               T* upd_ij =
                  &upd[(i_-calc_nblk(n_,block_size_))*block_size_+u_ncol];
               host_gemm(
                     OP_N, OP_T, nrow(), u_ncol, cdata_[elim_col].nelim,
                     -1.0, &ld[rfrom], ldld, &jsrc.aval_[ncol()], lda_,
                     beta, upd_ij, ldupd
                     );
            }
         }
      } else {
         // Update to left of elim column (UpdateT)
         int elim_col = jsrc.i_;
         if(cdata_[elim_col].nelim == 0) return; // nothing to do
         int rfrom = (i_ <= elim_col) ? cdata_[i_].nelim : 0;
         int cfrom = (j_ <= elim_col) ? cdata_[j_].nelim : 0;
         int ldld = align_lda<T>(block_size_);
         T* ld = work.get_ptr<T>(block_size_*ldld);
         // NB: we use ld[rfrom] below so alignment matches that of aval[rfrom]
         if(isrc.j_==elim_col) {
            calcLD<OP_N>(
                  nrow()-rfrom, cdata_[elim_col].nelim,
                  &isrc.aval_[rfrom], lda_,
                  cdata_[elim_col].d, &ld[rfrom], ldld
                  );
         } else {
            calcLD<OP_T>(
                  nrow()-rfrom, cdata_[elim_col].nelim, &
                  isrc.aval_[rfrom*lda_], lda_,
                  cdata_[elim_col].d, &ld[rfrom], ldld
                  );
         }
         host_gemm(
               OP_N, OP_N, nrow()-rfrom, ncol()-cfrom, cdata_[elim_col].nelim,
               -1.0, &ld[rfrom], ldld, &jsrc.aval_[cfrom*lda_], lda_,
               1.0, &aval_[cfrom*lda_+rfrom], lda_
               );
      }
   }

   /** \brief Update this block as part of contribution block.
    *  \details Treat this block's coordinates as beloning to the trailing
    *           matrix (contribution block/generated elment) and perform an
    *           update using the outer product of the supplied blocks.
    *           \f[ U_{ij} = U_{ij} - L_{ik} D_k L_{jk}^T \f]
    *           If this is the first update to \f$ U_{ij} \f$, the existing
    *           values are multipled by a user-supplied coefficient
    *           \f$ \beta \f$.
    *  \param isrc the Block L_{ik}.
    *  \param jsrc the Block L_{jk}.
    *  \param work this thread's workspace.
    *  \param beta Global coefficient of original \f$ U_{ij} \f$ value.
    *  \param upd_ij pointer to \f$ U_{ij} \f$ values to be updated.
    *  \param ldupd leading dimension of upd_ij.
    */
   void form_contrib(Block const& isrc, Block const& jsrc, Workspace& work, double beta, T* upd_ij, int ldupd) {
      int elim_col = isrc.j_;
      int ldld = align_lda<T>(block_size_);
      T* ld = work.get_ptr<T>(block_size_*ldld);
      calcLD<OP_N>(
            nrow(), cdata_[elim_col].nelim, isrc.aval_, lda_,
            cdata_[elim_col].d, ld, ldld
            );
      // User-supplied beta only on first update; otherwise 1.0
      T rbeta = (cdata_[elim_col].first_elim) ? beta : 1.0;
      int blkn = get_nrow(j_); // nrow not ncol as we're on contrib
      host_gemm(
            OP_N, OP_T, nrow(), blkn, cdata_[elim_col].nelim,
            -1.0, ld, ldld, jsrc.aval_, lda_,
            rbeta, upd_ij, ldupd
            );
   }

   /** \brief Returns true if block contains NaNs or Infs (debug only).
    *  \param elim_col if supplied, the block column currently being considered
    *         for elimination. Entries in that block row/column marked as
    *         failed are ignored.
    */
   bool isnan(int elim_col=-1) const {
      int m = (i_==elim_col) ? cdata_[i_].get_npass() : nrow();
      int n = (j_==elim_col) ? cdata_[j_].get_npass() : ncol();
      for(int j=0; j<n; ++j)
      for(int i=((i_==j_)?j:0); i<m; ++i) {
         if(std::isnan(aval_[j*lda_+i])) {
            printf("%d, %d is nan\n", i, j);
            return true;
         }
         if(!std::isfinite(aval_[j*lda_+i])) {
            printf("%d, %d is inf\n", i, j);
            return true;
         }
      }
      return false;
   }

   /** \brief Prints block (debug only) */
   void print() const {
      printf("Block %d, %d (%d x %d):\n", i_, j_, nrow(), ncol());
      for(int i=0; i<nrow(); ++i) {
         printf("%d:", i);
         for(int j=0; j<ncol(); ++j)
            printf(" %e", aval_[j*lda_+i]);
         printf("\n");
      }
   }

   /** \brief return number of rows in this block */
   int nrow() const { return get_nrow(i_); }
   /** \brief return number of columns in this block */
   int ncol() const { return get_ncol(j_); }
private:
   /** \brief return number of columns in given block column */
   inline int get_ncol(int blk) const {
      return calc_blkn(blk, n_, block_size_);
   }
   /** \brief return number of rows in given block row */
   inline int get_nrow(int blk) const {
      return calc_blkn(blk, m_, block_size_);
   }

   int const i_; ///< block's row
   int const j_; ///< block's column
   int const m_; ///< number of rows in matrix
   int const n_; ///< number of columns in matrix
   int const lda_; ///< leading dimension of underlying storage
   int const block_size_; ///< block size
   ColumnData<T,IntAlloc>& cdata_; ///< global column data array
   T* aval_; ///< pointer to underlying matrix storage
};

/** \brief Grouping of assorted functions for LDL^T factorization that share
 *         template paramters.
 *  \tparam T underlying datatype, e.g. double
 *  \tparam BLOCK_SIZE inner block size for factorization, must be a multiple
 *          of vector length.
 *  \tparam Backup class to be used for handling block backups,
 *          e.g. PoolBackup or CopyBackup.
 *  \tparam use_tasks enable use of OpenMP tasks if true (used to serialise
 *          internal call for small block sizes).
 *  \tparam debug enable debug output.
 *  \tparam Allocator allocator to use for internal memory allocations
 */
template<typename T,
         int BLOCK_SIZE,
         typename Backup,
         bool use_tasks,
         bool debug,
         typename Allocator
         >
class LDLT {
   /// \{
   typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
   typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T> TAlloc;
   /// \}
private:
   /** Performs LDL^T factorization with block pivoting. Detects failure
    *  and aborts only column if an a posteori pivot test fails. */
   static
   int run_elim_pivoted(int const m, int const n, int* perm, T* a,
         int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
         struct cpu_factor_options const& options, int const block_size,
         T const beta, T* upd, int const ldupd, std::vector<Workspace>& work,
         Allocator const& alloc, int const from_blk=0) {
      typedef Block<T, BLOCK_SIZE, IntAlloc> BlockSpec;

      int const nblk = calc_nblk(n, block_size);
      int const mblk = calc_nblk(m, block_size);
      //printf("ENTRY PIV %d %d vis %d %d %d\n", m, n, mblk, nblk, block_size);

      /* Setup */
      int next_elim = from_blk*block_size;
      int flag;
      #pragma omp atomic write
      flag = 0;

      /* Inner loop - iterate over block columns */
      bool abort;
      #pragma omp atomic write
      abort = false;

      #pragma omp taskgroup
      for (int blk = from_blk; blk < nblk; blk++) {
         /*if(debug) {
            printf("Bcol %d:\n", blk);
            print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
         }*/

         // Factor diagonal: depend on perm[blk*block_size] as we init npass
         #pragma omp task default(none)                           \
            firstprivate(blk)                                     \
            shared(a, abort, perm, backup, cdata, next_elim, d,   \
                   options, work, alloc, flag)                    \
            depend(inout: a[blk*block_size*lda+blk*block_size:1]) \
            depend(inout: perm[blk*block_size:1])
         {
           bool my_abort;
           #pragma omp atomic read
           my_abort = abort;
           if (!my_abort) {
             try {
               #pragma omp cancellation point taskgroup
#ifdef PROFILE
               Profile::Task task("TA_LDLT_DIAG");
#endif
               if (debug) printf("Factor(%d)\n", blk);
               BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
               // Store a copy for recovery in case of a failed column
               dblk.backup(backup);
               // Perform actual factorization
               int nelim = dblk.template factor<Allocator>(next_elim, perm, d, options, work, alloc);
               if (nelim < 0) {
                 #pragma omp atomic write
                 flag = nelim;
#ifdef _OPENMP
                 #pragma omp atomic write
                 abort = true;
                 #pragma omp cancel taskgroup
#else
                 return flag;
#endif /* _OPENMP */
               } else {
                 // Init threshold check (non locking => task dependencies)
                 cdata[blk].init_passed(nelim);
               }
#ifdef PROFILE
               task.done();
#endif
            } catch(std::bad_alloc const&) {
               #pragma omp atomic write
               flag = Flag::ERROR_ALLOCATION;
#ifdef _OPENMP
               #pragma omp atomic write
               abort = true;
               #pragma omp cancel taskgroup
#else
               return flag;
#endif /* _OPENMP */
            } catch(SingularError const&) {
               #pragma omp atomic write
               flag = Flag::ERROR_SINGULAR;
#ifdef _OPENMP
               #pragma omp atomic write
               abort = true;
               #pragma omp cancel taskgroup
#else
               return flag;
#endif /* _OPENMP */
            }
         } } /* task/abort */
         
         // Loop over off-diagonal blocks applying pivot
         for(int jblk = 0; jblk < blk; jblk++) {
            #pragma omp task default(none)                            \
               firstprivate(blk, jblk)                                \
               shared(a, abort, backup, cdata, options)               \
               depend(in: a[blk*block_size*lda+blk*block_size:1])     \
               depend(inout: a[jblk*block_size*lda+blk*block_size:1]) \
               depend(in: perm[blk*block_size:1])
            {
              bool my_abort;
              #pragma omp atomic read
              my_abort = abort;
              if (!my_abort) {
                #pragma omp cancellation point taskgroup
#ifdef PROFILE
                Profile::Task task("TA_LDLT_APPLY");
#endif
                if (debug) printf("ApplyT(%d,%d)\n", blk, jblk);
                BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
                BlockSpec cblk(blk, jblk, m, n, cdata, a, lda, block_size);
                // Apply row permutation from factorization of dblk and in
                // the process, store a (permuted) copy for recovery in case of
                // a failed column
                cblk.apply_rperm_and_backup(backup);
                // Perform elimination and determine number of rows in block
                // passing a posteori threshold pivot test
                int blkpass = cblk.apply_pivot_app(dblk, options.u, options.small);
                // Update column's passed pivot count
                cdata[blk].update_passed(blkpass);
#ifdef PROFILE
                task.done();
#endif
            } } /* task/abort */
         }
         for (int iblk = blk + 1; iblk < mblk; iblk++) {
            #pragma omp task default(none)                            \
               firstprivate(blk, iblk)                                \
               shared(a, abort, backup, cdata, options)               \
               depend(in: a[blk*block_size*lda+blk*block_size:1])     \
               depend(inout: a[blk*block_size*lda+iblk*block_size:1]) \
               depend(in: perm[blk*block_size:1])
            {
              bool my_abort;
              #pragma omp atomic read
              my_abort = abort;
              if (!my_abort) {
                #pragma omp cancellation point taskgroup
#ifdef PROFILE
                Profile::Task task("TA_LDLT_APPLY");
#endif
                if (debug) printf("ApplyN(%d,%d)\n", iblk, blk);
                BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
                BlockSpec rblk(iblk, blk, m, n, cdata, a, lda, block_size);
                // Apply column permutation from factorization of dblk and in
                // the process, store a (permuted) copy for recovery in case of
                // a failed column
                rblk.apply_cperm_and_backup(backup);
                // Perform elimination and determine number of rows in block
                // passing a posteori threshold pivot test
                int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
                // Update column's passed pivot count
                cdata[blk].update_passed(blkpass);
#ifdef PROFILE
                task.done();
#endif
            } } /* task/abort */
         }

         // Adjust column once all applys have finished and we know final
         // number of passed columns.
         #pragma omp task default(none)           \
            firstprivate(blk)                     \
            shared(abort, cdata, next_elim)       \
            depend(inout: perm[blk*block_size:1])
         {
           bool my_abort;
           #pragma omp atomic read
           my_abort = abort;
           if (!my_abort) {
             #pragma omp cancellation point taskgroup
#ifdef PROFILE
             Profile::Task task("TA_LDLT_ADJUST");
#endif
             if (debug) printf("Adjust(%d)\n", blk);
             cdata[blk].adjust(next_elim);
#ifdef PROFILE
             task.done();
#endif
         } } /* task/abort */

         // Update uneliminated columns
         for (int jblk = 0; jblk < blk; jblk++) {
            for (int iblk = jblk; iblk < mblk; iblk++) {
               // Calculate block index we depend on for i
               // (we only work with lower half of matrix)
               int adep_idx = (blk < iblk) ? blk*block_size*lda + iblk*block_size
                                           : iblk*block_size*lda + blk*block_size;
               #pragma omp task default(none)                             \
                  firstprivate(blk, iblk, jblk)                           \
                  shared(a, abort, cdata, backup, work)                   \
                  depend(inout: a[jblk*block_size*lda+iblk*block_size:1]) \
                  depend(in: perm[blk*block_size:1])                      \
                  depend(in: a[jblk*block_size*lda+blk*block_size:1])     \
                  depend(in: a[adep_idx:1])
               {
                 bool my_abort;
                 #pragma omp atomic read
                 my_abort = abort;
                 if (!my_abort) {
                  #pragma omp cancellation point taskgroup
#ifdef PROFILE
                  Profile::Task task("TA_LDLT_UPDA");
#endif
                  if (debug) printf("UpdateT(%d,%d,%d)\n", iblk, jblk, blk);
                  int thread_num = omp_get_thread_num();
                  BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;
                  BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, a, lda,
                        block_size);
                  BlockSpec jsrc(blk, jblk, m, n, cdata, a, lda, block_size);
                  // If we're on the block row we've just eliminated, restore
                  // any failed rows and release resources storing backup
                  ublk.restore_if_required(backup, blk);
                  // Perform actual update
                  ublk.update(isrc, jsrc, work[thread_num]);
#ifdef PROFILE
                  task.done();
#endif
               } } /* task/abort */
            }
         }
         for(int jblk = blk; jblk < nblk; jblk++) {
            for(int iblk = jblk; iblk < mblk; iblk++) {
               #pragma omp task default(none)                             \
                  firstprivate(blk, iblk, jblk)                           \
                  shared(a, abort, cdata, backup, work, upd)              \
                  depend(inout: a[jblk*block_size*lda+iblk*block_size:1]) \
                  depend(in: perm[blk*block_size:1])                      \
                  depend(in: a[blk*block_size*lda+iblk*block_size:1])     \
                  depend(in: a[blk*block_size*lda+jblk*block_size:1])
               {
                 bool my_abort;
                 #pragma omp atomic read
                 my_abort = abort;
                 if (!my_abort) {
                   #pragma omp cancellation point taskgroup
#ifdef PROFILE
                   Profile::Task task("TA_LDLT_UPDA");
#endif
                   if (debug) printf("UpdateN(%d,%d,%d)\n", iblk, jblk, blk);
                   int thread_num = omp_get_thread_num();
                   BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                   BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
                   BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
                   // If we're on the block col we've just eliminated, restore
                   // any failed cols and release resources storing backup
                   ublk.restore_if_required(backup, blk);
                   // Perform actual update
                   ublk.update(isrc, jsrc, work[thread_num], beta, upd, ldupd);
#ifdef PROFILE
                   task.done();
#endif
               } } /* task/abort */
            }
         }

         // Handle update to contribution block, if required
         if (upd && (mblk > nblk)) {
            int uoffset = std::min(nblk*block_size, m) - n;
            T *upd2 = &upd[uoffset*(ldupd+1)];
            for(int jblk = nblk; jblk < mblk; ++jblk)
              for(int iblk = jblk; iblk < mblk; ++iblk) {
                T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + (iblk-nblk)*block_size];
                #pragma omp task default(none)                        \
                  firstprivate(iblk, jblk, blk, upd_ij)               \
                  shared(a, abort, upd2, cdata, work)                 \
                  depend(inout: upd_ij[0:1])                          \
                  depend(in: perm[blk*block_size:1])                  \
                  depend(in: a[blk*block_size*lda+iblk*block_size:1]) \
                  depend(in: a[blk*block_size*lda+jblk*block_size:1])
                {
                  bool my_abort;
                  #pragma omp atomic read
                  my_abort = abort;
                  if (!my_abort) {
                    #pragma omp cancellation point taskgroup
#ifdef PROFILE
                    Profile::Task task("TA_LDLT_UPDC");
#endif
                    if (debug) printf("FormContrib(%d,%d,%d)\n", iblk, jblk, blk);
                    int thread_num = omp_get_thread_num();
                    BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                    BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
                    BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
                    ublk.form_contrib(isrc, jsrc, work[thread_num], beta, upd_ij, ldupd);
#ifdef PROFILE
                    task.done();
#endif
                  } } /* task/abort */
              }
         }
      } // taskgroup and for
      int my_flag;
      #pragma omp atomic read
      my_flag = flag;
      if (my_flag < 0) return my_flag; // Error

      /*if(debug) {
         printf("PostElim:\n");
         print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
      }*/

      return next_elim;
   }

   /** Performs LDL^T factorization with block pivoting. Detects failure
    *  and aborts only column if an a posteori pivot test fails.
    *  Serial version without tasks. */
   static
   int run_elim_pivoted_notasks(int const m, int const n, int* perm, T* a,
         int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
         struct cpu_factor_options const& options, int const block_size,
         T const beta, T* upd, int const ldupd, std::vector<Workspace>& work,
         Allocator const& alloc, int const from_blk=0) {
      typedef Block<T, BLOCK_SIZE, IntAlloc> BlockSpec;

      int const nblk = calc_nblk(n, block_size);
      int const mblk = calc_nblk(m, block_size);
      //printf("ENTRY PIV %d %d vis %d %d %d\n", m, n, mblk, nblk, block_size);

      /* Setup */
      int next_elim = from_blk*block_size;

      /* Inner loop - iterate over block columns */
      try {
         for(int blk=from_blk; blk<nblk; blk++) {
            /*if(debug) {
               printf("Bcol %d:\n", blk);
               print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
            }*/

            // Factor diagonal: depend on perm[blk*block_size] as we init npass
            {
               if(debug) printf("Factor(%d)\n", blk);
               BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
               // Store a copy for recovery in case of a failed column
               dblk.backup(backup);
               // Perform actual factorization
               int nelim = dblk.template factor<Allocator>(
                     next_elim, perm, d, options, work, alloc
                     );
               if(nelim<0) return nelim;
               // Init threshold check (non locking => task dependencies)
               cdata[blk].init_passed(nelim);
            }
            
            // Loop over off-diagonal blocks applying pivot
            for(int jblk=0; jblk<blk; jblk++) {
               if(debug) printf("ApplyT(%d,%d)\n", blk, jblk);
               BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
               BlockSpec cblk(blk, jblk, m, n, cdata, a, lda, block_size);
               // Apply row permutation from factorization of dblk and in
               // the process, store a (permuted) copy for recovery in case of
               // a failed column
               cblk.apply_rperm_and_backup(backup);
               // Perform elimination and determine number of rows in block
               // passing a posteori threshold pivot test
               int blkpass = cblk.apply_pivot_app(
                     dblk, options.u, options.small
                     );
               // Update column's passed pivot count
               cdata[blk].update_passed(blkpass);
            }
            for(int iblk=blk+1; iblk<mblk; iblk++) {
               if(debug) printf("ApplyN(%d,%d)\n", iblk, blk);
               BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
               BlockSpec rblk(iblk, blk, m, n, cdata, a, lda, block_size);
               // Apply column permutation from factorization of dblk and in
               // the process, store a (permuted) copy for recovery in case of
               // a failed column
               rblk.apply_cperm_and_backup(backup);
               // Perform elimination and determine number of rows in block
               // passing a posteori threshold pivot test
               int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
               // Update column's passed pivot count
               cdata[blk].update_passed(blkpass);
            }

            // Adjust column once all applys have finished and we know final
            // number of passed columns.
            if(debug) printf("Adjust(%d)\n", blk);
            cdata[blk].adjust(next_elim);

            // Update uneliminated columns
            for(int jblk=0; jblk<blk; jblk++) {
               for(int iblk=jblk; iblk<mblk; iblk++) {
                  if(debug) printf("UpdateT(%d,%d,%d)\n", iblk, jblk, blk);
                  int thread_num = omp_get_thread_num();
                  BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;
                  BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, a, lda,
                        block_size);
                  BlockSpec jsrc(blk, jblk, m, n, cdata, a, lda, block_size);
                  // If we're on the block row we've just eliminated, restore
                  // any failed rows and release resources storing backup
                  ublk.restore_if_required(backup, blk);
                  // Perform actual update
                  ublk.update(isrc, jsrc, work[thread_num]);
               }
            }
            for(int jblk=blk; jblk<nblk; jblk++) {
               for(int iblk=jblk; iblk<mblk; iblk++) {
                  if(debug) printf("UpdateN(%d,%d,%d)\n", iblk, jblk, blk);
                  int thread_num = omp_get_thread_num();
                  BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
                  BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
                  // If we're on the block col we've just eliminated, restore
                  // any failed cols and release resources storing backup
                  ublk.restore_if_required(backup, blk);
                  // Perform actual update
                  ublk.update(isrc, jsrc, work[thread_num],
                        beta, upd, ldupd);
               }
            }

            // Handle update to contribution block, if required
            if(upd && mblk>nblk) {
               int uoffset = std::min(nblk*block_size, m) - n;
               T *upd2 = &upd[uoffset*(ldupd+1)];
               for(int jblk=nblk; jblk<mblk; ++jblk)
               for(int iblk=jblk; iblk<mblk; ++iblk) {
                  T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + 
                                    (iblk-nblk)*block_size];
                  {
                     if(debug) printf("FormContrib(%d,%d,%d)\n", iblk, jblk, blk);
                     int thread_num = omp_get_thread_num();
                     BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                     BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
                     BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
                     ublk.form_contrib(
                           isrc, jsrc, work[thread_num], beta, upd_ij, ldupd
                           );
                  }
               }
            }
         }
      } catch(std::bad_alloc const&) {
         return Flag::ERROR_ALLOCATION;
      } catch(SingularError const&) {
         return Flag::ERROR_SINGULAR;
      }

      /*if(debug) {
         printf("PostElim:\n");
         print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
      }*/

      return next_elim;
   }

   /** Performs LDL^T factorization assuming everything works. Detects failure
    *  and aborts entire thing if a posteori pivot test fails. */
   static
   int run_elim_unpivoted(int const m, int const n, int* perm, T* a,
         int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
         int* up_to_date, struct cpu_factor_options const& options,
         int const block_size, T const beta, T* upd, int const ldupd,
         std::vector<Workspace>& work, Allocator const& alloc) {
      typedef Block<T, BLOCK_SIZE, IntAlloc> BlockSpec;

      int const nblk = calc_nblk(n, block_size);
      int const mblk = calc_nblk(m, block_size);
      //printf("ENTRY %d %d vis %d %d %d\n", m, n, mblk, nblk, block_size);

      /* Setup */
      int next_elim = 0;
      int flag;
      #pragma omp atomic write
      flag = 0;

      /* Inner loop - iterate over block columns */
      bool abort;
      #pragma omp atomic write
      abort = false;
      #pragma omp taskgroup
      for(int blk = 0; blk < nblk; blk++) {
         /*if(debug) {
            printf("Bcol %d:\n", blk);
            print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
         }*/

         // Factor diagonal
         #pragma omp task default(none)                         \
            firstprivate(blk)                                   \
            shared(a, abort, perm, backup, cdata, next_elim, d, \
                   options, work, alloc, up_to_date, flag)      \
            depend(inout: a[blk*block_size*lda+blk*block_size:1])
         {
           bool my_abort;
           #pragma omp atomic read
           my_abort = abort;
           if (!my_abort) {
             try {
               #pragma omp cancellation point taskgroup
#ifdef PROFILE
               Profile::Task task("TA_LDLT_DIAG");
#endif
               if(debug) printf("Factor(%d)\n", blk);
               BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
               // On first access to this block, store copy in case of failure
               if (blk == 0) dblk.backup(backup);
               // Record block state as assuming we've done up to col blk
               up_to_date[blk*mblk+blk] = blk;
               // Perform actual factorization
               int nelim = dblk.template factor<Allocator>(next_elim, perm, d, options, work, alloc);
               if (nelim < get_ncol(blk, n, block_size)) {
                 cdata[blk].init_passed(0); // diagonal block has NOT passed
#ifdef _OPENMP
                 #pragma omp atomic write
                 abort = true;
                 #pragma omp cancel taskgroup
#else
                 return cdata.calc_nelim(m);
#endif /* _OPENMP */
               } else {
                  cdata[blk].first_elim = (blk==0);
                  cdata[blk].init_passed(1); // diagonal block has passed
                  next_elim += nelim; // we're assuming everything works
               }
#ifdef PROFILE
               task.done();
#endif
            } catch(std::bad_alloc const&) {
               #pragma omp atomic write
               flag = Flag::ERROR_ALLOCATION;
#ifdef _OPENMP
               #pragma omp atomic write
               abort = true;
               #pragma omp cancel taskgroup
#else
               return flag;
#endif /* _OPENMP */
            } catch(SingularError const&) {
               #pragma omp atomic write
               flag = Flag::ERROR_SINGULAR;
#ifdef _OPENMP
               #pragma omp atomic write
               abort = true;
               #pragma omp cancel taskgroup
#else
               return flag;
#endif /* _OPENMP */
            }
         } } /* task/abort */
         
         // Loop over off-diagonal blocks applying pivot
         for (int jblk = 0; jblk < blk; jblk++) {
            #pragma omp task default(none)                                \
               firstprivate(blk, jblk)                                    \
               shared(a, abort, backup, cdata, options, work, up_to_date) \
               depend(in: a[blk*block_size*lda+blk*block_size:1])         \
               depend(inout: a[jblk*block_size*lda+blk*block_size:1])
            {
              bool my_abort;
              #pragma omp atomic read
              my_abort = abort;
              if (!my_abort) {
                #pragma omp cancellation point taskgroup
#ifdef PROFILE
                Profile::Task task("TA_LDLT_APPLY");
#endif
                if (debug) printf("ApplyT(%d,%d)\n", blk, jblk);
                int thread_num = omp_get_thread_num();
                BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
                BlockSpec cblk(blk, jblk, m, n, cdata, a, lda, block_size);
                // Record block state as assuming we've done up to col blk
                up_to_date[jblk*mblk+blk] = blk;
                // Apply row permutation from factorization of dblk
                cblk.apply_rperm(work[thread_num]);
                // NB: no actual application of pivot must be done, as we are
                // assuming everything has passed...
#ifdef PROFILE
                task.done();
#endif
            } } /* task/abort */
         }
         for (int iblk = blk+1; iblk < mblk; iblk++) {
            #pragma omp task default(none)                                \
               firstprivate(blk, iblk)                                    \
               shared(a, abort, backup, cdata, options, work, up_to_date) \
               depend(in: a[blk*block_size*lda+blk*block_size:1])         \
               depend(inout: a[blk*block_size*lda+iblk*block_size:1])
            {
              bool my_abort;
              #pragma omp atomic read
              my_abort = abort;
              if (!my_abort) {
                #pragma omp cancellation point taskgroup
#ifdef PROFILE
                Profile::Task task("TA_LDLT_APPLY");
#endif
                if (debug) printf("ApplyN(%d,%d)\n", iblk, blk);
                int thread_num = omp_get_thread_num();
                BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
                BlockSpec rblk(iblk, blk, m, n, cdata, a, lda, block_size);
                // On first access to this block, store copy in case of failure
                if (blk==0) rblk.backup(backup);
                // Record block state as assuming we've done up to col blk
                up_to_date[blk*mblk+iblk] = blk;
                // Apply column permutation from factorization of dblk
                rblk.apply_cperm(work[thread_num]);
                // Perform elimination and determine number of rows in block
                // passing a posteori threshold pivot test
                int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
                // Update column's passed pivot count
                if (cdata[blk].test_fail(blkpass)) {
#ifdef _OPENMP
                  #pragma omp atomic write
                  abort = true;
                  #pragma omp cancel taskgroup
#else
                  return cdata.calc_nelim(m);
#endif /* _OPENMP */
               }
#ifdef PROFILE
                task.done();
#endif
            } } /* task/abort */
         }

         // Update uneliminated columns
         // Column blk only needed if upd is present
         int jsa = (upd) ? blk : blk + 1;
         for(int jblk = jsa; jblk < nblk; jblk++) {
            for(int iblk = jblk; iblk < mblk; iblk++) {
               #pragma omp task default(none)                             \
                  firstprivate(blk, iblk, jblk)                           \
                  shared(a, abort, cdata, backup, work, upd, up_to_date)  \
                  depend(inout: a[jblk*block_size*lda+iblk*block_size:1]) \
                  depend(in: a[blk*block_size*lda+iblk*block_size:1])     \
                  depend(in: a[blk*block_size*lda+jblk*block_size:1])
               {
                 bool my_abort;
                 #pragma omp atomic read
                 my_abort = abort;
                 if (!my_abort) {
                   #pragma omp cancellation point taskgroup
#ifdef PROFILE
                   Profile::Task task("TA_LDLT_UPDA");
#endif
                   if (debug) printf("UpdateN(%d,%d,%d)\n", iblk, jblk, blk);
                   int thread_num = omp_get_thread_num();
                   BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                   BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
                   BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
                   // On first access to this block, store copy in case of fail
                   if ((blk == 0) && (jblk != blk)) ublk.backup(backup);
                   // Record block state as assuming we've done up to col blk
                   up_to_date[jblk*mblk+iblk] = blk;
                   // Actual update
                   ublk.update(isrc, jsrc, work[thread_num], beta, upd, ldupd);
#ifdef PROFILE
                   task.done();
#endif
               } } /* task/abort */
            }
         }

         // Handle update to contribution block, if required
         if (upd && (mblk > nblk)) {
            int uoffset = std::min(nblk*block_size, m) - n;
            T *upd2 = &upd[uoffset*(ldupd+1)];
            for(int jblk = nblk; jblk < mblk; ++jblk)
              for(int iblk = jblk; iblk < mblk; ++iblk) {
                T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + (iblk-nblk)*block_size];
                #pragma omp task default(none)                        \
                  firstprivate(iblk, jblk, blk, upd_ij)               \
                  shared(a, abort, upd2, cdata, work, up_to_date)     \
                  depend(inout: upd_ij[0:1])                          \
                  depend(in: a[blk*block_size*lda+iblk*block_size:1]) \
                  depend(in: a[blk*block_size*lda+jblk*block_size:1])
               {
                 bool my_abort;
                 #pragma omp atomic read
                 my_abort = abort;
                 if (!my_abort) {
                   #pragma omp cancellation point taskgroup
#ifdef PROFILE
                   Profile::Task task("TA_LDLT_UPDC");
#endif
                   if (debug) printf("FormContrib(%d,%d,%d)\n", iblk, jblk, blk);
                   int thread_num = omp_get_thread_num();
                   BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                   BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
                   BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
                   // Record block state as assuming we've done up to col blk
                   up_to_date[jblk*mblk+iblk] = blk;
                   // Perform update
                   ublk.form_contrib(isrc, jsrc, work[thread_num], beta, upd_ij, ldupd);
#ifdef PROFILE
                   task.done();
#endif
               } } /* task/abort */
            }
         }
      } // taskgroup and for

      /*if(debug) {
         printf("PostElim:\n");
         print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
      }*/

      int my_flag;
      #pragma omp atomic read
      my_flag = flag;
      if (my_flag < 0) return my_flag;
      return cdata.calc_nelim(m);
   }

   /** Performs LDL^T factorization assuming everything works. Detects failure
    *  and aborts entire thing if a posteori pivot test fails. */
   static
   int run_elim_unpivoted_notasks(int const m, int const n, int* perm, T* a,
         int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
         int* up_to_date, struct cpu_factor_options const& options,
         int const block_size, T const beta, T* upd, int const ldupd,
         std::vector<Workspace>& work, Allocator const& alloc) {
      typedef Block<T, BLOCK_SIZE, IntAlloc> BlockSpec;

      int const nblk = calc_nblk(n, block_size);
      int const mblk = calc_nblk(m, block_size);
      //printf("ENTRY %d %d vis %d %d %d\n", m, n, mblk, nblk, block_size);

      /* Setup */
      int next_elim = 0;

      /* Inner loop - iterate over block columns */
      for(int blk=0; blk<nblk; blk++) {
         /*if(debug) {
            printf("Bcol %d:\n", blk);
            print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
         }*/

         // Factor diagonal
         try {
            if(debug) printf("Factor(%d)\n", blk);
            BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
            // On first access to this block, store copy in case of failure
            if(blk==0) dblk.backup(backup);
            // Record block state as assuming we've done up to col blk
            up_to_date[blk*mblk+blk] = blk;
            // Perform actual factorization
            int nelim = dblk.template factor<Allocator>(
                  next_elim, perm, d, options, work, alloc
                  );
            if(nelim < get_ncol(blk, n, block_size)) {
               cdata[blk].init_passed(0); // diagonal block has NOT passed
               return cdata.calc_nelim(m);
            } else {
               cdata[blk].first_elim = (blk==0);
               cdata[blk].init_passed(1); // diagonal block has passed
               next_elim += nelim; // we're assuming everything works
            }
         } catch(std::bad_alloc const&) {
            return Flag::ERROR_ALLOCATION;
         } catch(SingularError const&) {
            return Flag::ERROR_SINGULAR;
         }
         
         // Loop over off-diagonal blocks applying pivot
         for(int jblk=0; jblk<blk; jblk++) {
            if(debug) printf("ApplyT(%d,%d)\n", blk, jblk);
            int thread_num = omp_get_thread_num();
            BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
            BlockSpec cblk(blk, jblk, m, n, cdata, a, lda, block_size);
            // Record block state as assuming we've done up to col blk
            up_to_date[jblk*mblk+blk] = blk;
            // Apply row permutation from factorization of dblk
            cblk.apply_rperm(work[thread_num]);
            // NB: no actual application of pivot must be done, as we are
            // assuming everything has passed...
         }
         for(int iblk=blk+1; iblk<mblk; iblk++) {
            if(debug) printf("ApplyN(%d,%d)\n", iblk, blk);
            int thread_num = omp_get_thread_num();
            BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
            BlockSpec rblk(iblk, blk, m, n, cdata, a, lda, block_size);
            // On first access to this block, store copy in case of failure
            if(blk==0) rblk.backup(backup);
            // Record block state as assuming we've done up to col blk
            up_to_date[blk*mblk+iblk] = blk;
            // Apply column permutation from factorization of dblk
            rblk.apply_cperm(work[thread_num]);
            // Perform elimination and determine number of rows in block
            // passing a posteori threshold pivot test
            int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
            // Update column's passed pivot count
            if(cdata[blk].test_fail(blkpass))
               return cdata.calc_nelim(m);
         }

         // Update uneliminated columns
         // Column blk only needed if upd is present
         int jsa = (upd) ? blk : blk + 1;
         for(int jblk=jsa; jblk<nblk; jblk++) {
            for(int iblk=jblk; iblk<mblk; iblk++) {
               if(debug) printf("UpdateN(%d,%d,%d)\n", iblk, jblk, blk);
               int thread_num = omp_get_thread_num();
               BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
               BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
               BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
               // On first access to this block, store copy in case of fail
               if(blk==0 && jblk!=blk) ublk.backup(backup);
               // Record block state as assuming we've done up to col blk
               up_to_date[jblk*mblk+iblk] = blk;
               // Actual update
               ublk.update(isrc, jsrc, work[thread_num], beta, upd, ldupd);
            }
         }

         // Handle update to contribution block, if required
         if(upd && mblk>nblk) {
            int uoffset = std::min(nblk*block_size, m) - n;
            T *upd2 = &upd[uoffset*(ldupd+1)];
            for(int jblk=nblk; jblk<mblk; ++jblk)
            for(int iblk=jblk; iblk<mblk; ++iblk) {
            T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + 
                              (iblk-nblk)*block_size];
               if(debug) printf("FormContrib(%d,%d,%d)\n", iblk, jblk, blk);
               int thread_num = omp_get_thread_num();
               BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
               BlockSpec isrc(iblk, blk, m, n, cdata, a, lda, block_size);
               BlockSpec jsrc(jblk, blk, m, n, cdata, a, lda, block_size);
               // Record block state as assuming we've done up to col blk
               up_to_date[jblk*mblk+iblk] = blk;
               // Perform update
               ublk.form_contrib(
                     isrc, jsrc, work[thread_num], beta, upd_ij, ldupd
                     );
            }
         }
      }

      return cdata.calc_nelim(m);
   }

   /** Restore matrix to original state prior to aborted factorization.
    *
    * We take the first nelim_blk block columns as having suceeded, and for
    * each block look at up_to_date to see if they have assumed anything that
    * is incorrect:
    * 1) If up_to_date < nelim_blk then we apply any missing operations
    * 2) If up_to_date == nelim_blk then we do nothing
    * 3) If up_to_date > nelim_blk then we reset and recalculate completely
    * */
   static
   void restore(int const nelim_blk, int const m, int const n, int* perm, T* a,
         int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
         int const* old_perm, int const* up_to_date, int const block_size,
         std::vector<Workspace>& work, T* upd, int const ldupd) {
      typedef Block<T, BLOCK_SIZE, IntAlloc> BlockSpec;

      int const nblk = calc_nblk(n, block_size);
      int const mblk = calc_nblk(m, block_size);

      /* Restore perm for failed part */
      for(int i=nelim_blk*block_size; i<n; ++i)
         perm[i] = old_perm[i];

      /* Restore a */
      // NB: If we've accepted a block column as eliminated, then everything in
      // that column must have been updated fully. However, if the row is in
      // [nelim_blk+1:nblk] it may have a failed row permutation applied, so
      // for those we need a dependency structure.
      // e.g. we have a structure something like this:
      // OK |
      // OK | OK |
      // ?? | ?? | FAIL |
      // ?? | ?? |  ??  | ?? |
      // OK | OK |  ??  | ?? | ?? |
      // Hence we skip the "passed" diagonal block, and the rectangular block
      // below it. Then we just apply a reverse row permutation if required to
      // the failed rows in the passed columns.
      for(int jblk=0; jblk<nelim_blk; ++jblk) {
         for(int iblk=nelim_blk; iblk<nblk; ++iblk) {
            int progress = up_to_date[jblk*mblk+iblk];
            if(progress >= nelim_blk) {
               #pragma omp task default(none) \
                  firstprivate(iblk, jblk) \
                  shared(a, cdata, work) \
                  depend(inout: a[jblk*block_size*lda+iblk*block_size:1])
               {
                  int thread_num = omp_get_thread_num();
                  BlockSpec rblk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  rblk.apply_inv_rperm(work[thread_num]);
               }
            }
         }
      }
      // Now all eliminated columns are good, fix up remainder of node
      for(int jblk=nelim_blk; jblk<nblk; ++jblk) {
         for(int iblk=jblk; iblk<mblk; ++iblk) {
            int progress = up_to_date[jblk*mblk+iblk];
            if(progress >= nelim_blk) {
               // Bad updates applied, needs reset and full recalculation
               #pragma omp task default(none) \
                  firstprivate(iblk, jblk) \
                  shared(a, backup, cdata) \
                  depend(inout: a[jblk*block_size*lda+iblk*block_size:1])
               {
                  BlockSpec rblk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  rblk.full_restore(backup);
               }
               progress = -1;
            }
            // Apply any missing updates to a
            for(int kblk=progress+1; kblk<nelim_blk; ++kblk) {
               #pragma omp task default(none) \
                  firstprivate(iblk, jblk, kblk) \
                  shared(a, upd, cdata, work) \
                  depend(inout: a[jblk*block_size*lda+iblk*block_size:1]) \
                  depend(in: a[kblk*block_size*lda+iblk*block_size:1]) \
                  depend(in: a[kblk*block_size*lda+jblk*block_size:1])
               {
                  int thread_num = omp_get_thread_num();
                  BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  BlockSpec isrc(iblk, kblk, m, n, cdata, a, lda, block_size);
                  BlockSpec jsrc(jblk, kblk, m, n, cdata, a, lda, block_size);
                  ublk.update(isrc, jsrc, work[thread_num], 0.0, upd, ldupd);
               }
            }
         }
      }
      // Now all eliminated columns are good, fix up contribution block
      if(upd) {
         int uoffset = std::min(nblk*block_size, m) - n;
         T *upd2 = &upd[uoffset*(ldupd+1)];
         for(int jblk=nblk; jblk<mblk; ++jblk)
         for(int iblk=jblk; iblk<mblk; ++iblk) {
            int progress = up_to_date[jblk*mblk+iblk];
            if(progress >= nelim_blk) progress = -1; // needs complete reset
            T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + 
                              (iblk-nblk)*block_size];
            for(int kblk=progress+1; kblk<nelim_blk; ++kblk) {
               // NB: no need for isrc or jsrc dep as must be good already
               #pragma omp task default(none) \
                  firstprivate(iblk, jblk, kblk, upd_ij) \
                  shared(a, cdata, work) \
                  depend(inout: upd_ij[0:1])
               {
                  int thread_num = omp_get_thread_num();
                  BlockSpec ublk(iblk, jblk, m, n, cdata, a, lda, block_size);
                  BlockSpec isrc(iblk, kblk, m, n, cdata, a, lda, block_size);
                  BlockSpec jsrc(jblk, kblk, m, n, cdata, a, lda, block_size);
                  // Perform update
                  ublk.form_contrib(
                        isrc, jsrc, work[thread_num], 0.0, upd_ij, ldupd
                        );
               }
            }
         }
      }
      // FIXME: ...
      #pragma omp taskwait
      /*if(use_tasks && mblk > 1) {
         // We only need a taskwait here if we've launched any subtasks...
         // NB: we don't use taskgroup as it doesn't support if()
         #pragma omp taskwait
      }*/

   }

   /** \brief Print given matrix (for debug usage)
    *  \param m number of rows
    *  \param n number of columns
    *  \param perm[n] permutation of fully summed variables
    *  \param eliminated[n] status of fully summed variables
    *  \param a matrix values
    *  \param lda leading dimension of a
    */
   static
   void print_mat(int m, int n, const int *perm, std::vector<bool> const& eliminated, const T *a, int lda) {
      for(int row=0; row<m; row++) {
         if(row < n)
            printf("%d%s:", perm[row], eliminated[row]?"X":" ");
         else
            printf("%d%s:", row, "U");
         for(int col=0; col<std::min(n,row+1); col++)
            printf(" %10.4f", a[col*lda+row]);
         printf("\n");
      }
   }

   /** \brief return number of columns in given block column */
   static
   inline int get_ncol(int blk, int n, int block_size) {
      return calc_blkn(blk, n, block_size);
   }
   /** \brief return number of rows in given block row */
   static
   inline int get_nrow(int blk, int m, int block_size) {
      return calc_blkn(blk, m, block_size);
   }

public:
   /** Factorize an entire matrix */
   static
   int factor(int m, int n, int *perm, T *a, int lda, T *d, Backup& backup, struct cpu_factor_options const& options, PivotMethod pivot_method, int block_size, T beta, T* upd, int ldupd, std::vector<Workspace>& work, Allocator const& alloc=Allocator()) {
      /* Sanity check arguments */
      if(m < n) return -1;
      if(lda < n) return -4;

      /* Initialize useful quantities: */
      int nblk = calc_nblk(n, block_size);
      int mblk = calc_nblk(m, block_size);

      /* Temporary workspaces */
      ColumnData<T, IntAlloc> cdata(n, block_size, IntAlloc(alloc));
#ifdef PROFILE
      Profile::setNullState();
#endif

      /* Main loop
       *    - Each pass leaves any failed pivots in place and keeps everything
       *      up-to-date.
       *    - If no pivots selected across matrix, perform swaps to get large
       *      entries into diagonal blocks
       */
      int num_elim;
      if(pivot_method == PivotMethod::app_aggressive) {
         if(beta!=0.0) {
            // We don't support backup of contribution block at present,
            // so we only work if we assume it is zero to begin with
            throw std::runtime_error(
                  "run_elim_unpivoted currently only supports beta=0.0"
                  );
         }
         // Take a copy of perm
         typedef std::allocator_traits<IntAlloc> IATraits;
         IntAlloc intAlloc(alloc);
         int* perm_copy = IATraits::allocate(intAlloc, n);
         for(int i=0; i<n; ++i)
            perm_copy[i] = perm[i];
         size_t num_blocks = (upd) ? ((size_t) mblk)*mblk
                                   : ((size_t) mblk)*nblk;
         int* up_to_date = IATraits::allocate(intAlloc, num_blocks);
         for(size_t i=0; i<num_blocks; ++i)
            up_to_date[i] = -1; // not even backed up yet
         // Run the elimination
         if(use_tasks && mblk>1) {
            num_elim = run_elim_unpivoted(
                  m, n, perm, a, lda, d, cdata, backup, up_to_date, options,
                  block_size, beta, upd, ldupd, work, alloc
                  );
         } else {
            num_elim = run_elim_unpivoted_notasks(
                  m, n, perm, a, lda, d, cdata, backup, up_to_date, options,
                  block_size, beta, upd, ldupd, work, alloc
                  );
         }
         if(num_elim < 0) return num_elim; // error
         if(num_elim < n) {
#ifdef PROFILE
            {
               char buffer[200];
               snprintf(buffer, 200, "tpp-aggressive failed at %d / %d\n",
                        num_elim, n);
               Profile::addEvent("EV_AGG_FAIL", buffer);
            }
#endif
            // Factorization ecountered a pivoting failure.
            int nelim_blk = num_elim/block_size;
            // Rollback to known good state
            restore(
                  nelim_blk, m, n, perm, a, lda, d, cdata, backup, perm_copy,
                  up_to_date, block_size, work, upd, ldupd
                  );
            // Factorize more carefully
            if(use_tasks && mblk>1) {
               num_elim = run_elim_pivoted(
                     m, n, perm, a, lda, d, cdata, backup, options, block_size,
                     beta, upd, ldupd, work, alloc, nelim_blk
                     );
            } else {
               num_elim = run_elim_pivoted_notasks(
                     m, n, perm, a, lda, d, cdata, backup, options, block_size,
                     beta, upd, ldupd, work, alloc, nelim_blk
                     );
            }
            if(num_elim < 0) return num_elim; // error
         }
         IATraits::deallocate(intAlloc, up_to_date, num_blocks);
         IATraits::deallocate(intAlloc, perm_copy, n);
      } else {
         if(use_tasks && mblk>1) {
            num_elim = run_elim_pivoted(
                  m, n, perm, a, lda, d, cdata, backup, options,
                  block_size, beta, upd, ldupd, work, alloc
                  );
         } else {
            num_elim = run_elim_pivoted_notasks(
                  m, n, perm, a, lda, d, cdata, backup, options,
                  block_size, beta, upd, ldupd, work, alloc
                  );
         }
         if(num_elim < 0) return num_elim; // error
         backup.release_all_memory(); // we're done with it now, but we want
                                      // the memory back for reuse before we
                                      // get it automatically when it goes out
                                      // of scope.
      }

      if(num_elim < n) {
         // Permute failed entries to end
#ifdef PROFILE
         Profile::Task task_post("TA_LDLT_POST");
#endif
         std::vector<int, IntAlloc> failed_perm(n-num_elim, alloc);
         for(int jblk=0, insert=0, fail_insert=0; jblk<nblk; jblk++) {
            cdata[jblk].move_back(
                  get_ncol(jblk, n, block_size), &perm[jblk*block_size],
                  &perm[insert], &failed_perm[fail_insert]
                  );
            insert += cdata[jblk].nelim;
            fail_insert += get_ncol(jblk, n, block_size) - cdata[jblk].nelim;
         }
         for(int i=0; i<n-num_elim; ++i)
            perm[num_elim+i] = failed_perm[i];

         // Extract failed entries of a
         int nfail = n-num_elim;
         std::vector<T, TAlloc> failed_diag(nfail*n, alloc);
         std::vector<T, TAlloc> failed_rect(nfail*(m-n), alloc);
         for(int jblk=0, jfail=0, jinsert=0; jblk<nblk; ++jblk) {
            // Diagonal part
            for(int iblk=jblk, ifail=jfail, iinsert=jinsert; iblk<nblk; ++iblk) {
               copy_failed_diag(
                     get_ncol(iblk, n, block_size), get_ncol(jblk, n, block_size),
                     cdata[iblk], cdata[jblk],
                     &failed_diag[jinsert*nfail+ifail],
                     &failed_diag[iinsert*nfail+jfail],
                     &failed_diag[num_elim*nfail+jfail*nfail+ifail],
                     nfail, &a[jblk*block_size*lda+iblk*block_size], lda
                     );
               iinsert += cdata[iblk].nelim;
               ifail += get_ncol(iblk, n, block_size) - cdata[iblk].nelim;
            }
            // Rectangular part
            // (be careful with blocks that contain both diag and rect parts)
            copy_failed_rect(
                  get_nrow(nblk-1, m, block_size), get_ncol(jblk, n, block_size),
                  get_ncol(nblk-1, n, block_size), cdata[jblk],
                  &failed_rect[jfail*(m-n)+(nblk-1)*block_size-n], m-n,
                  &a[jblk*block_size*lda+(nblk-1)*block_size], lda
                  );
            for(int iblk=nblk; iblk<mblk; ++iblk) {
               copy_failed_rect(
                     get_nrow(iblk, m, block_size),
                     get_ncol(jblk, n, block_size), 0, cdata[jblk],
                     &failed_rect[jfail*(m-n)+iblk*block_size-n], m-n,
                     &a[jblk*block_size*lda+iblk*block_size], lda
                     );
            }
            jinsert += cdata[jblk].nelim;
            jfail += get_ncol(jblk, n, block_size) - cdata[jblk].nelim;
         }

         // Move data up
         for(int jblk=0, jinsert=0; jblk<nblk; ++jblk) {
            // Diagonal part
            for(int iblk=jblk, iinsert=jinsert; iblk<nblk; ++iblk) {
               move_up_diag(
                     cdata[iblk], cdata[jblk], &a[jinsert*lda+iinsert],
                     &a[jblk*block_size*lda+iblk*block_size], lda
                     );
               iinsert += cdata[iblk].nelim;
            }
            // Rectangular part
            // (be careful with blocks that contain both diag and rect parts)
            move_up_rect(
                  get_nrow(nblk-1, m, block_size),
                  get_ncol(nblk-1, n, block_size), cdata[jblk],
                  &a[jinsert*lda+(nblk-1)*block_size],
                  &a[jblk*block_size*lda+(nblk-1)*block_size], lda
                  );
            for(int iblk=nblk; iblk<mblk; ++iblk)
               move_up_rect(
                     get_nrow(iblk, m, block_size), 0, cdata[jblk],
                     &a[jinsert*lda+iblk*block_size],
                     &a[jblk*block_size*lda+iblk*block_size], lda
                     );
            jinsert += cdata[jblk].nelim;
         }
         
         // Store failed entries back to correct locations
         // Diagonal part
         for(int j=0; j<n; ++j)
         for(int i=std::max(j,num_elim), k=i-num_elim; i<n; ++i, ++k)
            a[j*lda+i] = failed_diag[j*nfail+k];
         // Rectangular part
         T* arect = &a[num_elim*lda+n];
         for(int j=0; j<nfail; ++j)
         for(int i=0; i<m-n; ++i)
            arect[j*lda+i] = failed_rect[j*(m-n)+i];
#ifdef PROFILE
         task_post.done();
#endif
      }

      if(debug) {
         std::vector<bool> eliminated(n);
         for(int i=0; i<num_elim; i++) eliminated[i] = true;
         for(int i=num_elim; i<n; i++) eliminated[i] = false;
         printf("FINAL:\n");
         print_mat(m, n, perm, eliminated, a, lda);
      }

      return num_elim;
   }
};

} /* namespace spral::ssids:cpu::ldlt_app_internal */

using namespace spral::ssids::cpu::ldlt_app_internal;

template<typename T>
size_t ldlt_app_factor_mem_required(int m, int n, int block_size) {
#if defined(__AVX512F__)
  int const align = 64;
#elif defined(__AVX__)
  int const align = 32;
#else
  int const align = 16;
#endif
   return align_lda<T>(m) * n * sizeof(T) + align; // CopyBackup
}

template<typename T, typename Allocator>
int ldlt_app_factor(int m, int n, int* perm, T* a, int lda, T* d, T beta, T* upd, int ldupd, struct cpu_factor_options const& options, std::vector<Workspace>& work, Allocator const& alloc) {
   // If we've got a tall and narrow node, adjust block size so each block
   // has roughly blksz**2 entries
   // FIXME: Decide if this reshape is actually useful, given it will generate
   //        a lot more update tasks instead?
   int outer_block_size = options.cpu_block_size;
   /*if(n < outer_block_size) {
       outer_block_size = int((long(outer_block_size)*outer_block_size) / n);
   }*/

#ifdef PROFILE
   Profile::setState("TA_MISC1");
#endif

   // Template parameters and workspaces
   bool const debug = false;
   //PoolBackup<T, Allocator> backup(m, n, outer_block_size, alloc);
   CopyBackup<T, Allocator> backup(m, n, outer_block_size, alloc);

   // Actual call
   bool const use_tasks = true;
   return LDLT
      <T, INNER_BLOCK_SIZE, CopyBackup<T,Allocator>, use_tasks, debug,
       Allocator>
      ::factor(
            m, n, perm, a, lda, d, backup, options, options.pivot_method,
            outer_block_size, beta, upd, ldupd, work, alloc
            );
}
template int ldlt_app_factor<double, BuddyAllocator<double,std::allocator<double>>>(int, int, int*, double*, int, double*, double, double*, int, struct cpu_factor_options const&, std::vector<Workspace>&, BuddyAllocator<double,std::allocator<double>> const& alloc);

template <typename T>
void ldlt_app_solve_fwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx) {
   if(nrhs==1) {
      host_trsv(FILL_MODE_LWR, OP_N, DIAG_UNIT, n, l, ldl, x, 1);
      if(m > n)
         gemv(OP_N, m-n, n, -1.0, &l[n], ldl, x, 1, 1.0, &x[n], 1);
   } else {
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT, n, nrhs, 1.0, l, ldl, x, ldx);
      if(m > n)
         host_gemm(OP_N, OP_N, m-n, nrhs, n, -1.0, &l[n], ldl, x, ldx, 1.0, &x[n], ldx);
   }
}
template void ldlt_app_solve_fwd<double>(int, int, double const*, int, int, double*, int);

template <typename T>
void ldlt_app_solve_diag(int n, T const* d, int nrhs, T* x, int ldx) {
   for(int i=0; i<n; ) {
      if(i+1==n || std::isfinite(d[2*i+2])) {
         // 1x1 pivot
         T d11 = d[2*i];
         for(int r=0; r<nrhs; ++r)
            x[r*ldx+i] *= d11;
         i++;
      } else {
         // 2x2 pivot
         T d11 = d[2*i];
         T d21 = d[2*i+1];
         T d22 = d[2*i+3];
         for(int r=0; r<nrhs; ++r) {
            T x1 = x[r*ldx+i];
            T x2 = x[r*ldx+i+1];
            x[r*ldx+i]   = d11*x1 + d21*x2;
            x[r*ldx+i+1] = d21*x1 + d22*x2;
         }
         i += 2;
      }
   }
}
template void ldlt_app_solve_diag<double>(int, double const*, int, double*, int);

template <typename T>
void ldlt_app_solve_bwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx) {
   if(nrhs==1) {
      if(m > n)
         gemv(OP_T, m-n, n, -1.0, &l[n], ldl, &x[n], 1, 1.0, x, 1);
      host_trsv(FILL_MODE_LWR, OP_T, DIAG_UNIT, n, l, ldl, x, 1);
   } else {
      if(m > n)
         host_gemm(OP_T, OP_N, n, nrhs, m-n, -1.0, &l[n], ldl, &x[n], ldx, 1.0, x, ldx);
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_T, DIAG_UNIT, n, nrhs, 1.0, l, ldl, x, ldx);
   }
}
template void ldlt_app_solve_bwd<double>(int, int, double const*, int, int, double*, int);

}}} /* namespaces spral::ssids::cpu */
