/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

namespace spral { namespace ssids { namespace cpu {

/** \brief Supported CPU architectures that can be targeted */
enum cpu_arch {
   CPU_ARCH_GENERIC, // No explicit vectorization
   CPU_ARCH_AVX,     // Allow AVX optimized kernel (Sandy-/Ivy-Bridge)
   CPU_ARCH_AVX2     // Allow use of AVX2 (FMA3)
};

/** \brief CPU_BEST_ARCH is set to a value of enum cpu_arch that represents the best supported instruction set supported by current compiler and compiler flags */
#ifdef __AVX2__
const enum cpu_arch CPU_BEST_ARCH = CPU_ARCH_AVX2;
#else
# ifdef __AVX__
const enum cpu_arch CPU_BEST_ARCH = CPU_ARCH_AVX;
# else
const enum cpu_arch CPU_BEST_ARCH = CPU_ARCH_GENERIC;
# endif
#endif


/** \brief The warpSize for the current architecture as a constant */
const int WARPSIZE = 32;

/** \brief bub::operation enumerates operations that can be applied to a matrix
  * argument of a BLAS call.
  */
enum operation {
   /// No operation (i.e. non-transpose). Equivalent to BLAS op='N'.
   OP_N,
   /// Transposed. Equivalent to BLAS op='T'.
   OP_T
};

/// \brief bub::diagonal enumerates nature of matrix diagonal.
enum diagonal {
   /// All diagonal elements are assumed to be identically 1.0
   DIAG_UNIT,
   /// Diagonal elements are specified in matrix data
   DIAG_NON_UNIT
};

/// \brief bub::fillmode enumerates which part of the matrix is specified.
enum fillmode {
   /// The lower triangular part of the matrix is specified
   FILL_MODE_LWR,
   /// The upper triangular part of the matrix is specified
   FILL_MODE_UPR
};

/** \brief bub::side enumerates whether the primary operand is applied on the
 *  left or right of a secondary operand */
enum side {
   /// Primary operand applied on left of secondary
   SIDE_LEFT,
   /// Primary operand applied on right of secondary
   SIDE_RIGHT
};

/// \brief enumerates different layouts that can be used for data
enum layout {
   /** Use a block cyclic layout
     * \par Example
     * For a \f$2\times2\f$ threadblock with 2 rows and 4 cols per thread,
     * entries are assigned as: \n
     * \f$\left(\begin{array}{cc|cc|cc|cc}
     * 0.0 & 0.1 & 0.0 & 0.1 & 0.0 & 0.1 & 0.0 & 0.1\\
     * 1.0 & 1.1 & 1.0 & 1.1 & 1.0 & 1.1 & 1.0 & 1.1\\
     * \hline
     * 0.0 & 0.1 & 0.0 & 0.1 & 0.0 & 0.1 & 0.0 & 0.1\\
     * 1.0 & 1.1 & 1.0 & 1.1 & 1.0 & 1.1 & 1.0 & 1.1\\
     * \end{array}\right)\f$
     */
   LAYOUT_BLOCK_CYCLIC,
   /** Use a block layout
     * \par Example
     * For a \f$2\times2\f$ threadblock with 2 rows and 4 cols per thread,
     * entries are assigned as: \n
     * \f$\left(\begin{array}{cccc|cccc}
     * 0.0 & 0.0 & 0.0 & 0.0 & 0.1 & 0.1 & 0.1 & 0.1\\
     * 0.0 & 0.0 & 0.0 & 0.0 & 0.1 & 0.1 & 0.1 & 0.1\\
     * \hline
     * 1.0 & 1.0 & 1.0 & 1.0 & 1.1 & 1.1 & 1.1 & 1.1\\
     * 1.0 & 1.0 & 1.0 & 1.0 & 1.1 & 1.1 & 1.1 & 1.1\\
     * \end{array}\right)\f$
     */
   LAYOUT_BLOCK
};

/**
 * \brief bub::TrsmAlgorithm enumerates alternative algorithms for bub::Trsm to
 * solve a triangular linear system.
 */
enum TrsmAlgorithm {
   /** The __shfl() instruction is used to communicate data between threads.
    *
    * \par Portability
    * This requires a device with CUDA Capability at least 3.x
    */
   TRSM_SHFL,
   /** A shared memory array is used to communicate data between threads.
    *
    * \par Performance Considerations
    * TRSM_SHFL is likely to be faster where supported (particuarly for float
    * rather than double) and reduces the amount of shared memory required.
    */
   TRSM_SHMEM
};

/**
 * \brief bub::PotrfAlgorithm enumerates alternative algorithms for bub::Potrf
 * to factorize a positive-definite matrix.
 */
enum PotrfAlgorithm {
   POTRF_SHFL,
   POTRF_SHMEM
};

namespace util {
#ifdef __CUDACC__
   template <typename T>
   __device__
   __forceinline__
   T loadVolatile(volatile T &val) {
      return val;
   }
   template <typename T>
   __device__
   __forceinline__
   const T loadVolatile(const volatile T &val) {
      return val;
   }
#endif
}

}}} /* namespaces spral::ssids::cpu */
