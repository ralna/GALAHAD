/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <cmath>
#include <cstdio>
#include <limits>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

namespace spral { namespace ssids { namespace cpu {

/** \brief The SimdVec class isolates use of AVX/whatever intrinsics in a
 *  single place for ease of upgrading to future instruction sets.
 *
 *  Support is only added as required, so don't expect all intrinsics to be
 *  wrapped yet! */
template <typename T>
class SimdVec;

template <>
class SimdVec<double> {
public:
   /*******************************************
    * Properties of the type
    *******************************************/

#if defined(__AVX2__) || defined(__AVX__)
   /// Length of underlying vector type
   static const int vector_length = 4;
   /// Typedef for underlying vector type containing doubles
   typedef __m256d simd_double_type;
#else
   /// Length of underlying vector type
   static const int vector_length = 1;
   /// Typedef for underlying vector type containing doubles
   typedef double simd_double_type;
#endif

   /*******************************************
    * Constructors
    *******************************************/

   /// Uninitialized value constructor
   SimdVec()
   {}
   /// Initialize all entries in vector to given scalar value
   SimdVec(const double initial_value)
   {
#if defined(__AVX2__) || defined(__AVX__)
      val = _mm256_set1_pd(initial_value);
#else
      val = initial_value;
#endif
   }
#if defined(__AVX2__) || defined(__AVX__)
   /// Initialize with underlying vector type
   SimdVec(const simd_double_type &initial_value) {
      val = initial_value;
   }
#endif
   /// Initialize with another SimdVec
   SimdVec(const SimdVec<double> &initial_value) {
      val = initial_value.val;
   }
#if defined(__AVX2__) || defined(__AVX__)
   /// Initialize as a vector by specifying all entries (no version for non-avx)
   SimdVec(double x1, double x2, double x3, double x4) {
      val = _mm256_set_pd(x4, x3, x2, x1); // Reversed order expected
   }
#endif

   /*******************************************
    * Memory load/store
    *******************************************/

   /// Load from suitably aligned memory
   static
   const SimdVec load_aligned(const double *src) {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec( _mm256_load_pd(src) );
#else
      return SimdVec( src[0] );
#endif
   }

   /// Load from unaligned memory
   static
   const SimdVec load_unaligned(const double *src) {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec( _mm256_loadu_pd(src) );
#else
      return SimdVec( src[0] );
#endif
   }

   /// Extract value as array
   void store_aligned(double *dest) const {
#if defined(__AVX2__) || defined(__AVX__)
      _mm256_store_pd(dest, val);
#else
      dest[0] = val;
#endif
   }

   /// Extract value as array
   void store_unaligned(double *dest) const {
#if defined(__AVX2__) || defined(__AVX__)
      _mm256_storeu_pd(dest, val);
#else
      dest[0] = val;
#endif
   }

   /*******************************************
    * Named operations
    *******************************************/

   /// Blend operation: returns (mask) ? x2 : x1
   friend
   SimdVec blend(const SimdVec &x1, const SimdVec &x2, const SimdVec &mask) {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec( _mm256_blendv_pd(x1.val, x2.val, mask.val) );
#else
      return SimdVec( (mask.val) ? x2 : x1 );
#endif
   }

   /// Returns absolute values
   friend
   SimdVec fabs(const SimdVec &x) {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec(
            _mm256_andnot_pd(_mm256_set1_pd(-0.0), x)
         );
#else
      return SimdVec( fabs(x.val) );
#endif
   }

   /// Return a = b * c + a
   friend
   SimdVec fmadd(const SimdVec &a, const SimdVec &b, const SimdVec &c) {
#if defined(__AVX2__)
      return SimdVec(
            _mm256_fmadd_pd(b.val, c.val, a.val)
         );
#else
      return b*c + a;
#endif
   }

   /*******************************************
    * Operators
    *******************************************/

   /// Conversion to underlying type
   operator simd_double_type() const {
      return val;
   }

   /// Extract indvidual elements of vector (messy and inefficient)
   /// idx MUST be < vector_length.
   double operator[](size_t idx) const {
      double
#if defined(__AVX512F__)
        __attribute__((aligned(64)))
#elif defined(__AVX__)
        __attribute__((aligned(32)))
#else
        __attribute__((aligned(16)))
#endif
        val_as_array[vector_length];
      store_aligned(val_as_array);
      return val_as_array[idx];
   }

   /// Vector valued GT comparison
   friend
   SimdVec operator>(const SimdVec &lhs, const SimdVec &rhs) {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec( _mm256_cmp_pd(lhs.val, rhs.val, _CMP_GT_OQ) );
#else
      return SimdVec( lhs.val > rhs.val );
#endif
   }

   /// Bitwise and
   friend
   SimdVec operator&(const SimdVec &lhs, const SimdVec &rhs) {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec( _mm256_and_pd(lhs.val, rhs.val) );
#else
      return SimdVec( lhs.val && rhs.val );
#endif
   }

   /// Multiply
   // NB: don't override builtin operator*(double,double) in scalar case
#if defined(__AVX2__) || defined(__AVX__)
   friend
   SimdVec operator*(const SimdVec &lhs, const SimdVec &rhs) {
      return SimdVec( _mm256_mul_pd(lhs.val, rhs.val) );
   }
#endif

   SimdVec& operator*=(const SimdVec &rhs) {
      *this = *this * rhs;
      return *this;
   }

   /// Add
   // NB: don't override builtin operator*(double,double) in scalar case
#if defined(__AVX2__) || defined(__AVX__)
   friend
   SimdVec operator+(const SimdVec &lhs, const SimdVec &rhs) {
      return SimdVec( _mm256_add_pd(lhs.val, rhs.val) );
   }
#endif

   /*******************************************
    * Factory functions for special cases
    *******************************************/

   /// Returns an instance initialized to zero using custom instructions
   static
   SimdVec zero() {
#if defined(__AVX2__) || defined(__AVX__)
      return SimdVec(_mm256_setzero_pd());
#else
      return SimdVec(0.0);
#endif
   }

   /// Returns a vector with all positions idx or above set to true, otherwise
   /// false.
   static
   SimdVec gt_mask(int idx) {
#if defined(__AVX2__) || defined(__AVX__)
      const double avx_true  = -std::numeric_limits<double>::quiet_NaN();
      const double avx_false = 0.0;
      switch(idx) {
         case 0:  return SimdVec(avx_true,   avx_true,  avx_true,  avx_true);
         case 1:  return SimdVec(avx_false,  avx_true,  avx_true,  avx_true);
         case 2:  return SimdVec(avx_false, avx_false,  avx_true,  avx_true);
         case 3:  return SimdVec(avx_false, avx_false, avx_false,  avx_true);
         default: return SimdVec(avx_false, avx_false, avx_false, avx_false);
      }
#else
      return (idx>0) ? SimdVec(false) : SimdVec(true);
#endif
   }

   /*******************************************
    * Debug functions
    *******************************************/

   /// Prints the vector (inefficient, use for debug only)
   void print() {
      for(int i=0; i<vector_length; i++) printf(" %e", (*this)[i]);
   }

private:
   /// Underlying vector that this type wraps
   simd_double_type val;
};

}}} /* namespaces spral::ssids::cpu */
