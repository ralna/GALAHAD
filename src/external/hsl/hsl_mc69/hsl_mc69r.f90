! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 14:30 GMT.

#include "hsl_subset.h"

    MODULE hsl_mc69_real
      use hsl_kinds_real, only: ip_, lp_, rp_
      implicit none

      private
      public :: HSL_MATRIX_UNDEFINED,                                          &
         HSL_MATRIX_REAL_RECT, HSL_MATRIX_CPLX_RECT,                           &
         HSL_MATRIX_REAL_UNSYM, HSL_MATRIX_CPLX_UNSYM,                         &
         HSL_MATRIX_REAL_SYM_PSDEF, HSL_MATRIX_CPLX_HERM_PSDEF,                &
         HSL_MATRIX_REAL_SYM_INDEF, HSL_MATRIX_CPLX_HERM_INDEF,                &
         HSL_MATRIX_CPLX_SYM,                                                  &
         HSL_MATRIX_REAL_SKEW, HSL_MATRIX_CPLX_SKEW
      public :: mc69_coord_convert, mc69_set_values
      LOGICAL, PUBLIC, PROTECTED :: mc69_available = .FALSE.

      integer(ip_), parameter :: HSL_MATRIX_UNDEFINED      =  0
      integer(ip_), parameter :: HSL_MATRIX_REAL_RECT      =  1
      integer(ip_), parameter :: HSL_MATRIX_REAL_UNSYM     =  2
      integer(ip_), parameter :: HSL_MATRIX_REAL_SYM_PSDEF =  3
      integer(ip_), parameter :: HSL_MATRIX_REAL_SYM_INDEF =  4
      integer(ip_), parameter :: HSL_MATRIX_REAL_SKEW      =  6
      integer(ip_), parameter :: HSL_MATRIX_CPLX_RECT      = -1
      integer(ip_), parameter :: HSL_MATRIX_CPLX_UNSYM     = -2
      integer(ip_), parameter :: HSL_MATRIX_CPLX_HERM_PSDEF= -3
      integer(ip_), parameter :: HSL_MATRIX_CPLX_HERM_INDEF= -4
      integer(ip_), parameter :: HSL_MATRIX_CPLX_SYM       = -5
      integer(ip_), parameter :: HSL_MATRIX_CPLX_SKEW      = -6

      interface mc69_coord_convert
         module procedure mc69_coord_convert_real
      end interface mc69_coord_convert

      interface mc69_set_values
         module procedure mc69_set_values_real
      end interface mc69_set_values

    CONTAINS

      subroutine mc69_coord_convert_real(matrix_type, m, n, ne, row, col,       &
          ptr_out, row_out, flag, val_in, val_out, lmap, map, lp, noor, ndup)
         integer(ip_), intent(in) :: matrix_type
         integer(ip_), intent(in) :: m
         integer(ip_), intent(in) :: n
         integer(ip_), intent(in) :: ne
         integer(ip_), intent(in) :: row(ne)
         integer(ip_), intent(in) :: col(ne)
         integer(ip_), intent(out) :: ptr_out(n+1)
         integer(ip_), allocatable, intent(out) :: row_out(:)
         integer(ip_), intent(out) :: flag
         real(rp_), optional, intent(in) :: val_in(*)
         real(rp_), optional, allocatable :: val_out(:)
         integer(ip_), optional, intent(out) :: lmap
         integer(ip_), optional, allocatable :: map(:)
         integer(ip_), optional, intent(in) :: lp
         integer(ip_), optional, intent(out) :: noor
         integer(ip_), optional, intent(out) :: ndup

!  Dummy subroutine available with GALAHAD

         flag = -1
      end subroutine mc69_coord_convert_real

      subroutine mc69_set_values_real(matrix_type, lmap, map, val, ne, val_out)
         integer(ip_), intent(in) :: matrix_type
         integer(ip_), intent(in) :: lmap
         integer(ip_), intent(in) :: map(lmap)
         real(rp_), intent(in) :: val(*)
         integer(ip_), intent(in) :: ne
         real(rp_), intent(out) :: val_out(ne)

!  Dummy subroutine available with GALAHAD

         val_out(1:ne) = 0.0_rp_
      end subroutine mc69_set_values_real

    END MODULE hsl_mc69_real
