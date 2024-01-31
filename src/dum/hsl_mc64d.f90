! THIS VERSION: GALAHAD 4.3 - 2024-01-14 AT 14:15 GMT.

#include "galahad_modules.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 6 4    M O D U L E  -*-*-*-

MODULE hsl_mc64_double

   USE GALAHAD_KINDS
   USE hsl_zd11_double

   IMPLICIT NONE

   private
   public :: mc64_control, mc64_info, mc64_initialize, mc64_matching

   TYPE mc64_control
!     real(dp_) :: relax = 0.0_dp_   ! Relaxes matching
      integer(ip_) :: lp = 6    ! Unit for error messages
      integer(ip_) :: wp = 6    ! Unit for warning messages
      integer(ip_) :: sp = -1    ! Unit for statistical output
      integer(ip_) :: ldiag = 2 ! Controls level of diagnostic output
      integer(ip_) :: checking = 0 ! Control for checking input

   END TYPE mc64_control

   TYPE mc64_info
      integer(ip_) :: flag = 0  ! Flags success or failure case
      integer(ip_) :: more = -1   ! More information on failure
      integer(ip_) :: strucrank = -1 ! Structural rank
      integer(ip_) :: stat = 0  ! STAT value after allocate failure
   END TYPE mc64_info

   interface mc64_matching
      module procedure mc64_matching_zd11_double
      module procedure mc64_matching_hslstd_double
   end interface mc64_matching

CONTAINS

   SUBROUTINE mc64_initialize(control)
      type(mc64_control), intent(out) :: control
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
      "( ' We regret that the solution options that you have ', /,             &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC64_initialize with its  HSL namesake ', /,              &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    END SUBROUTINE mc64_initialize

   SUBROUTINE mc64_matching_zd11_double(job,matrix,control,info,perm,scale)
      USE GALAHAD_SYMBOLS
      integer(ip_),  intent(in) :: job ! Control parameter for algorithm choice
      type(zd11_type), intent(in) :: matrix
      type(mc64_control), intent(in) :: control
      type(mc64_info), intent(out) :: info
      integer(ip_),  intent(out) :: perm(matrix%m + matrix%n)
      real(dp_), optional, intent(out) :: scale(matrix%m + matrix%n)
      IF ( control%lp >= 0 ) WRITE( control%lp,                                &
      "( ' We regret that the solution options that you have ', /,             &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC64_matching with its HSL namesake ', /,                 &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )
       info%flag = GALAHAD_unavailable_option
   END SUBROUTINE mc64_matching_zd11_double

   SUBROUTINE mc64_matching_hslstd_double(job,matrix_type,m,n,ptr,row,         &
                                          val,control,info,perm,scale)
      integer(ip_), intent(in) :: job ! Control parameter for algorithm choice
      integer(ip_), intent(in) :: matrix_type
      integer(ip_), intent(in) :: m
      integer(ip_), intent(in) :: n
      integer(ip_), dimension(n+1), intent(in) :: ptr
      integer(ip_), dimension(*), intent(in) :: row
      real(dp_), dimension(*), intent(in) :: val
      type(mc64_control), intent(in) :: control
      type(mc64_info), intent(out) :: info
      integer(ip_), intent(out) :: perm(m + n)
      real(dp_), optional, intent(out) :: scale(m + n)
   END SUBROUTINE mc64_matching_hslstd_double

END MODULE hsl_mc64_double
