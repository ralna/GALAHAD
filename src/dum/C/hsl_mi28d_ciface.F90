! THIS VERSION: GALAHAD 4.0 - 2022-01-27 AT 07:45 GMT

!-*-*-  G A L A H A D  -  D U M M Y   M I 2 8 _ C I F A C E   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 27th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

 MODULE HSL_MI28_double_ciface
   USE iso_c_binding
   USE HSL_MI28_double, ONLY:                                                  &
      f_mi28_control => mi28_control,                                          &
      f_mi28_info    => mi28_info

!--------------------
!   P r e c i s i o n
!--------------------

   IMPLICIT NONE

   INTEGER, PARAMETER :: wp = C_DOUBLE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

   TYPE, BIND( C ) :: MI28_control

!  use C (=0) or Fortran (/=0) sparse matrix indexing for arrays

    INTEGER( KIND = C_INT ) :: f_arrays

!  initial shift

     REAL( KIND = wp ) :: alpha

!  if set to true, user's data is checked. Otherwise, no checking and may 
!  fail in unexpected way if there are duplicates/out-of-range entries.

     LOGICAL( KIND = C_BOOL ) :: check

!  controls ordering of A. Options:
!   <= 0  no ordering
!      1  Reverse Cuthill McKee (RCM)
!      2  Approximate Minimum Degree (AMD)
!      3  user-supplied ordering
!      4  ascending degree
!      5  Metis
!   >= 6, Sloan (MC61)

     INTEGER( KIND = C_INT ) :: iorder

!  controls whether scaling is used. Options:
!    = 1 is Lin and More scaling (l2 scaling)
!    = 2 is mc77 scaling
!    = 3 is mc64 scaling
!    = 4 is diagonal scaling
!    = 5 user-supplied scaling
!   <= 0, no scaling
!   >= 6, Lin and More

     INTEGER( KIND = C_INT ) :: iscale

!  Shift after first breakdown is max(shift_factor*alpha,lowalpha)

     REAL( KIND = wp ) :: lowalpha

!  During search for shift, we decrease the lower bound max(alpha,lowalpha) 
!  on the shift by shift_factor2 at most maxshift times (so this limits the
!  number of refactorizations that are performed ... idea is reducing alpha 
!  as much as possible will give better preconditioner but reducing too far 
!  will lead to breakdown and then a refactorization is required (expensive 
!  so limit number of reductions). Note: Lin and More set this to 3.

     INTEGER( KIND = C_INT ) :: maxshift

!  controls whether entries of RR^T that cause no additional fill are allowed. 
!  They are allowed if rrt = .true. and not otherwise.

     LOGICAL( KIND = C_BOOL ) :: rrt

!  if the current shift is found to be too small, it is increased by at least 
!  a factor of shift_factor. Values <= 1.0 are treated as default.

     REAL( KIND = wp ) :: shift_factor

!  if factorization is successful with current (non zero) shift, the shift is
!  reduced by a factor of shift_factor2.  Values <= 1.0 are treated as default.

     REAL( KIND = wp ) :: shift_factor2
     REAL( KIND = wp ) :: small

!  used to select "small" entries that are dropped from L (but may be 
!  included in R). 

     REAL( KIND = wp ) :: tau1

!  used to select "tiny" entries that are dropped from R.  Require
!  tau2 < tau1 (otherwise, tau2 = 0.0 is used locally).

     REAL( KIND = wp ) :: tau2

!  unit number for error messages. Printing is suppressed if unit_error < 0.

     INTEGER( KIND = C_INT ) :: unit_error

!  unit number for warning messages. Printing is suppressed if unit_warning < 0.

     INTEGER( KIND = C_INT ) :: unit_warning
   END TYPE MI28_control

   TYPE, BIND(C) :: MI28_info

!  semibandwidth after MC61

     INTEGER( KIND = C_INT ) :: band_after

!  semibandwidth before MC61

     INTEGER( KIND = C_INT ) :: band_before

!  number of duplicated entries found in row.

     INTEGER( KIND = C_INT ) :: dup

!  error flag

     INTEGER( KIND = C_INT ) :: flag

!  error flag from mc61

     INTEGER( KIND = C_INT ) :: flag61

!  error flag from hsl_mc64

     INTEGER( KIND = C_INT ) :: flag64

!  error flag from hsl_mc68

     INTEGER( KIND = C_INT ) :: flag68

!  error flag from mc77

     INTEGER( KIND = C_INT ) :: flag77

!  number of restarts (after reducing the shift)

     INTEGER( KIND = C_INT ) :: nrestart

!  number of non-zero shifts used

     INTEGER( KIND = C_INT ) :: nshift

!  number of out-of-range entries found in row.

     INTEGER( KIND = C_INT ) :: oor

!  semibandwidth before MC61

     REAL( KIND = wp ) :: profile_before

!  semibandwidth after MC61

     REAL( KIND = wp ) :: profile_after

!  size of arrays jr and ar that are used for r   

     INTEGER( KIND = C_INT64_T ) :: size_r

!  Fortran stat parameter

     INTEGER( KIND = C_INT ) :: stat

!  on successful exit, holds shift used

     REAL( KIND = wp ) :: alpha
   END TYPE MI28_info

 CONTAINS

   SUBROUTINE copy_control_in( ccontrol, fcontrol, f_arrays )
     TYPE( MI28_control ), INTENT( IN ) :: ccontrol
     TYPE( f_mi28_control), INTENT( OUT ) :: fcontrol
     LOGICAL, INTENT( OUT ) :: f_arrays
     f_arrays               = ccontrol%f_arrays /= 0
     fcontrol%alpha         = ccontrol%alpha
     fcontrol%check         = ccontrol%check
     fcontrol%iorder        = ccontrol%iorder
     fcontrol%iscale        = ccontrol%iscale
     fcontrol%lowalpha      = ccontrol%lowalpha
     fcontrol%maxshift      = ccontrol%maxshift
     fcontrol%rrt           = ccontrol%rrt
     fcontrol%shift_factor  = ccontrol%shift_factor
     fcontrol%shift_factor2 = ccontrol%shift_factor2
     fcontrol%small         = ccontrol%small
     fcontrol%tau1          = ccontrol%tau1
     fcontrol%tau2          = ccontrol%tau2
     fcontrol%unit_error    = ccontrol%unit_error
     fcontrol%unit_warning  = ccontrol%unit_warning
   END SUBROUTINE copy_control_in

   SUBROUTINE copy_info_out( finfo, cinfo )
     TYPE( f_mi28_info ), INTENT( IN ) :: finfo
     TYPE( mi28_info ), INTENT( OUT ) :: cinfo
     cinfo%band_after     = finfo%band_after
     cinfo%band_before    = finfo%band_before
     cinfo%dup            = finfo%dup
     cinfo%flag           = finfo%flag
     cinfo%flag61         = finfo%flag61
     cinfo%flag64         = finfo%flag64
     cinfo%flag68         = finfo%flag68
     cinfo%flag77         = finfo%flag77
     cinfo%nrestart       = finfo%nrestart
     cinfo%nshift         = finfo%nshift
     cinfo%oor            = finfo%oor
     cinfo%profile_before = finfo%profile_before
     cinfo%profile_after  = finfo%profile_after
     cinfo%size_r         = finfo%size_r
     cinfo%stat           = finfo%stat
     cinfo%alpha          = finfo%alpha
   END SUBROUTINE copy_info_out

 END MODULE HSL_MI28_double_ciface
