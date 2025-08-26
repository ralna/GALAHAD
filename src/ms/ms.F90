! THIS VERSION: GALAHAD 5.3 - 2025-08-25 AT 09:30 GMT.

#include "ssids_procedures.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ M S   M O D U L E  *-*-*-*-*-*-*-*-*-*-*-

!        ------------------------------------------------------------
!       | Matrix scaling package originally spral_scaling from SPRAL |
!        ------------------------------------------------------------

!  COPYRIGHT (c) 2014 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked from SPRAL and extended for GALAHAD, Nick Gould, version 3.1, 2016

      MODULE GALAHAD_MS_precision

        USE GALAHAD_KINDS_precision
        USE GALAHAD_MU_precision, ONLY: MU_half_to_full
        IMPLICIT NONE

        PRIVATE
        PUBLIC :: MS_auction_scale_sym, MS_auction_scale_unsym,                &
                  MS_equilib_scale_sym, MS_equilib_scale_unsym,                &
                  MS_hungarian_scale_sym, MS_hungarian_scale_unsym,            &
                  MS_hungarian_match

!----------------------
!   P a r a m e t e r s
!----------------------

        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ALLOCATION = - 1
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_SINGULAR = - 2
        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_SINGULAR = 1
        REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
        REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
        REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
        REAL ( KIND = rp_ ), PARAMETER :: rinf = HUGE( rinf )

!----------------------
!   I n t e r f a c e s
!----------------------

        INTERFACE MS_auction_scale_sym
          MODULE PROCEDURE auction_scale_sym_int32
          MODULE PROCEDURE auction_scale_sym_int64
        END INTERFACE MS_auction_scale_sym

        INTERFACE MS_auction_scale_unsym
          MODULE PROCEDURE auction_scale_unsym_int32
          MODULE PROCEDURE auction_scale_unsym_int64
        END INTERFACE MS_auction_scale_unsym

        INTERFACE MS_equilib_scale_sym
          MODULE PROCEDURE equilib_scale_sym_int32
          MODULE PROCEDURE equilib_scale_sym_int64
        END INTERFACE MS_equilib_scale_sym

        INTERFACE MS_equilib_scale_unsym
          MODULE PROCEDURE equilib_scale_unsym_int32
          MODULE PROCEDURE equilib_scale_unsym_int64
        END INTERFACE MS_equilib_scale_unsym

        INTERFACE MS_hungarian_scale_sym
          MODULE PROCEDURE hungarian_scale_sym_int32
          MODULE PROCEDURE hungarian_scale_sym_int64
        END INTERFACE MS_hungarian_scale_sym

        INTERFACE MS_hungarian_scale_unsym
          MODULE PROCEDURE hungarian_scale_unsym_int32
          MODULE PROCEDURE hungarian_scale_unsym_int64
        END INTERFACE MS_hungarian_scale_unsym

        INTERFACE MS_hungarian_match
          MODULE PROCEDURE hungarian_match
        END INTERFACE MS_hungarian_match

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

        TYPE, PUBLIC :: MS_auction_control_type
          INTEGER ( KIND = ip_ ) :: max_iterations = 30000
          INTEGER ( KIND = ip_ ) :: max_unchanged( 3 ) = (/ 10, 100, 100 /)
          REAL :: min_proportion( 3 ) = (/ 0.90, 0.0, 0.0 /)
          REAL :: eps_initial = 0.01
        END TYPE MS_auction_control_type

        TYPE, PUBLIC :: MS_auction_inform_type
          INTEGER ( KIND = ip_ ) :: flag = 0 ! success or failure
          INTEGER ( KIND = ip_ ) :: stat = 0 ! stat value on allocation failure
          INTEGER ( KIND = ip_ ) :: matched = 0 ! # matched rows/cols
          INTEGER ( KIND = ip_ ) :: iterations = 0 ! # iterations
          INTEGER ( KIND = ip_ ) :: unmatchable = 0 ! # classified unmatchable
        END TYPE MS_auction_inform_type

        TYPE, PUBLIC :: MS_equilib_control_type
          INTEGER ( KIND = ip_ ) :: max_iterations = 10
          REAL :: tol = 1.0E-8
        END TYPE MS_equilib_control_type

        TYPE, PUBLIC :: MS_equilib_inform_type
          INTEGER ( KIND = ip_ ) :: flag
          INTEGER ( KIND = ip_ ) :: stat
          INTEGER ( KIND = ip_ ) :: iterations
        END TYPE MS_equilib_inform_type

        TYPE, PUBLIC :: MS_hungarian_control_type
          LOGICAL :: scale_if_singular = .FALSE.
        END TYPE MS_hungarian_control_type

        TYPE, PUBLIC :: MS_hungarian_inform_type
          INTEGER ( KIND = ip_ ) :: flag
          INTEGER ( KIND = ip_ ) :: stat
          INTEGER ( KIND = ip_ ) :: matched
        END TYPE MS_hungarian_inform_type

      CONTAINS

!-*-*-*-  G A L A H A D -  hungarian_scale_sym_int32  S U B R O U T I N E  -*-*-

        SUBROUTINE hungarian_scale_sym_int32( n, ptr, row, val, scaling,       &
                                              control, inform, match )

!  matching-based scaling obtained using Hungarian algorithm (symmetric)

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  order of system

!  column pointers of A

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_hungarian_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_hungarian_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( n ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr64

        ALLOCATE ( ptr64( n + 1 ), STAT=inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF
        ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

        CALL hungarian_scale_sym_int64( n, ptr64, row, val, scaling, control,  &
                                        inform, match=match )
        RETURN

        END SUBROUTINE hungarian_scale_sym_int32

!-*-*-*-  G A L A H A D -  hungarian_scale_sym_int64  S U B R O U T I N E  -*-*-

        SUBROUTINE hungarian_scale_sym_int64( n, ptr, row, val, scaling,       &
                                              control, inform, match )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  order of system

 !  column pointers of A

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A ( lower triangle )

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A ( in same order as in row ).

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_hungarian_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_hungarian_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( n ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: perm
        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: rscaling, cscaling

        inform%flag = 0 !  Initialize to success

        ALLOCATE ( rscaling( n ), cscaling( n ), STAT=inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        IF ( PRESENT( match ) ) THEN
          CALL hungarian_wrapper( .TRUE., n, n, ptr, row, val, match,          &
                                  rscaling, cscaling, control, inform )
        ELSE
          ALLOCATE ( perm( n ), STAT=inform%stat )
          IF ( inform%stat /= 0 ) THEN
            inform%flag = ERROR_ALLOCATION
            RETURN
          END IF
          CALL hungarian_wrapper( .TRUE., n, n, ptr, row, val, perm,           &
                                  rscaling, cscaling, control, inform )
        END IF
        scaling( 1 : n ) = EXP( ( rscaling( 1 : n ) + cscaling( 1 : n ) ) / 2 )
        RETURN

        END SUBROUTINE hungarian_scale_sym_int64

!-*-*-  G A L A H A D - hungarian_scale_unsym_int32  S U B R O U T I N E  -*-*-

        SUBROUTINE hungarian_scale_unsym_int32( m, n, ptr, row, val,           &
                                                rscaling, cscaling,            &
                                                control, inform, match )

!  matching-based scaling obtained using Hungarian algorithm (unsymmetric)

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols

!  column pointers of A

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A ( in same order as in row ).

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_hungarian_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_hungarian_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr64

!  copy from int32 to int64

        ALLOCATE ( ptr64( n + 1 ), STAT=inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF
        ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

        CALL hungarian_scale_unsym_int64( m, n, ptr64, row, val, rscaling,     &
                                          cscaling, control, inform,           &
                                          match = match )
        RETURN

        END SUBROUTINE hungarian_scale_unsym_int32

!-*-*-  G A L A H A D - hungarian_scale_unsym_int64  S U B R O U T I N E  -*-*-

        SUBROUTINE hungarian_scale_unsym_int64( m, n, ptr, row, val, rscaling, &
        cscaling, control, inform, match )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols

!  column pointers of A

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_hungarian_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_hungarian_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: perm

        inform%flag = 0 !  Initialize to success

!  call main routine

        IF ( PRESENT( match ) ) THEN
          CALL hungarian_wrapper( .FALSE., m, n, ptr, row, val, match,         &
                                  rscaling, cscaling, control, inform )
        ELSE
          ALLOCATE ( perm( m ), STAT=inform%stat )
          IF ( inform%stat /= 0 ) THEN
            inform%flag = ERROR_ALLOCATION
            RETURN
          END IF
          CALL hungarian_wrapper( .FALSE., m, n, ptr, row, val, perm,          &
                                  rscaling, cscaling, control, inform )
        END IF

!  apply post processing

        rscaling( 1 : m ) = EXP( rscaling( 1 : m ) )
        cscaling( 1 : n ) = EXP( cscaling( 1 : n ) )
        RETURN

        END SUBROUTINE hungarian_scale_unsym_int64

!-*-*-*-  G A L A H A D - auction_scale_sym_int32  S U B R O U T I N E  -*-*-*-

        SUBROUTINE auction_scale_sym_int32( n, ptr, row, val, scaling,         &
                                            control, inform, match )

!  auction algorithm to get a scaling, then symmetrize it

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  order of system

!  column pointers of A

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_auction_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_auction_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( n ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr64

        ALLOCATE ( ptr64( n + 1 ), STAT=inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF
        ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

        CALL auction_scale_sym_int64( n, ptr64, row, val, scaling, control,    &
                                      inform, match = match )
        RETURN

        END SUBROUTINE auction_scale_sym_int32

!-*-*-*-  G A L A H A D - auction_scale_sym_int64  S U B R O U T I N E  -*-*-*-

        SUBROUTINE auction_scale_sym_int64( n, ptr, row, val, scaling,         &
                                            control, inform, match )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  order of system

!  column pointers of A

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_auction_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_auction_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( n ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: perm
        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: rscaling, cscaling

        inform%flag = 0 !  Initialize to sucess

!  allocate memory

        ALLOCATE ( rscaling( n ), cscaling( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

!  call unsymmetric implementation with flag to expand half matrix

        IF ( PRESENT( match ) ) THEN
          CALL auction_match( .TRUE., n, n, ptr, row, val, match, rscaling,    &
                              cscaling, control, inform )
        ELSE
          ALLOCATE ( perm( n ), STAT = inform%stat )
          IF ( inform%stat /= 0 ) THEN
            inform%flag = ERROR_ALLOCATION
            RETURN
          END IF
          CALL auction_match( .TRUE., n, n, ptr, row, val, perm, rscaling,     &
                              cscaling, control, inform )
        END IF

!  average rscaling and cscaling to get symmetric scaling

        scaling( 1 : n )                                                       &
          = EXP( ( rscaling( 1 : n ) + cscaling( 1 : n ) ) / two )
        RETURN

        END SUBROUTINE auction_scale_sym_int64

!-*-*-*-  G A L A H A D - auction_scale_unsym_int32  S U B R O U T I N E  -*-*-

        SUBROUTINE auction_scale_unsym_int32( m, n, ptr, row, val, rscaling,   &
                                              cscaling, control, inform, match )

!  auction algorithm to get a scaling (unsymmetric version)

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column pointers of A

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_auction_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_auction_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr64

        ALLOCATE ( ptr64( n + 1 ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF
        ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

        CALL auction_scale_unsym_int64( m, n, ptr64, row, val, rscaling, &
          cscaling, control, inform, match=match )
        RETURN

        END SUBROUTINE auction_scale_unsym_int32

!-*-*-*-  G A L A H A D - auction_scale_unsym_int64  S U B R O U T I N E  -*-*-

        SUBROUTINE auction_scale_unsym_int64( m, n, ptr, row, val, rscaling,   &
                                              cscaling, control, inform, match )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column pointers of A

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_auction_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_auction_inform_type ), INTENT ( OUT ) :: inform
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), OPTIONAL,                     &
                                                 INTENT ( OUT ) :: match

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: perm

        inform%flag = 0 !  Initialize to sucess

        IF ( PRESENT( match ) ) THEN
          CALL auction_match( .FALSE., m, n, ptr, row, val, match, rscaling,   &
                              cscaling, control, inform )
        ELSE
          ALLOCATE ( perm( m ), STAT = inform%stat )
          IF ( inform%stat /= 0 ) THEN
            inform%flag = ERROR_ALLOCATION
            RETURN
          END IF
          CALL auction_match( .FALSE., m, n, ptr, row, val, perm, rscaling,    &
                              cscaling, control, inform )
        END IF

        rscaling( 1 : m ) = exp( rscaling( 1 : m ) )
        cscaling( 1 : n ) = exp( cscaling( 1 : n ) )
        RETURN

        END SUBROUTINE auction_scale_unsym_int64

!-*-*-*-  G A L A H A D - equilib_scale_sym_int32  S U B R O U T I N E  -*-*-*-

        SUBROUTINE equilib_scale_sym_int32( n, ptr, row, val, scaling,         &
                                            control, inform )

!  infinity-norm equilibriation algorithm (symmetric version)

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  order of system

!  column pointers of A

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_equilib_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_equilib_inform_type ), INTENT ( OUT ) :: inform

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr64

        ALLOCATE ( ptr64( n + 1 ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF
        ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

        CALL equilib_scale_sym_int64( n, ptr64, row, val, scaling, control,    &
                                      inform )
        RETURN

        END SUBROUTINE equilib_scale_sym_int32

!-*-*-*-  G A L A H A D - equilib_scale_sym_int64  S U B R O U T I N E  -*-*-*-

        SUBROUTINE equilib_scale_sym_int64( n, ptr, row, val, scaling,         &
                                            control, inform )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  order of system

!  column pointers of A

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_equilib_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_equilib_inform_type ), INTENT ( OUT ) :: inform

        inform%flag = 0 !  Initialize to sucess

        CALL inf_norm_equilib_sym( n, ptr, row, val, scaling, control, inform )
        RETURN

        END SUBROUTINE equilib_scale_sym_int64

!-*-*-*-  G A L A H A D - equilib_scale_unsym_int32  S U B R O U T I N E  -*-*-

        SUBROUTINE equilib_scale_unsym_int32( m, n, ptr, row, val, rscaling,  &
                                              cscaling, control, inform )
        IMPLICIT NONE

!  the infinity-norm equilibriation algorithm (unsymmetric version)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols

!  column pointers of A

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_equilib_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_equilib_inform_type ), INTENT ( OUT ) :: inform

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr64

        ALLOCATE ( ptr64( n + 1 ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF
        ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

        CALL equilib_scale_unsym_int64( m, n, ptr64, row, val, rscaling,       &
                                        cscaling, control, inform )
        RETURN

        END SUBROUTINE equilib_scale_unsym_int32

!-*-*-*-  G A L A H A D - equilib_scale_unsym_int64  S U B R O U T I N E  -*-*-

        SUBROUTINE equilib_scale_unsym_int64( m, n, ptr, row, val, rscaling,   &
                                              cscaling, control, inform )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols

!  column pointers of A

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row indices of A (lower triangle)

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( * )

!  entries of A (in same order as in row)

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( * )
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_equilib_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_equilib_inform_type ), INTENT ( OUT ) :: inform

        inform%flag = 0 !  Initialize to sucess

        CALL inf_norm_equilib_unsym( m, n, ptr, row, val, rscaling, cscaling,  &
                                     control, inform )
        RETURN

        END SUBROUTINE equilib_scale_unsym_int64

!-*-*-*-  G A L A H A D - inf_norm_equilib_sym  S U B R O U T I N E  -*-*-*-

        SUBROUTINE inf_norm_equilib_sym( n, ptr, row, val, scaling, control,   &
                                         inform )

!  Inf-norm Equilibriation Algorithm 1 of "A Symmetry Preserving Algorithm 
!  for Matrix Scaling", Philip Knight, Daniel Ruiz and Bora Ucar
!  INRIA Research Report 7552 (Novemeber 2012). This is a complete
!  reimplementation of the algorithm used in HSL's MC77 to allow it
!  to be released as open source. This version preserves symmetry

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                &
                               INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                   &
                             INTENT ( IN ) :: val
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scaling
        TYPE ( MS_equilib_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_equilib_inform_type ), INTENT ( INOUT ) :: inform

        INTEGER ( KIND = ip_ ) :: itr, r, c
        INTEGER ( KIND = long_ ) :: j
        REAL ( KIND = rp_ ) :: v
        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: maxentry

        ALLOCATE ( maxentry( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        scaling( 1 : n ) = one

!  find the maximum entry in each row and column, recalling that the matrix
!  is symmetric, but we only have stored half

        DO itr = 1, control%max_iterations
          maxentry( 1 : n ) = zero
          DO c = 1, n
            DO j = ptr( c ), ptr( c + 1 ) - 1
              r = row( j )
              v = ABS( scaling( r ) * val( j ) * scaling( c ) )
              maxentry( r ) = max( maxentry( r ), v )
              maxentry( c ) = max( maxentry( c ), v )
            END DO
          END DO

!  update scaling ( but beware empty cols )

          WHERE ( maxentry( 1 : n ) > zero )                                   &
            scaling( 1 : n ) = scaling( 1 : n ) / SQRT( maxentry( 1 : n ) )

!  test for convergence

          IF ( MAXVAL( ABS( one - maxentry( 1 : n ) ) ) < control%tol ) EXIT
        END DO
        inform%iterations = itr - 1
        RETURN

        END SUBROUTINE inf_norm_equilib_sym

!-*-*-*-  G A L A H A D - inf_norm_equilib_unsym  S U B R O U T I N E  -*-*-*-

        SUBROUTINE inf_norm_equilib_unsym( m, n, ptr, row, val, rscaling,      &
                                           cscaling, control, inform )

!  Inf-norm Equilibriation Algorithm 1 of "A Symmetry Preserving Algorithm 
!  for Matrix Scaling", Philip Knight, Daniel Ruiz and Bora Ucar
!  INRIA Research Report 7552 (Novemeber 2012). This is a complete
!  reimplementation of the algorithm used in HSL's MC77 to allow it
!  to be released as open source. This version produces unsymmetric scalings

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                &
                                INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                   &
                             INTENT ( IN ) :: val
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_equilib_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_equilib_inform_type ), INTENT ( INOUT ) :: inform

        INTEGER ( KIND = ip_ ) :: itr, r, c
        INTEGER ( KIND = long_ ) :: j
        REAL ( KIND = rp_ ) :: v
        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: rmaxentry
        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: cmaxentry

        ALLOCATE ( rmaxentry( m ), cmaxentry( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        rscaling( 1 : m ) = one
        cscaling( 1 : n ) = one

!  find the maximum entry in each row and column, recalling that the matrix
!  is symmetric, but we only have stored half

        DO itr = 1, control%max_iterations
          rmaxentry( 1 : m ) = zero
          cmaxentry( 1 : n ) = zero
          DO c = 1, n
            DO j = ptr( c ), ptr( c + 1 ) - 1
              r = row( j )
              v = ABS( rscaling( r ) * val( j ) * cscaling( c ) )
              rmaxentry( r ) = MAX( rmaxentry( r ), v )
              cmaxentry( c ) = MAX( cmaxentry( c ), v )
            END DO
          END DO

!  update scaling (but beware empty cols)

          WHERE ( rmaxentry( 1 : m ) > 0 )                                     &
            rscaling( 1 : m ) = rscaling( 1 : m ) / SQRT( rmaxentry( 1 : m ) )
          WHERE ( cmaxentry( 1 : n ) > 0 )                                     &
            cscaling( 1 : n ) = cscaling( 1 : n ) / SQRT( cmaxentry( 1 : n ) )

!  test for convergence

          IF ( MAXVAL( ABS( one - rmaxentry( 1 : m ) ) )                       &
               < control%tol .AND.                                             &
               MAXVAL( ABS( one - cmaxentry( 1 : n ) ) )                       &
               < control%tol ) EXIT
        END DO
        inform%iterations = itr - 1
        RETURN

        END SUBROUTINE inf_norm_equilib_unsym

!-*-*-*-  G A L A H A D - hungarian_wrapper  S U B R O U T I N E  -*-*-*-

        SUBROUTINE hungarian_wrapper( sym, m, n, ptr, row, val, match,         &
                                      rscaling, cscaling, control, inform )

!  Hungarian Algorithm implementation, adapted from HSL_MC64 v2.3.1.
!  This subroutine wraps the core algorithm of hungarian_match(). It provides
!  pre- and post-processing to transform a maximum product assignment to a
!  minimum sum assignment problem (and back again). It also has post-processing
!  to handle the case of a structurally singular matrix as per Duff and Pralet
!  (though the efficacy of such an approach is disputed!)
!
        IMPLICIT NONE
        LOGICAL, INTENT ( IN ) :: sym
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( * ), INTENT ( IN ) :: val
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), INTENT ( OUT ) :: match
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_hungarian_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_hungarian_inform_type ), INTENT ( OUT ) :: inform

        INTEGER ( KIND = long_ ), ALLOCATABLE :: ptr2( : )
        INTEGER ( KIND = ip_ ), ALLOCATABLE :: row2( : ), iw( : ), cperm( : )
        INTEGER ( KIND = ip_ ), ALLOCATABLE :: new_to_old( : ), old_to_new( : )
        REAL ( KIND = rp_ ), ALLOCATABLE :: val2( : ), cmax( : ), cscale( : )
        REAL ( KIND = rp_ ), ALLOCATABLE :: dualu( : ), dualv( : )
        REAL ( KIND = rp_ ) :: colmax
        INTEGER ( KIND = ip_ ) :: i, j, nn, jj, k
        INTEGER ( KIND = long_ ) :: j1, j2, jlong, klong, ne
        REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0

        inform%flag = 0
        inform%stat = 0
        ne = ptr( n + 1 ) - 1

!  reset ne for the expanded symmetric matrix

        ne = 2 * ne

!  expand matrix, drop explicit zeroes and take log absolute values

        ALLOCATE ( ptr2( n + 1 ), row2( ne ), val2( ne ), iw( 5 * n ),         &
                   dualu( m ), dualv( n ), cmax( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        klong = 1
        DO i = 1, n
          ptr2( i ) = klong
          DO jlong = ptr( i ), ptr( i + 1 ) - 1
            IF ( val( jlong ) == zero ) CYCLE
            row2( klong ) = row( jlong )
            val2( klong ) = abs( val( jlong ) )
            klong = klong + 1
          END DO

!  following log is seperated from above loop to expose expensive
!  log operation to vectorization.

          val2( ptr2( i ) : klong - 1 ) = LOG( val2( ptr2( i ) : klong - 1 ) )
        END DO
        ptr2( n + 1 ) = klong
        IF ( sym ) CALL MU_half_to_full( n, row2, ptr2, iw, a = val2 )

!  compute column maxima

        DO i = 1, n
          colmax = MAXVAL( val2( ptr2( i ) : ptr2( i + 1 ) - 1 ) )
          cmax( i ) = colmax
          val2( ptr2( i ) : ptr2( i + 1 ) - 1 )                                &
            = colmax - val2( ptr2( i ) : ptr2( i + 1 ) - 1 )
        END DO

        CALL hungarian_match( m, n, ptr2, row2, val2, match, inform%matched,   &
                              dualu, dualv, inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

!  singular matrix

        IF ( inform%matched /= min( m,n ) ) THEN

!  just issue warning then continue

          IF ( control%scale_if_singular ) THEN
            inform%flag = WARNING_SINGULAR

!  issue error and return identity scaling

          ELSE
            inform%flag = ERROR_SINGULAR
            rscaling( 1 : m ) = 0
            cscaling( 1 : n ) = 0
          END IF
        END IF

!  unsymmetric or symmetric and full rank. Note that in this case m = n

        IF ( .NOT. sym .OR. inform%matched == n ) THEN
          rscaling( 1 : m ) = dualu( 1 : m )
          cscaling( 1 : n ) = dualv( 1 : n ) - cmax( 1 : n )
          CALL match_postproc( m, n, ptr, row, val, rscaling, cscaling,        &
                               inform%matched, match, inform%flag, inform%stat )
          RETURN
        END IF

!  if we reach this point then A is structually rank deficient.
!  As matching may not involve full set of rows and columns, but we need
!  a symmetric matching/scaling, we cannot just return the current matching.
!  Instead, we build a full rank submatrix and call matching on it.

        ALLOCATE ( old_to_new( n ), new_to_old( n ), cperm( n ),               &
                   STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        j = inform%matched + 1
        k = 0
        DO i = 1, m

!  row i is not part of the matching

          IF ( match( i ) < 0 ) THEN
            old_to_new( i ) = - j
            j = j + 1
          ELSE
            k = k + 1

!  old_to_new(i) holds the new index for variable i after removal of 
!  singular part and new_to_old(k) is the  original index for k

            old_to_new( i ) = k
            new_to_old( k ) = i
          END IF
        END DO

!  overwrite ptr2, row2 and val2

        ne = 0
        k = 0
        ptr2( 1 ) = 1
        j2 = 1
        DO i = 1, n
          j1 = j2
          j2 = ptr2( i + 1 )

!  skip over unmatched entries

          IF ( match( i ) < 0 ) CYCLE
          k = k + 1
          DO jlong = j1, j2 - 1
            jj = row2( jlong )
            IF ( match( jj ) < 0 ) CYCLE
            ne = ne + 1
            row2( ne ) = old_to_new( jj )
            val2( ne ) = val2( jlong )
          END DO
          ptr2( k + 1 ) = ne + 1
        END DO

!  nn is order of non-singular part

        nn = k
        CALL hungarian_match( nn, nn, ptr2, row2, val2, cperm, inform%matched, &
                              dualu, dualv, inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        DO i = 1, n
          j = old_to_new( i )
          IF ( j < 0 ) THEN
            rscaling( i ) = -huge( rscaling )

!  note: we need to subtract col max using old matrix numbering

          ELSE
            rscaling( i ) = ( dualu( j ) + dualv( j ) - cmax( i ) ) / two
          END IF
        END DO

        match( 1 : n ) = - 1
        DO i = 1, nn
          j = cperm( i )
          match( new_to_old( i ) ) = j
        END DO

        DO i = 1, n
          IF ( match( i ) == - 1 ) THEN
            match( i ) = old_to_new( i )
          END IF
        END DO

!  apply Duff and Pralet correction to unmatched row scalings

        ALLOCATE ( cscale( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

!  for columns i not in the matched set I, set s_i = 1/max_{k in I} |a_ik s_k|
!  with convention that 1/0 = 1

        cscale( 1 : n ) = rscaling( 1 : n )
        DO i = 1, n
          DO jlong = ptr( i ), ptr( i + 1 ) - 1
            k = row( jlong )

!  i not in I, k in I

            IF ( cscale( i ) == - HUGE( rscaling ) .AND.                       &
                 cscale( k ) /= - HUGE( rscaling ) ) THEN
              rscaling( i ) =                                                  &
                MAX( rscaling( i ), LOG( ABS( val( jlong ) ) ) + rscaling( k ) )
            END IF

!  k not in I, i in I

            IF ( cscale( k ) == - HUGE( rscaling ) .AND.                       &
                 cscale( i ) /= - HUGE( rscaling ) ) THEN
              rscaling( k ) =                                                  &
                MAX( rscaling( k ), LOG( ABS( val( jlong ) ) ) + rscaling( i ) )
            END IF
          END DO
        END DO
        DO i = 1, n
          IF ( cscale( i ) /= - HUGE( rscaling ) ) CYCLE !  matched part
          IF ( rscaling( i ) == - HUGE( rscaling ) ) THEN
            rscaling( i ) = zero
          ELSE
            rscaling( i ) = - rscaling( i )
          END IF
        END DO

!  as symmetric, scaling is averaged on return, but rscaling(:) is correct,
!  so just copy to cscaling to fix this

        cscaling( 1 : n ) = rscaling( 1 : n )
        RETURN

        END SUBROUTINE hungarian_wrapper

!-*-*-*-  G A L A H A D - hungarian_init_heurisitic  S U B R O U T I N E  -*-*-

        SUBROUTINE hungarian_init_heurisitic( m, n, ptr, row, val, num, iperm, &
                                              jperm, dualu, d, l, search_from )

!  Subroutine that initialize matching and (row) dual variable into a suitable
!  state for main Hungarian algorithm

!  The heuristic guaruntees that the generated partial matching is optimal
!  on the restriction of the graph to the matched rows and columns

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                &
                                INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                   &
                             INTENT ( IN ) :: val
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: num
        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( INOUT ) :: iperm
        INTEGER ( KIND = long_ ), DIMENSION ( * ), INTENT ( INOUT ) :: jperm
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: dualu

!  d(j) current improvement from  matching in col j

        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: d

!  position of smallest entry of row

        INTEGER ( KIND = long_ ), DIMENSION ( m ), INTENT ( OUT ) :: l

!  position we have reached in current search

        INTEGER ( KIND = long_ ), DIMENSION ( n ),                             &
                                  INTENT ( INOUT ) :: search_from

        INTEGER ( KIND = ip_ ) :: i, i0, ii, j, jj
        INTEGER ( KIND = long_ ) :: k, k0, kk
        REAL ( KIND = rp_ ) :: di, vj
!
!  set up initial matching on smallest entry in each row (as far as possible).
!  Find smallest entry in each col, and record it

        dualu( 1 : m ) = rinf
        l( 1 : m ) = 0
        DO j = 1, n
          DO k = ptr( j ), ptr( j + 1 ) - 1
            i = row( k )
            IF ( val( k ) > dualu( i ) ) CYCLE
            dualu( i ) = val( k ) !  initialize dual variables
            iperm( i ) = j !  record column
            l( i ) = k !  record position in row( : )
          END DO
        END DO

!  loop over rows in turn. If we can match on smallest entry in row (i.e.
!  column not already matched) then do so. Avoid matching on dense columns
!  as this makes Hungarian algorithm take longer

        DO i = 1, m
          j = iperm( i ) !  Smallest entry in row i is ( i,j )
          IF ( j == 0 ) CYCLE !  skip empty rows
          iperm( i ) = 0

!  If we've already matched column j, skip this row. Don't choose cheap 
!  assignment from dense columns

          IF ( jperm( j ) /= 0 ) CYCLE
          IF ( ( ptr( j + 1 )-ptr( j )>m/10 ) .AND. ( m>50 ) ) CYCLE

!  assignment of column j to row i

          num = num + 1
          iperm( i ) = j
          jperm( j ) = l( i )
        END DO

!  if we already have a complete matching, we are done

        IF ( num == MIN( m, n ) ) RETURN


!  scan unassigned columns; improve assignment

        d( 1 : n ) = zero
        search_from( 1 : n ) = ptr( 1 : n )
improve: DO j = 1, n
          IF ( jperm( j ) /= 0 ) CYCLE !  column j already matched

!  column j is empty. Find smallest value of di = a_ij - u_i in column j
!  In case of a tie, prefer first unmatched, then first matched row

          IF ( ptr( j ) > ptr( j + 1 ) - 1 ) CYCLE
          i0 = row( ptr( j ) )
          vj = val( ptr( j ) ) - dualu( i0 )
          k0 = ptr( j )
          DO k = ptr( j ) + 1, ptr( j + 1 ) - 1
            i = row( k )
            di = val( k ) - dualu( i )
            IF ( di > vj ) CYCLE
            IF ( di == vj .AND. di /= rinf ) THEN
              IF ( iperm( i ) /= 0 .OR. iperm( i0 ) == 0 ) CYCLE
            END IF
            vj = di
            i0 = i
            k0 = k
          END DO

!  record value of matching on (i0,j)

          d( j ) = vj

!  if row i is unmatched, then match on (i0,j) immediately

          IF ( iperm( i0 ) == 0 ) THEN
            num = num + 1
            jperm( j ) = k0
            iperm( i0 ) = j
            search_from( j ) = k0 + 1
            CYCLE
          END IF

!  otherwise, row i is matched. Consider all rows i in column j that tie
!  for this vj value. Such a row currently matches on (i,jj). Scan column
!  jj looking for an unmatched row ii that improves value of matching. If
!  one exists, then augment along length 2 path (i,j)->(ii,jj)

          DO k = k0, ptr( j + 1 ) - 1
            i = row( k )
            IF ( ( val( k )-dualu( i ) ) > vj ) CYCLE !  not a tie for vj value
            jj = iperm( i )

!  scan remaining part of assigned column jj

            DO kk = search_from( jj ), ptr( jj + 1 ) - 1
              ii = row( kk )
              IF ( iperm( ii )>0 ) CYCLE ! row ii already matched
              IF ( ( val( kk )-dualu( ii ) ) <= d( jj ) ) THEN

!  by matching on ( i,j ) and ( ii,jj ) we do better than existing matching 
!  on (i,jj) alone

                jperm( jj ) = kk
                iperm( ii ) = jj
                search_from( jj ) = kk + 1
                num = num + 1
                jperm( j ) = k
                iperm( i ) = j
                search_from( j ) = k + 1
                CYCLE improve
              END IF
            END DO
            search_from( jj ) = ptr( jj + 1 )
          END DO
          CYCLE
        END DO improve
        RETURN

        END SUBROUTINE hungarian_init_heurisitic

!-*-*-*-  G A L A H A D - hungarian_match  S U B R O U T I N E  -*-*-*-

        SUBROUTINE hungarian_match( m, n, ptr, row, val, iperm, num, dualu,    &
                                    dualv, st )

!  Provides the core Hungarian Algorithm implementation for solving the
!  minimum sum assignment problem as per Duff and Koster.
!
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols

!  cardinality of the matching

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: num

!  column pointers

        INTEGER ( KIND = long_ ), INTENT ( IN ) :: ptr( n + 1 )

!  row pointers

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: row( ptr( n + 1 ) - 1 )

!  matching itself: row i is matched to column iperm(i)

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: iperm( m )

!  value of the entry that corresponds to row(k). All values val(k) must 
!  be non-negative

        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( ptr( n + 1 ) - 1 )

!  dualu(i) is the reduced weight for row(i)

        REAL ( KIND = rp_ ), INTENT ( OUT ) :: dualu( m )

!  dualv(j) is the reduced weight for col(j)

        REAL ( KIND = rp_ ), INTENT ( OUT ) :: dualv( n )
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: st

!  a(jperm(j)) is entry of A for matching in column j

        INTEGER ( KIND = long_ ), ALLOCATABLE, DIMENSION ( : ) :: jperm

!  a(out(i)) is the new entry in a on which we match going along the short
!  path back to the original column

        INTEGER ( KIND = long_ ), ALLOCATABLE, DIMENSION ( : ) :: out

!  pr( i ) is a pointer to the next column along the shortest path back to 
!  the original column

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ) :: pr

!  q(1:qlen) forms a binary heap data structure sorted by d(q(i)) value. 
!  q(low:up) is a list of rows with equal d(i) which is lower or equal to 
!  smallest in the heap. q(up:n) is a list of already visited rows

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ) :: q
        INTEGER ( KIND = long_ ), ALLOCATABLE, DIMENSION ( : ) :: longwork

!  l(:) is an inverse of q(:)

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ) :: l

!  d(i) is current shortest  distance to row i from current column 
!  (d_i from Fig 4.1 of Duff and Koster paper)

        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION ( : ) :: d

        INTEGER ( KIND = ip_ ) :: i, j, jj, jord, q0, qlen, jdum, jsp
        INTEGER ( KIND = ip_ ) :: kk, up, low, lpos
        INTEGER ( KIND = long_ ) :: klong, isp
        REAL ( KIND = rp_ ) :: csp, di, dmin, dnew, dq0, vj

!  initialization

        ALLOCATE ( jperm( n ), out( n ), pr( n ), q( m ), longwork( m ),       &
                   l( m ), d( MAX( m, n ) ), STAT = st )
        IF ( st /= 0 ) RETURN
        num = 0
        iperm( 1 : m ) = 0
        jperm( 1 : n ) = 0

        CALL hungarian_init_heurisitic( m, n, ptr, row, val, num,              &
                                        iperm, jperm, dualu, d, longwork, out )

 !  if we found a complete matching, done

        IF ( num == MIN( m, n ) ) GO TO 110

!  repeatedly find augmenting paths until all columns are included in the
!  matching. At every step the current matching is optimal on the restriction
!  of the graph to currently matched rows and columns

!  main loop ... each pass round this loop is similar to Dijkstra's
!  algorithm for solving the single source shortest path problem

        d( 1 : m ) = rinf
        l( 1 : m ) = 0
        isp = - 1
        jsp = - 1 !  initalize to avoid "may be used unitialized" warning

!  jord is next unmatched column

        DO jord = 1, n
          IF ( jperm( jord ) /= 0 ) CYCLE

!  dmin is the length of shortest path in the tree

          dmin = rinf
          qlen = 0
          low = m + 1
          up = m + 1

!  csp is the cost of the shortest augmenting path to unassigned row
!  row( isp ). The corresponding column index is jsp.

          csp = rinf

!  build shortest path tree starting from unassigned column ( root ) jord

          j = jord
          pr( j ) = - 1

!  scan column j

          DO klong = ptr( j ), ptr( j + 1 ) - 1
            i = row( klong )
            dnew = val( klong ) - dualu( i )
            IF ( dnew >= csp ) CYCLE
            IF ( iperm( i ) == 0 ) THEN
              csp = dnew
              isp = klong
              jsp = j
            ELSE
              IF ( dnew < dmin ) dmin = dnew
              d( i ) = dnew
              qlen = qlen + 1
              longwork( qlen ) = klong
            END IF
          END DO

!  Initialize heap Q and Q2 with rows held in longwork(1:qlen)

          q0 = qlen
          qlen = 0
          DO kk = 1, q0
            klong = longwork( kk )
            i = row( klong )
            IF ( csp <= d( i ) ) THEN
              d( i ) = rinf
              CYCLE
            END IF
            IF ( d( i ) <= dmin ) THEN
              low = low - 1
              q( low ) = i
              l( i ) = low
            ELSE
              qlen = qlen + 1
              l( i ) = qlen
              CALL heap_update( i, m, q, d, l )
            END IF

!  update tree

            jj = iperm( i )
            out( jj ) = klong
            pr( jj ) = j
          END DO

          DO jdum = 1, num

!  if Q2 is empty, extract rows from Q

            IF ( low == up ) THEN
              IF ( qlen == 0 ) EXIT
              i = q( 1 ) !  Peek at top of heap
              IF ( d( i )>=csp ) EXIT
              dmin = d( i )

!  extract all paths that have length dmin and store in q(low:up-1)

              DO WHILE ( qlen>0 )
                i = q( 1 ) !  Peek at top of heap
                IF ( d( i )>dmin ) EXIT
                i = heap_pop( qlen, m, q, d, l )
                low = low - 1
                q( low ) = i
                l( i ) = low
              END DO
            END IF

!  q0 is row whose distance d(q0) to the root is smallest

            q0 = q( up - 1 )
            dq0 = d( q0 )

!  exit loop if path to q0 is longer than the shortest augmenting path

            IF ( dq0 >= csp ) EXIT
            up = up - 1

!  scan column that matches with row q0

            j = iperm( q0 )
            vj = dq0 - val( jperm( j ) ) + dualu( q0 )
            DO klong = ptr( j ), ptr( j + 1 ) - 1
              i = row( klong )
              IF ( l( i )>=up ) CYCLE

!  dnew is new cost

              dnew = vj + val( klong ) - dualu( i )

!  do not update d(i) if dnew is greater or equal to the cost of shortest path

              IF ( dnew >= csp ) CYCLE

!  row i is unmatched; update shortest path info

              IF ( iperm( i ) == 0 ) THEN
                csp = dnew
                isp = klong
                jsp = j

!  row i is matched; do not update d(i) if dnew is larger

              ELSE
                di = d( i )
                IF ( di <= dnew ) CYCLE
                IF ( l( i )>=low ) CYCLE
                d( i ) = dnew
                IF ( dnew <= dmin ) THEN
                  lpos = l( i )
                  IF ( lpos /= 0 ) CALL heap_delete( lpos, qlen, m, q, d, l )
                  low = low - 1
                  q( low ) = i
                  l( i ) = low
                ELSE
                  IF ( l( i ) == 0 ) THEN
                    qlen = qlen + 1
                    l( i ) = qlen
                  END IF
                  CALL heap_update( i, m, q, d, l ) !  d( i ) has changed
                END IF

!  update tree

                jj = iperm( i )
                out( jj ) = klong
                pr( jj ) = j
              END IF
            END DO
          END DO

!  no augmenting path is found

!         IF ( csp == rinf ) GO TO 100
          IF ( csp /= rinf ) THEN

!  find augmenting path by tracing backward in pr; update iperm, jperm

          num = num + 1
          i = row( isp )
          iperm( i ) = jsp
          jperm( jsp ) = isp
          j = jsp
          DO jdum = 1, num
            jj = pr( j )
            IF ( jj == - 1 ) EXIT
            klong = out( j )
            i = row( klong )
            iperm( i ) = jj
            jperm( jj ) = klong
            j = jj
          END DO

!  Update U for rows in q(up:m)

          DO kk = up, m
            i = q( kk )
            dualu( i ) = dualu( i ) + d( i ) - csp
          END DO
          END IF
!100       CONTINUE

          DO kk = low, m
            i = q( kk )
            d( i ) = rinf
            l( i ) = 0
          END DO
          DO kk = 1, qlen
            i = q( kk )
            d( i ) = rinf
            l( i ) = 0
          END DO

        END DO !  End of main loop

!  set dual column variables

110     CONTINUE
        DO j = 1, n
          klong = jperm( j )
          IF ( klong /= 0 ) THEN
            dualv( j ) = val( klong ) - dualu( row( klong ) )
          ELSE
            dualv( j ) = zero
          END IF
        END DO

!  zero dual row variables for unmatched rows

        WHERE ( iperm( 1 : m ) == 0 ) dualu( 1 : m ) = zero

        RETURN

        END SUBROUTINE hungarian_match

!-*-*-*-  G A L A H A D - h e a p _ u p d a t e  S U B R O U T I N E  -*-*-*-

        SUBROUTINE heap_update( idx, n, q, val, l )

!  value associated with index i has decreased, update position in heap
!  as approriate.

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: idx
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: q( n )
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: l( n )
        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( n )

        INTEGER ( KIND = ip_ ) :: pos, parent_pos
        INTEGER ( KIND = ip_ ) :: parent_idx
        REAL ( KIND = rp_ ) :: v

!  get current position of i in heap

        pos = l( idx )
        IF ( pos <= 1 ) THEN

!  idx is already at root of heap, but set q as it may have only just
!  been inserted

          q( pos ) = idx
          RETURN
        END IF

!  keep trying to move i towards root of heap until it can't go any further

        v = val( idx )
        DO WHILE ( pos > 1 ) !  while not at root of heap
          parent_pos = pos / 2
          parent_idx = q( parent_pos )

!  if parent is better than idx, stop moving

          IF ( v>=val( parent_idx ) ) EXIT

!  otherwise, swap idx and parent

          q( pos ) = parent_idx
          l( parent_idx ) = pos
          pos = parent_pos
        END DO

!  finally set idx in the place it reached

        q( pos ) = idx
        l( idx ) = pos
        RETURN

        END SUBROUTINE heap_update

!-*-*-*-  G A L A H A D - h e a p _ p o p   F U N C T I O N  -*-*-*-

        INTEGER ( ip_ ) FUNCTION heap_pop( qlen, n, q, val, l )

!  the root node is deleted from the binary heap.

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: qlen
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: q( n )
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: l( n )
        REAL ( KIND = rp_ ), INTENT ( IN ) :: val( n )

!  return value is the old root of the heap

        heap_pop = q( 1 )

!  delete the root

        CALL heap_delete( 1_ip_, qlen, n, q, val, l )
        RETURN

        END FUNCTION heap_pop

        SUBROUTINE heap_delete( pos0, qlen, n, q, d, l )

!  delete element in poisition pos0 from the heap

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ) :: pos0, qlen, n
        INTEGER ( KIND = ip_ ) :: q( n ), l( n )
        REAL ( KIND = rp_ ) :: d( n )

        INTEGER ( KIND = ip_ ) :: idx, pos, parent, child, qk
        REAL ( KIND = rp_ ) :: dk, dr, v

!  if we are trying to remove the last item, just delete it

        IF ( qlen == pos0 ) THEN
          qlen = qlen - 1
          RETURN
        END IF

!  replace index in position pos0 with last item and fix heap property

        idx = q( qlen )
        v = d( idx )
        qlen = qlen - 1 !  shrink heap
        pos = pos0 !  pos is current position of node I in the tree

!  move up if appropriate

        IF ( pos > 1 ) THEN
          DO
            parent = pos / 2
            qk = q( parent )
            IF ( v >= d( qk ) ) EXIT
            q( pos ) = qk
            l( qk ) = pos
            pos = parent
            IF ( pos <= 1 ) EXIT
          END DO
        END IF
        q( pos ) = idx
        l( idx ) = pos

!  item moved up, hence doesn't need to move down

        IF ( pos /= pos0 ) RETURN

!  otherwise, move item down

        DO
          child = 2 * pos
          IF ( child > qlen ) EXIT
          dk = d( q( child ) )
          IF ( child < qlen ) THEN
            dr = d( q( child + 1 ) )
            IF ( dk > dr ) THEN
              child = child + 1
              dk = dr
            END IF
          END IF
          IF ( v <= dk ) EXIT
          qk = q( child )
          q( pos ) = qk
          l( qk ) = pos
          pos = child
        END DO
        q( pos ) = idx
        l( idx ) = pos
        RETURN

        END SUBROUTINE heap_delete

!-*-*-*-  G A L A H A D - auction_match_core  S U B R O U T I N E  -*-*-*-

        SUBROUTINE auction_match_core( m, n, ptr, row, val, match,             &
                                       dualu, dualv, control, inform )

!  an implementation of the auction algorithm to solve the assignment problem
!  i.e. max_M sum_{(i,j)\in M} a_{ij}, where M is a matching.
!  The dual variables u_i for row i and v_j for col j can be used to find
!  a good scaling after postprocessing. We are aiming for:
!  a_ij - u_i - v_j  ==  0    if (i,j) in M
!  a_ij - u_i - v_j  <=  0    otherwise

!  Motivation of algorithm:
!  Each unmatched column bids for its preferred row. Best bid wins.
!  Prices (dual variables) are updated to reflect cost of using 2nd best 
!  instead for future auction rounds.
!  i.e. Price of using entry (i,j) is a_ij - u_i

!  In this implementation, only one column is bidding in each phase. This is
!  largely equivalent to the motivation above but allows for faster progression
!  as the same row can move through multiple partners during a single pass
!  through the matrix

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                &
                                INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                   &
                             INTENT ( IN ) :: val

!  match(j) = i => column j matched to row i

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( OUT ) :: match

!  row dual variables

        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: dualu

!  column dual variables

        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( INOUT ) :: dualv
        TYPE ( MS_auction_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_auction_inform_type ), INTENT ( INOUT ) :: inform

!  inverse of match

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: owner

!  The list next(1:tail) is the search space of unmatched columns
!  this is overwritten as we proceed such that next(1:insert) is the
!  search space for the subsequent iteration

        INTEGER ( KIND = ip_ ) :: tail, insert
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: next
        INTEGER ( KIND = ip_ ) :: unmatched ! current number of unmatched cols

        INTEGER ( KIND = ip_ ) :: itr, minmn
        INTEGER ( KIND = ip_ ) :: i, k
        INTEGER ( KIND = long_ ) :: j
        INTEGER ( KIND = ip_ ) :: col, cptr, bestr
        REAL :: ratio
        REAL ( KIND = rp_ ) :: u, bestu, bestv

        REAL ( KIND = rp_ ) :: eps !  minimum improvement

 !  number of unmatched cols on previous iteration

        INTEGER ( KIND = ip_ ) :: prev

!  number of iterations where #unmatched cols has been constant

        INTEGER ( KIND = ip_ ) :: nunchanged

        inform%flag = 0
        inform%unmatchable = 0

!  allocate memory

        ALLOCATE ( owner( m ), next( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

!  set everything as unmatched

        minmn = MIN( m, n )
        unmatched = minmn
        match( 1 : n ) = 0 !  0 = unmatched, - 1 = unmatched + ineligible
        owner( 1 : m ) = 0
        dualu( 1 : m ) = 0

!  dualv is set for each column as it becomes matched, otherwise we use
!  the value supplied on input (calculated as something sensible during
!  preprocessing)

!  set up monitoring of progress

        prev = - 1
        nunchanged = 0

!  initially all columns are unmatched

        tail = n
        DO i = 1, n
          next( i ) = i
        END DO

!  iterate until we run out of unmatched buyers

        eps = control%eps_initial
        DO itr = 1, control%max_iterations
          IF ( unmatched == 0 ) EXIT !  nothing left to match

!  book-keeping to determine number of iterations with no change

          IF ( unmatched /= prev ) nunchanged = 0
          prev = unmatched
          nunchanged = nunchanged + 1

!  test if we satisfy termination conditions

          ratio = REAL( minmn - unmatched ) / REAL( minmn )
          IF ( nunchanged >= control%max_unchanged( 1 ) .AND.                  &
               ratio >= control%min_proportion( 1 ) ) EXIT
          IF ( nunchanged >= control%max_unchanged( 2 ) .AND.                  &
               ratio >= control%min_proportion( 2 ) ) EXIT
          IF ( nunchanged >= control%max_unchanged( 3 ) .AND.                  &
               ratio >= control%min_proportion( 3 ) ) EXIT

!  update epsilon scaling

          eps = MIN( one, eps + one / REAL( n + 1, rp_ ) )

!  now iterate over all unmatched entries listed in next(1:tail)
!  As we progress, build list for next iteration in next(1:insert)

          insert = 0
          DO cptr = 1, tail
            col = next( cptr )
            IF ( match( col ) /= 0 ) CYCLE !  already matched or ineligible

!  empty col (only ever fails on first iteration - not put in next(1:insert) 
!  thereafter)

            IF ( ptr( col ) == ptr( col + 1 ) ) CYCLE 

!  find best value of a_ij - u_i for current column. This occurs for i=bestr 
!  with value bestu second best value stored as bestv

            j = ptr( col )
            bestr = row( j )
            bestu = val( j ) - dualu( bestr )
            bestv = - HUGE( bestv )
            DO j = ptr( col ) + 1, ptr( col + 1 ) - 1
              u = val( j ) - dualu( row( j ) )
              IF ( u > bestu ) THEN
                bestv = bestu
                bestr = row( j )
                bestu = u
              ELSE IF ( u > bestv ) THEN
                bestv = u
              END IF
            END DO
            IF ( bestv == - HUGE( bestv ) ) bestv = zero !  no second best

!  check if matching this column gives us a net benefit

            IF ( bestu > 0 ) THEN

!  there is a net benefit, match column col to row bestr if bestr was 
!  previously matched to col k, unmatch it

              dualu( bestr ) = dualu( bestr ) + bestu - bestv + eps
              dualv( col ) = bestv - eps !  satisfy a_ij - u_i - v_j = 0
              match( col ) = bestr
              unmatched = unmatched - 1
              k = owner( bestr )
              owner( bestr ) = col

!  mark column k as unmatched

              IF ( k /= 0 ) THEN
                match( k ) = 0 !  unmatched
                unmatched = unmatched + 1
                insert = insert + 1
                next( insert ) = k
              END IF

!  no net benefit, mark col as ineligible for future consideration

            ELSE
              match( col ) = - 1 !  ineligible
              unmatched = unmatched - 1
              inform%unmatchable = inform%unmatchable + 1
            END IF
          END DO
          tail = insert
        END DO
        inform%iterations = itr - 1

!  we expect unmatched columns to have match( col ) = 0

        WHERE ( match( : ) == - 1 ) match( : ) = 0
        RETURN

        END SUBROUTINE auction_match_core

!-*-*-*-  G A L A H A D - a u c t i o n _ m a t c h  S U B R O U T I N E  -*-*-

        SUBROUTINE auction_match( expand, m, n, ptr, row, val, match,          &
                                  rscaling, cscaling, control, inform )

!  Find a scaling through a matching-based approach using the auction algorithm
!  This subroutine actually performs pre/post-processing around the call to
!  auction_match_core(  ) to actually use the auction algorithm
!
!  This consists of finding a2_ij = 2*maxentry - cmax + log( |a_ij| )
!  where cmax is the log of the absolute maximum in the column
!  and maxentry is the maximum value of cmax-log( |a_ij| ) across entire matrix
!  The cmax-log( |a_ij| ) term converts from max product to max sum problem and
!  normalises scaling across matrix. The 2*maxentry term biases the result
!  towards a high cardinality solution.
!
!  Lower triangle only as input (log( half )+half->full faster than log(full))
!
        IMPLICIT NONE
        LOGICAL, INTENT ( IN ) :: expand
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( * ), INTENT ( IN ) :: val
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), INTENT ( OUT ) :: match
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( OUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: cscaling
        TYPE ( MS_auction_control_type ), INTENT ( IN ) :: control
        TYPE ( MS_auction_inform_type ), INTENT ( INOUT ) :: inform

        INTEGER ( KIND = long_ ), ALLOCATABLE :: ptr2( : )
        INTEGER ( KIND = ip_ ), ALLOCATABLE :: row2( : ), iw( : ), cmatch( : )
        REAL ( KIND = rp_ ), ALLOCATABLE :: val2( : ), cmax( : )
        REAL ( KIND = rp_ ) :: colmax
        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = long_ ) :: jlong, klong
        INTEGER ( KIND = long_ ) :: ne
        REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0
        REAL ( KIND = rp_ ) :: maxentry

        inform%flag = 0

!  reset ne for the expanded symmetric matrix

        ne = ptr( n + 1 ) - 1
        ne = 2*ne

!  expand matrix, drop explicit zeroes and take log absolute values

        ALLOCATE ( ptr2( n + 1 ), row2( ne ), val2( ne ), cmax( n ),           &
                   cmatch( n ), STAT = inform%stat )
        IF ( inform%stat /= 0 ) THEN
          inform%flag = ERROR_ALLOCATION
          RETURN
        END IF

        klong = 1
        DO i = 1, n
          ptr2( i ) = klong
          DO jlong = ptr( i ), ptr( i + 1 ) - 1
            IF ( val( jlong ) == zero ) CYCLE
            row2( klong ) = row( jlong )
            val2( klong ) = ABS( val( jlong ) )
            klong = klong + 1
          END DO

!  following log is seperated from above loop to expose expensive log 
!  operation to vectorization

          val2( ptr2( i ) : klong - 1 ) = log( val2( ptr2( i ) : klong - 1 ) )
        END DO
        ptr2( n + 1 ) = klong
        IF ( expand ) THEN

!  should never get this far with a non-square matrix

          IF ( m /= n ) THEN
            inform%flag = - 99
            RETURN
          END IF
          ALLOCATE ( iw( 5 * n ), STAT = inform%stat )
          IF ( inform%stat /= 0 ) THEN
            inform%flag = ERROR_ALLOCATION
            RETURN
          END IF
          CALL MU_half_to_full( n, row2, ptr2, iw, a = val2 )
        END IF

!  compute column maximums

        DO i = 1, n
          IF ( ptr2( i + 1 ) <= ptr2( i ) ) THEN !  empty col
            cmax( i ) = zero
            CYCLE
          END IF
          colmax = MAXVAL( val2( ptr2( i ) : ptr2( i + 1 ) - 1 ) )
          cmax( i ) = colmax
          val2( ptr2( i ) : ptr2( i + 1 ) - 1 )                                &
            = colmax - val2( ptr2( i ) : ptr2( i + 1 ) - 1 )
        END DO

        maxentry = MAXVAL( val2( 1 : ptr2( n + 1 ) - 1 ) )

!  Use 2*maxentry + 1 to prefer high cardinality matchings (+1 avoids 0 cols)

        maxentry = 2 * maxentry + 1
        val2( 1 : ptr2( n + 1 ) - 1 ) = maxentry - val2( 1 : ptr2( n + 1 ) - 1 )

!  equivalent to scale=1.0 for unmatched cols that core algorithm doesn't visit

        cscaling( 1 : n ) = - cmax( 1 : n )

        CALL auction_match_core( m, n, ptr2, row2, val2, cmatch, rscaling,     &
                                 cscaling, control, inform )
        inform%matched = COUNT( cmatch /= 0 )

!  Calculate an adjustment so row and col scaling similar orders of magnitude
!  and undo pre processing

        rscaling( 1 : m ) = - rscaling( 1 : m ) + maxentry
        cscaling( 1 : n ) = - cscaling( 1 : n ) - cmax( 1 : n )

!  convert row->col matching into col->row one

        match( 1 : m ) = 0
        DO i = 1, n
          IF ( cmatch( i ) == 0 ) CYCLE !  unmatched row
          match( cmatch( i ) ) = i
        END DO
        CALL match_postproc( m, n, ptr, row, val, rscaling, cscaling,          &
                             inform%matched, match, inform%flag, inform%stat )
        RETURN

        END SUBROUTINE auction_match

!-*-*-  G A L A H A D - m a t c h _ p o s t p r o c  S U B R O U T I N E  -*-*-

        SUBROUTINE match_postproc( m, n, ptr, row, val, rscaling, cscaling,    &
                                   nmatch, match, flag, st )
        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                &
                                INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( ptr( n + 1 ) - 1 ),                   &
                             INTENT ( IN ) :: val
        REAL ( KIND = rp_ ), DIMENSION ( m ), INTENT ( INOUT ) :: rscaling
        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( INOUT ) :: cscaling
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: nmatch
        INTEGER ( KIND = ip_ ), DIMENSION ( m ), INTENT ( IN ) :: match
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: flag
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: st

        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = long_ ) :: jlong
        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: rmax, cmax
        REAL ( KIND = rp_ ) :: ravg, cavg, adjust, colmax, v

!  square, just perform post-processing and magnitude adjustment

        IF ( m == n ) THEN
          ravg = SUM( rscaling( 1 : m ) ) / REAL( m, rp_ )
          cavg = SUM( cscaling( 1 : n ) ) / REAL( n, rp_ )
          adjust = ( ravg - cavg ) / 2
          rscaling( 1 : m ) = rscaling( 1 : m ) - adjust
          cscaling( 1 : n ) = cscaling( 1 : n ) + adjust

!  more columns than rows, allocate some workspace

        ELSE IF ( m < n ) THEN
          ALLOCATE ( cmax( n ), STAT = st )
          IF ( st /= 0 ) THEN
            flag = ERROR_ALLOCATION
            RETURN
          END IF

!  first perform post-processing and magnitude adjustment based on match

          ravg = 0
          cavg = 0
          DO i = 1, m
            IF ( match( i ) == 0 ) CYCLE
            ravg = ravg + rscaling( i )
            cavg = cavg + cscaling( match( i ) )
          END DO
          ravg = ravg / REAL( nmatch, rp_ )
          cavg = cavg / REAL( nmatch, rp_ )
          adjust = ( ravg-cavg )/2
          rscaling( 1 : m ) = rscaling( 1 : m ) - adjust
          cscaling( 1 : n ) = cscaling( 1 : n ) + adjust

!  for each unmatched col, scale max entry to 1.0

          DO i = 1, n
            colmax = zero
            DO jlong = ptr( i ), ptr( i + 1 ) - 1
              v = ABS( val( jlong ) ) * EXP( rscaling( row( jlong ) ) )
              colmax = MAX( colmax, v )
            END DO
            IF ( colmax == zero ) THEN
              cmax( i ) = zero
            ELSE
              cmax( i ) = LOG( one / colmax )
            END IF
          END DO
          DO i = 1, m
            IF ( match( i ) == 0 ) CYCLE
            cmax( match( i ) ) = cscaling( match( i ) )
          END DO
          cscaling( 1 : n ) = cmax( 1 : n )

!  More rows than columns, allocate some workspace

        ELSE IF ( n < m ) THEN
          ALLOCATE ( rmax( m ), STAT = st )
          IF ( st /= 0 ) THEN
            flag = ERROR_ALLOCATION
            RETURN
          END IF

!  first perform post-processing and magnitude adjustment based on match
!  also record which rows have been matched

          ravg = 0
          cavg = 0
          DO i = 1, m
            IF ( match( i ) == 0 ) CYCLE
            ravg = ravg + rscaling( i )
            cavg = cavg + cscaling( match( i ) )
          END DO
          ravg = ravg/real( nmatch, rp_ )
          cavg = cavg/real( nmatch, rp_ )
          adjust = ( ravg - cavg ) / two
          rscaling( 1 : m ) = rscaling( 1 : m ) - adjust
          cscaling( 1 : n ) = cscaling( 1 : n ) + adjust

!  find max column-scaled value in each row from unmatched cols

          rmax( : ) = zero
          DO i = 1, n
            DO jlong = ptr( i ), ptr( i + 1 ) - 1
              v = ABS( val( jlong ) ) * EXP( cscaling( i ) )
              rmax( row( jlong ) ) = MAX( rmax( row( jlong ) ), v )
            END DO
          END DO

!  calculate scaling for each row, but overwrite with correct values for
!  matched rows, then copy entire array over rscaling( : )

          DO i = 1, m
            IF ( match( i ) /= 0 ) CYCLE
            IF ( rmax( i ) == zero ) THEN
              rscaling( i ) = zero
            ELSE
              rscaling( i ) = LOG( one / rmax( i ) )
            END IF
          END DO
        END IF
        RETURN

        END SUBROUTINE match_postproc

      END MODULE GALAHAD_MS_precision
