! *****************************************************************************
! ************************* BROKEN ** DO NOT USE ******************************

! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  May 15th 2002

   MODULE GALAHAD_QPS_precision

!     -------------------------------------------------
!     | Scale the data for the quadratic program      |
!     |                                               |
!     |    minimize     1/2 x(T) H x + g(T) x + f     |
!     |                                               |
!     |    subject to     x_l <=   x   <= x_u         |
!     |    and            c_l <=  A x  <= c_u         |
!     |                                               |
!     | to make later solution easier                 |
!     |                                               |
!     | Nick Gould                                    |
!     | Started May 2002, Finished ???                |
!     -------------------------------------------------

      USE GALAHAD_KINDS_precision
      USE GALAHAD_QPT_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPS_initialize, QPS_get_scalings, QPS_apply, QPS_restore

      TYPE, PUBLIC :: QPS_control_type
        INTEGER ( KIND = ip_ ) :: error
        REAL ( KIND = rp_ ) :: infinity
        LOGICAL :: treat_zero_bounds_as_general
      END TYPE

      TYPE, PUBLIC :: QPS_inform_type
        INTEGER ( KIND = ip_ ) :: status, alloc_status
      END TYPE

      TYPE, PUBLIC :: QPS_scale_type
         REAL ( KIND = rp_ ) :: col_scale_rhs
         REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: row_scale
         REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: col_scale_x
         REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: col_scale_c
         REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: shift_x, shift_c
      END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER ( KIND = ip_ ), PARAMETER :: max_cycle = 10
      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )

   CONTAINS

!-*-*-*-*-*-   Q P P _ i n i t i a l i z e   S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE QPS_initialize( map, control )

!  End of subroutine QPS_initialize

      END SUBROUTINE QPS_initialize

      SUBROUTINE QPS_apply( prob, scale, control, inform )

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QPS_scale_type ), INTENT( INOUT ) :: scale
      TYPE ( QPS_control_type ), INTENT( IN ) :: control
      TYPE ( QPS_inform_type ), INTENT( OUT ) :: inform

      INTEGER ( KIND = ip_ ) :: cycle, i, k, n_slacks
      REAL ( KIND = rp_ ) :: row_norm, cl, cu, xl, xu
      INTEGER ( KIND = ip_ ), DIMENSION( prob%m ) :: SLACKS
      REAL ( KIND = rp_ ), DIMENSION( prob%m ) :: RHS

      n_slacks = 0

!  Shift the variables

      DO i = 1, prob%n
        xl = prob%X_l( i ) ; xu = prob%X_u( i )
        IF ( xl <= zero .AND. xu >= zero ) THEN
          scale%shift_x( i ) = zero
        ELSE IF ( xl > zero ) THEN
          scale%shift_x( i ) = xl
          prob%X_l( i ) = zero ; prob%X_u( i ) = xu - xl
        ELSE
          scale%shift_x( i ) = xu
          prob%X_l( i ) = xl - xu ; prob%X_u( i ) = zero
        END IF
      END DO

!  Shift the constraints

      DO i = 1, prob%m
        cl = prob%C_l( i ) ; cu = prob%C_u( i )
        IF ( cl <= zero .AND. cu >= zero ) THEN
          scale%shift_c( i ) = zero
        ELSE IF ( cl > zero ) THEN
          scale%shift_c( i ) = cl
          prob%C_l( i ) = zero ; prob%C_u( i ) = cu - cl
        ELSE
          scale%shift_c( i ) = cu
          prob%C_l( i ) = cl - cu ; prob%C_u( i ) = zero
        END IF

!  Record slack variables

        IF ( cl /= cu ) THEN
          n_slacks = n_slacks + 1
          SLACKS( n_slacks ) = i
        END IF
        RHS( i ) = scale%shift_c( i )
        DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          RHS( i ) =                                                           &
            RHS( i ) - prob%A%val( k ) * scale%shift_x( prob%A%col( k ) )
        END DO
      END DO

!  Now scale ( A : - I : rhs ) in a cycle of alternate row and column
!  equilibrations. First, set the scale factors to 1.0

      scale%row_scale = one
      scale%col_scale_x = one ; scale%col_scale_c = one
      scale%col_scale_rhs = one

      DO cycle = 1, max_cycle
        DO i = 1, prob%m

!  Compute row norms

          row_norm = ( scale%col_scale_rhs * RHS( i ) ) ** 2
          DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            row_norm = row_norm +                                              &
              ( scale%col_scale_x( prob%A%col( k ) ) * prob%A%val( k ) ) ** 2
          END DO
          row_norm = ABS( scale%row_scale( i ) ) * SQRT( row_norm )

!  Divide the scale factors by the row norms

          IF ( row_norm /= zero )                                              &
            scale%row_scale( i ) = scale%row_scale( i ) / row_norm
        END DO

        DO i = 1, prob%m

!  Compute column norms

!!!! finish this !!!

          row_norm = ( scale%col_scale_rhs * RHS( i ) ) ** 2
          DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            row_norm = row_norm +                                              &
              ( scale%col_scale_x( prob%A%col( k ) ) * prob%A%val( k ) ) ** 2
          END DO
          row_norm( i ) = ABS( scale%row_scale( i ) ) * SQRT( row_norm( i ) )

!  Divide the scale factors by the column norms

          IF ( row_norm( i ) /= zero )                                         &
            scale%row_scale( i ) = scale%row_scale( i ) / row_norm
        END DO
      END DO

!  Compute row norms

      END DO

!  End of subroutine QPS_apply

      END SUBROUTINE QPS_apply


      SUBROUTINE QPS_restore( prob, scale, control, inform )

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QPS_scale_type ), INTENT( IN ) :: scale
      TYPE ( QPS_control_type ), INTENT( IN ) :: control
      TYPE ( QPS_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i

!  Un-shift the variables

      DO i = 1, prob%n
        prob%X_l( i ) = prob%X_l( i ) + scale%shift_x( i )
        prob%X_u( i ) = prob%X_u( i ) + scale%shift_x( i )
      END DO

!  Un-shift the constraints

      DO i = 1, prob%m
        prob%C_l( i ) = prob%C_l( i ) + scale%shift_c( i )
        prob%C_u( i ) = prob%C_u( i ) + scale%shift_c( i )
      END DO

!  End of subroutine QPS_restore

      END SUBROUTINE QPS_restore

!  End of module QPS

   END MODULE GALAHAD_QPS_precision

! ************************* BROKEN ** DO NOT USE ******************************
! *****************************************************************************
