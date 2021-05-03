! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  May 15th 2002

   MODULE GALAHAD_QPS_double

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

      USE GALAHAD_QPT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPS_initialize, QPS_get_scalings, QPS_apply, QPS_restore

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

      TYPE, PUBLIC :: QPS_control_type
        INTEGER :: error
        REAL ( KIND = wp ) :: infinity
        LOGICAL :: treat_zero_bounds_as_general
      END TYPE

      TYPE, PUBLIC :: QPS_inform_type
        INTEGER :: status, alloc_status
      END TYPE

      TYPE, PUBLIC :: QPS_scale_type
         REAL ( KIND = wp ) :: col_scale_rhs
         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: row_scale, col_scale_x 
         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: col_scale_c
         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: shift_x, shift_c
      END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

   CONTAINS

!-*-*-*-*-*-   Q P P _ i n i t i a l i z e   S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE QPS_initialize( map, control )

!  End of subroutine QPS_initialize

      END SUBROUTINE QPS_initialize

      SUBROUTINE QPS_apply( prob, scale, control, inform )

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QPS_scale_type ), INTENT( IN ) :: scale
      TYPE ( QPS_control_type ), INTENT( IN ) :: control
      TYPE ( QPS_inform_type ), INTENT( OUT ) :: inform

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
        DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
          RHS( i ) = RHS( i ) - A_val( k ) * scale%shift_x( A_col( k ) )
        END DO
      END DO

!  Now scale ( A : - I : rhs ) in a cycle of alternate row and column
!  equilibrations. First, set the scale factors to 1.0

      row_scale = one
      col_scale_x = one ; col_scale_c = one ; col_scale_rhs = one
      
      DO cycle = 1, max_cycle



        DO i = 1, prob%m

!  Compute row norms

          row_norm = ( col_scale_rhs * RHS( i ) ) ** 2
          DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
            row_norm = row_norm +                                              &
              ( col_scale_x( A_col( k ) ) * A_val( k ) ) ** 2
          END DO
          row_norm = ABS( row_scale( i ) ) * SQRT( row_norm )

!  Divide the scale factors by the row norms

          IF ( row_norm /= zero ) row_scale( i ) = row_scale( i ) / row_norm
        END DO

        DO i = 1, prob%m

!  Compute column norms

!!!! finish this !!!

          row_norm = ( col_scale_rhs * RHS( i ) ) ** 2
          DO k = A_ptr( i ), A_ptr( i + 1 ) - 1
            row_norm = row_norm +                                              &
              ( col_scale_x( A_col( k ) ) * A_val( k ) ) ** 2
          END DO
          row_norm( i ) = ABS( row_scale( i ) ) * SQRT( row_norm( i ) )

!  Divide the scale factors by the column norms



          IF ( row_norm( i ) /= zero )                                         &
            row_scale( i ) = row_scale( i ) / row_norm
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

   END MODULE GALAHAD_QPS_double
