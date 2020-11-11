! THIS VERSION: GALAHAD 2.6 - 18/01/2014 AT 12:30 GMT.
!
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*    GALAHAD QPT  M O D U L E     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Philippe Toint

!  History -
!   originally released pre GALAHAD Version 1.0. July 16th 2000
!   update released with GALAHAD Version 2.0. April 7th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_QPT_double

!     -----------------------------------------------------------------------
!     |                                                                     |
!     | Provide a derived data type for the (possible parametric)           |
!     | quadratic programming problem                                       |
!     |                                                                     |
!     |    minimize 1/2 x(T) H x + g(T) x + f + theta dg(T) x               |
!     |                                                                     |
!     |                           or                                        |
!     |                                                                     |
!     |       1/2 || W * ( x - x_0 ) ||_2^2 + g^T x + f + theta dg(T) x     |
!     |                                                                     |
!     |    subject to c_l + theta dc_l <= A x <= c_u + theta dc_u           |
!     |               x_l + theta dx_l <=  x  <= x_u + theta dx_u           |
!     |               y_l + theta dy_l <=  y  <= y_u + theta dy_u           |
!     |    and        z_l + theta dz_l <=  z  <= z_u + theta dz_u           |
!     |                                                                     |
!     | for all 0 <= theta <= theta_max,                                    |
!     |                                                                     |
!     | where y are the multipliers associated with the linear constraints  |
!     | and z the dual variables associated with the bounds.                |
!     |                                                                     |
!     | Additionally, the general linear constraints may be replaced by     |
!     | the quadratic ones                                                  |
!     |                                                                     |
!     |    c_l + theta dc_l <= A x + 1/2 x . H_c x <= c_u + theta dc_u      |
!     |                                                                     |
!     | to form a quadratically-constrained quadratic program               |
!     |                                                                     |
!     -----------------------------------------------------------------------

!-------------------------
!  M o d u l e s   u s e d
!-------------------------

!     Storing matrices

      USE GALAHAD_SMT_double

!     Sorting

      USE GALAHAD_SORT_double

!     Storing limited-memory matrices

      USE GALAHAD_LMT_double, LMS_control_type => LMT_control_type,            &
                              LMS_data_type => LMT_data_type

!     Special values

      USE GALAHAD_SYMBOLS,                                                    &
          ALL_ZEROS           => GALAHAD_ALL_ZEROS,                           &
          ALL_ONES            => GALAHAD_ALL_ONES

!     Variable and constraints status

      USE GALAHAD_SYMBOLS,                                                    &
          INACTIVE            => GALAHAD_INACTIVE,                            &
          ELIMINATED          => GALAHAD_ELIMINATED

!     string funtions

      USE GALAHAD_STRING, ONLY: STRING_upper

!     Exit codes

      USE GALAHAD_SYMBOLS,                                                    &
          OK                  => GALAHAD_SUCCESS,                             &
          MEMORY_FULL         => GALAHAD_MEMORY_FULL,                         &
          NOT_DIAGONAL        => GALAHAD_NOT_DIAGONAL

      IMPLICIT NONE

      PRIVATE

!     Ensure the private nature of the imported symbols.

      PRIVATE :: INACTIVE, ELIMINATED, OK, MEMORY_FULL

!     Make the tools public.

      PUBLIC :: SMT_type, SMT_put, SMT_get, QPT_put_H, QPT_put_A,              &
                QPT_keyword_H, QPT_keyword_A,                                  &
                QPT_summarize_problem,                                         &
                QPT_write_problem, QPT_write_to_sif,                           &
                QPT_A_from_D_to_S, QPT_A_from_S_to_D,                          &
                QPT_A_from_C_to_S, QPT_A_from_S_to_C,                          &
                QPT_H_from_D_to_S, QPT_H_from_S_to_D,                          &
                QPT_H_from_Di_to_S, QPT_H_from_S_to_Di,                        &
                QPT_H_from_C_to_S, QPT_H_from_S_to_C,                          &
                LMS_control_type, LMS_data_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0E+0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PRIVATE, PARAMETER :: wp = dp

!-----------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n
!-----------------------------------------------

      TYPE, PUBLIC :: QPT_problem_type
        INTEGER :: m                        ! number of constraints
        INTEGER :: n                        ! number of variables
        INTEGER :: x_free                   ! number of free variables
        INTEGER :: x_l_start                ! first variable with a lower bound
        INTEGER :: x_l_end                  ! last variable with a lower bound
        INTEGER :: x_u_start                ! first variable with an upper bound
        INTEGER :: x_u_end                  ! last variable with an upper bound
        INTEGER :: h_diag_end_free
        INTEGER :: h_diag_end_nonneg
        INTEGER :: h_diag_end_nonpos
        INTEGER :: h_diag_end_lower
        INTEGER :: h_diag_end_range
        INTEGER :: h_diag_end_upper
        INTEGER :: h_diag_end_fixed
        INTEGER :: c_equality
        INTEGER :: c_l_end
        INTEGER :: c_u_start
        INTEGER :: c_u_end
        INTEGER :: Hessian_kind = - 1        ! whether H or WEIGHT need be given
        INTEGER :: target_kind = - 1         ! kind of target X0
        INTEGER :: gradient_kind = - 1       ! kind of gradient
        REAL ( KIND = wp ) :: f = 0.0_wp     ! constant term
        REAL ( KIND = wp ) :: infinity = ( 10.0_wp ) ** 20 ! bound infinity
        REAL ( KIND = wp ) :: df = 0.0_wp    ! parametric constant term
        REAL ( KIND = wp ) :: q = 0.0_wp     ! value of the objective
        REAL ( KIND = wp ) :: theta_max = 0.0_wp ! upper bound on parametric
                                             ! range
        REAL ( KIND = wp ) :: theta = 0.0    ! current value of parameter
        REAL ( KIND = wp ) :: rho_g = 1.0_wp ! penalty parameter for general
                                             ! linear constraints for l_1QPs
        REAL ( KIND = wp ) :: rho_b = 1.0_wp ! penalty parameter for simple
                                             ! bound constraints for l_1QPs
        LOGICAL :: new_problem_structure = .TRUE. ! has the structure changed?

!  allocatable arrays

        CHARACTER, ALLOCATABLE, DIMENSION( : ) :: name
        CHARACTER ( len = 10 ), ALLOCATABLE, DIMENSION( : ) :: X_names
        CHARACTER ( len = 10 ), ALLOCATABLE, DIMENSION( : ) :: C_names
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_type
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DC_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DC_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_l
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DG
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WEIGHT

        TYPE ( SMT_type ) :: A, H, H_c
!       TYPE ( SMT_type ), ALLOCATABLE, DIMENSION( : ) :: H_c
        TYPE ( LMS_data_type ) :: H_lm

      END TYPE

!  ----------------
!  Other parameters
!  ----------------

      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ZERO = 0.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER ::  ONE = 1.0_wp

   CONTAINS

!-*-*-*-*-*-*-*-*-   Q P T _ p u t _ H   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

     SUBROUTINE QPT_put_H( array, string, inform, stat )

!  Dummy arguments

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array
     CHARACTER ( len = * ), INTENT( IN ) :: string
     INTEGER, INTENT( OUT ), OPTIONAL ::  inform, stat

!  Local variables

     INTEGER :: i, status

!  Insert the string

     IF ( ALLOCATED( array ) ) DEALLOCATE( array )
     CALL SMT_put( array, string = string, stat = status )

!  Check for success

     IF ( PRESENT( stat ) ) stat = status
     IF ( PRESENT( inform ) ) THEN
       IF ( status == 0 ) THEN
         inform = 0
       ELSE
         inform = - 1
       END IF
     END IF
     IF ( status /= 0 ) RETURN

!  Convert the string to upper case

     DO i = 1, SIZE( array )
       CALL STRING_upper( array( i : i ) )
     END DO

!  Check to see if the string is an appropriate keyword

     IF ( PRESENT( inform ) ) THEN
       IF ( QPT_keyword_H( array ) ) inform = - 2
     END IF

     RETURN

!  End of QPT_put_H

     END SUBROUTINE QPT_put_H

!-*-*-*-*-*-*-*-*-   Q P T _ k e y w o r d _ H   F U N C T I O N  -*-*-*--*-*-*-

     FUNCTION QPT_keyword_H( array )
     LOGICAL :: QPT_keyword_H

!  Dummy arguments

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array

!  Check to see if the string is an appropriate keyword

     SELECT CASE( SMT_get( array ) )

!  Keyword known

     CASE( 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', 'LBFGS',                   &
           'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY', 'NONE', 'ZERO' )
       QPT_keyword_H = .TRUE.

!  Keyword unknown

     CASE DEFAULT
       QPT_keyword_H = .FALSE.
     END SELECT

     RETURN

!  End of QPT_keyword_H

     END FUNCTION QPT_keyword_H

!-*-*-*-*-*-*-*-*-   Q P T _ p u t _A   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

     SUBROUTINE QPT_put_A( array, string, inform, stat )

!  Dummy arguments

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array
     CHARACTER ( len = * ), INTENT( IN ) :: string
     INTEGER, INTENT( OUT ), OPTIONAL ::  inform, stat

!  Local variables

     INTEGER :: i, status

!  Insert the string

     IF ( ALLOCATED( array ) ) DEALLOCATE( array )
     CALL SMT_put( array, string = string, stat = status )

!  Check for success

     IF ( PRESENT( stat ) ) stat = status
     IF ( PRESENT( inform ) ) THEN
       IF ( status == 0 ) THEN
         inform = 0
       ELSE
         inform = - 1
       END IF
     END IF
     IF ( status /= 0 ) RETURN

!  Convert the string to upper case

     DO i = 1, SIZE( array )
       CALL STRING_upper( array( i : i ) )
     END DO

!  Check to see if the string is an appropriate keyword

     IF ( PRESENT( inform ) ) THEN
       IF ( QPT_keyword_A( array ) ) inform = - 2
     END IF

     RETURN

!  End of QPT_put_A

     END SUBROUTINE QPT_put_A

!-*-*-*-*-*-*-*-*-   Q P T _ k e y w o r d _ A   F U N C T I O N  -*-*-*-*-*-*-

     FUNCTION QPT_keyword_A( array )
     LOGICAL :: QPT_keyword_A

!  Dummy arguments

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array

!  Check to see if the string is an appropriate keyword

     SELECT CASE( SMT_get( array ) )

!  Keyword known

     CASE( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',                       &
           'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' )
       QPT_keyword_A = .TRUE.

!  Keyword unknown

     CASE DEFAULT
       QPT_keyword_A = .FALSE.
     END SELECT

     RETURN

!  End of QPT_keyword_A

     END FUNCTION QPT_keyword_A

!===============================================================================

      SUBROUTINE QPT_summarize_problem( out, prob, lp )

!  Summarizes the problem prob on output device out

!  Nick Gould, December 23rd 2014

!  Dummy arguments

      INTEGER, INTENT( IN ) :: out
      TYPE ( QPT_problem_type ), INTENT( IN ) :: prob
      LOGICAL, OPTIONAL, INTENT( IN ) :: lp

!  local variables

      INTEGER :: i, j
      LOGICAL :: is_lp

      IF ( PRESENT( lp ) ) THEN
        is_lp = lp
      ELSE
        is_lp = .FALSE.
      END IF

      WRITE( out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m

!  objective function

      WRITE( out, "( ' f = ', ES12.4 )" ) prob%f
      IF ( prob%gradient_kind == 0 ) THEN
        WRITE( out, "( ' G = zeros' )" )
      ELSE IF ( prob%gradient_kind == 1 ) THEN
        WRITE( out, "( ' G = ones' )" )
      ELSE
        WRITE( out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
      END IF
      IF ( .NOT. is_lp ) THEN
        IF ( prob%Hessian_kind == 0 ) THEN
          WRITE( out, "( ' W = zeros' )" )
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          WRITE( out, "( ' W = ones ' )" )
          IF ( prob%target_kind == 0 ) THEN
            WRITE( out, "( ' X0 = zeros' )" )
          ELSE IF ( prob%target_kind == 1 ) THEN
            WRITE( out, "( ' X0 = ones ' )" )
          ELSE
            WRITE( out, "( ' X0 = ', /, ( 5ES12.4 ) )" ) prob%X0( : prob%n )
          END IF
        ELSE IF ( prob%Hessian_kind == 2 ) THEN
          WRITE( out, "( ' W = ', /, ( 5ES12.4 ) )" ) prob%WEIGHT( : prob%n )
          IF ( prob%target_kind == 0 ) THEN
            WRITE( out, "( ' X0 = zeros' )" )
          ELSE IF ( prob%target_kind == 1 ) THEN
            WRITE( out, "( ' X0 = ones ' )" )
          ELSE
            WRITE( out, "( ' X0 = ', /, ( 5ES12.4 ) )" ) prob%X0( : prob%n )
          END IF
        ELSE
          IF ( SMT_get( prob%H%type ) == 'NONE' .OR.                           &
               SMT_get( prob%H%type ) == 'ZERO' ) THEN
            WRITE( out, "( ' No H' )" )
          ELSE IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
            WRITE( out, "( ' H (identity)' )" )
          ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
            WRITE( out, "( ' H (identity scaled by ', /, ES12.4,')')" )        &
              prob%H%val( 1 )
          ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            WRITE( out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )             &
              prob%H%val( : prob%n )
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            WRITE( out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )                &
              prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            WRITE( out, "( ' H (row-wise) = ' )" )
            DO i = 1, prob%n
              WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                         &
                ( i, prob%H%col( j ), prob%H%val( j ),                         &
                  j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
            END DO
          ELSE IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
            WRITE( out, "( ' H (co-ordinate) = ' )" )
            WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                           &
            ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ),               &
              i = 1, prob%H%ne)
          ELSE IF ( SMT_get( prob%H%type ) == 'LBFGS' ) THEN
            WRITE( out, "( ' L-BFGS H (not explicit)' )" )
          END IF
        END IF
      END IF

!  simple bounds

      WRITE( out, "( ' X_l = ', /, ( 5ES12.4 ) )" ) prob%X_l( : prob%n )
      WRITE( out, "( ' X_u = ', /, ( 5ES12.4 ) )" ) prob%X_u( : prob%n )

!  general constraints

      IF ( prob%m > 0 ) THEN
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )                  &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                           &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( out, "( ' A (co-ordinate) = ' )" )
          WRITE( out, "( ( 2( 2I8, ES12.4 ) ) )" )                             &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
        END IF
        WRITE( out, "( ' C_l = ', /, ( 5ES12.4 ) )" ) prob%C_l( : prob%m )
        WRITE( out, "( ' C_u = ', /, ( 5ES12.4 ) )" ) prob%C_u( : prob%m )
      END IF

      RETURN

!  end of subroutine QPT_summarize_problem

      END SUBROUTINE QPT_summarize_problem

!===============================================================================

      SUBROUTINE QPT_write_problem( out, prob, level )

!     Writes the problem prob on output device out.

!     Arguments

      INTEGER, INTENT( IN ) :: out

!            the output device

      TYPE ( QPT_problem_type ), INTENT( IN ) :: prob

!            the problem to print out

      INTEGER, INTENT( IN ), OPTIONAL :: level

!            the level of detail required on output:
!            0 : writes the values of n, m, X, C,  Z, Y, X_l, X_u, C_l, C_u,
!                Y_l, Y_u, Z_l, Z_u, A, f, q, G, H, X0, WEIGHT.
!            1 : additionally writes new_problem_structure, gradient_kind,
!                Hessian_kind, rho_g and rho_b and indicates which of the
!                variables/constraints components are inactive.
!            2 : writes every component of the prob structure.
!            Each vector is only written if it is associated.

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: i, j, k, lev
      LOGICAL :: lowv, valv, uppv, stav, gotH

!     Determine the print level

      IF ( PRESENT( level ) ) THEN
         lev = MAX( 0, MIN( 2, level ) )
      ELSE
         lev = 0
      END IF

!     Write banner.

      WRITE( out, * ) ' '
      WRITE( out, * ) '   =============== PROBLEM ====================='
      WRITE( out, * ) ' '

      IF ( lev > 0 ) THEN
         WRITE( out, * ) ' '
         WRITE( out, * ) '   new_problem_structure = ',                        &
              prob%new_problem_structure
         WRITE( out, * ) ' '
      END IF

!     --------------------------------------------------------------------------
!                             Write the variables.
!     --------------------------------------------------------------------------

      WRITE( out, * ) '   n = ', prob%n

      IF ( prob%n > 0 ) THEN

         IF ( lev == 2 ) THEN
            WRITE( out, * ) ' '
            WRITE( out, * ) '   number of free variables                  = ',&
                            prob%x_free
            WRITE( out, * ) '   index of the first lower bounded variable = ', &
                            prob%x_l_start
            WRITE( out, * ) '   index of the last  lower bounded variable = ', &
                            prob%x_l_end
            WRITE( out, * ) '   index of the first upper bounded variable = ', &
                            prob%x_u_start
            WRITE( out, * ) '   index of the last  upper bounded variable = ', &
                            prob%x_u_end
         END IF

         lowv = ALLOCATED( prob%X_l )
         valv = ALLOCATED( prob%X   )
         uppv = ALLOCATED( prob%X_u )
         stav = ALLOCATED( prob%X_status ) .AND. lev > 0

         IF ( lowv .AND. valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual     upper '
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 103 ) j, prob%X_l(j), prob%X(j), prob%X_u(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X_l(j), prob%X(j), prob%X_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X_l(j), prob%X(j), prob%X_u(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 102 ) j, prob%X_l(j), prob%X(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X_l(j), prob%X(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X_l(j), prob%X(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 102 ) j, prob%X_l(j), prob%X_u(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X_l(j), prob%X_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X_l(j), prob%X_u(j)
               END DO
            END IF

         ELSE IF ( valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual      upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 102 ) j, prob%X(j), prob%X_u(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X(j), prob%X_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X(j), prob%X_u(j)
               END DO
            END IF

         ELSE IF ( lowv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 101 ) j, prob%X_l(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X_l(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X_l(j)
               END DO
            END IF

         ELSE IF ( valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 101 ) j, prob%X(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X(j)
               END DO
            END IF

         ELSE IF ( uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   variables '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 101 ) j, prob%X_u(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%X_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%X_u(j)
               END DO
            END IF
         END IF

!     --------------------------------------------------------------------------
!                         Write the dual variables.
!     --------------------------------------------------------------------------

         lowv = ALLOCATED( prob%Z_l )
         valv = ALLOCATED( prob%Z   )
         uppv = ALLOCATED( prob%Z_u )

         IF ( lowv .AND. valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual     upper '
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 203 ) j, prob%Z_l(j), prob%Z(j), prob%Z_u(j)
                  CASE DEFAULT
                     WRITE( out, 200 ) j, prob%Z_l(j), prob%Z(j), prob%Z_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 200 ) j, prob%Z_l(j), prob%Z(j), prob%Z_u(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 102 ) j, prob%Z_l(j), prob%Z(j)
                  CASE DEFAULT
                     WRITE( out, 100 ) j, prob%Z_l(j), prob%Z(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 100 ) j, prob%Z_l(j), prob%Z(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 202 ) j, prob%Z_l(j), prob%Z_u(j)
                  CASE DEFAULT
                     WRITE( out, 200 ) j, prob%Z_l(j), prob%Z_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 200 ) j, prob%Z_l(j), prob%Z_u(j)
               END DO
            END IF

         ELSE IF ( valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual      upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 202 ) j, prob%Z(j), prob%Z_u(j)
                  CASE DEFAULT
                     WRITE( out, 200 ) j, prob%Z(j), prob%Z_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 200 ) j, prob%Z(j), prob%Z_u(j)
               END DO
            END IF

         ELSE IF ( lowv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 201 ) j, prob%Z_l(j)
                  CASE DEFAULT
                     WRITE( out, 200 ) j, prob%Z_l(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 200 ) j, prob%Z_l(j)
               END DO
            END IF

         ELSE IF ( valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 201 ) j, prob%Z(j)
                  CASE DEFAULT
                     WRITE( out, 200 ) j, prob%Z(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 200 ) j, prob%Z(j)
               END DO
            END IF

         ELSE IF ( uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   z multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 201 ) j, prob%Z_u(j)
                  CASE DEFAULT
                     WRITE( out, 200 ) j, prob%Z_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%n
                  WRITE( out, 200 ) j, prob%Z_u(j)
               END DO
            END IF

         END IF

      END IF

!     --------------------------------------------------------------------------
!                            Write the constraints.
!     --------------------------------------------------------------------------

      WRITE( out, * ) ' '
      WRITE( out, * ) '   m = ', prob%m

      IF ( lev == 2 ) THEN
         WRITE( out, * ) ' '
         WRITE( out, * ) '   number of equality constraints              = ',  &
                         prob%c_equality
         WRITE( out, * ) '   index of the last  lower bounded constraint = ',  &
                         prob%c_l_end
         WRITE( out, * ) '   index of the first upper bounded constraint = ',  &
                         prob%c_u_start
         WRITE( out, * ) '   index of the last  upper bounded constraint = ',  &
                         prob%c_u_end
      END IF

      IF ( prob%m > 0 ) THEN

         lowv = ALLOCATED( prob%C_l )
         valv = ALLOCATED( prob%C   )
         uppv = ALLOCATED( prob%C_u )
         stav = ALLOCATED( prob%C_status ) .AND. lev > 0

         IF ( lowv .AND. valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual     upper '
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 303 ) j, prob%C_l(j), prob%C(j), prob%C_u(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C_l(j), prob%C(j), prob%C_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C_l(j), prob%C(j), prob%C_u(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 302 ) j, prob%C_l(j), prob%C(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C_l(j), prob%C(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C_l(j), prob%C(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 302 ) j, prob%C_l(j), prob%C_u(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C_l(j), prob%C_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C_l(j), prob%C_u(j)
               END DO
            END IF

         ELSE IF ( valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual      upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 302 ) j, prob%C(j), prob%C_u(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C(j), prob%C_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C(j), prob%C_u(j)
               END DO
            END IF

         ELSE IF ( lowv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 301 ) j, prob%C_l(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C_l(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C_l(j)
               END DO
            END IF

         ELSE IF ( valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 301 ) j, prob%C(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C(j)
               END DO
            END IF

         ELSE IF ( uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   constraints '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 301 ) j, prob%C_u(j)
                  CASE DEFAULT
                     WRITE( out, 300 ) j, prob%C_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 300 ) j, prob%C_u(j)
               END DO
            END IF
         END IF

!     --------------------------------------------------------------------------
!                            Write the multipliers.
!     --------------------------------------------------------------------------

         lowv = ALLOCATED( prob%Y_l )
         valv = ALLOCATED( prob%Y   )
         uppv = ALLOCATED( prob%Y_u )

         IF ( lowv .AND. valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual     upper '
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 403 ) j, prob%Y_l(j), prob%Y(j), prob%Y_u(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y_l(j), prob%Y(j), prob%Y_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y_l(j), prob%Y(j), prob%Y_u(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 402 ) j, prob%Y_l(j), prob%Y(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y_l(j), prob%Y(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y_l(j), prob%Y(j)
               END DO
            END IF

         ELSE IF ( lowv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower       upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 402 ) j, prob%Y_l(j), prob%Y_u(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y_l(j), prob%Y_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y_l(j), prob%Y_u(j)
               END DO
            END IF

         ELSE IF ( valv .AND. uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual      upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 402 ) j, prob%Y(j), prob%Y_u(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y(j), prob%Y_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y(j), prob%Y_u(j)
               END DO
            END IF

         ELSE IF ( lowv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   lower'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 401 ) j, prob%Y_l(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y_l(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y_l(j)
               END DO
            END IF

         ELSE IF ( valv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   actual'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 401 ) j, prob%Y(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y(j)
               END DO
            END IF

         ELSE IF ( uppv ) THEN

            WRITE( out, * ) ' '
            WRITE( out, * ) '   y multipliers '
            WRITE( out, * ) ' '
            WRITE( out, * ) '                   upper'
            WRITE( out, * ) ' '
            IF ( stav ) THEN
               DO j = 1, prob%m
                  SELECT CASE ( prob%C_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 401 ) j, prob%Y_u(j)
                  CASE DEFAULT
                     WRITE( out, 400 ) j, prob%Y_u(j)
                  END SELECT
               END DO
             ELSE
               DO j = 1, prob%m
                  WRITE( out, 400 ) j, prob%Y_u(j)
               END DO
            END IF

         END IF

!     --------------------------------------------------------------------------
!                             Write the Jacobian.
!     --------------------------------------------------------------------------

         WRITE( out, * ) ' '
         WRITE( out, * ) '   Jacobian '
         WRITE( out, * ) ' '

         stav = ALLOCATED( prob%X_status ) .AND. ALLOCATED( prob%C_status )  &
                .AND. lev > 0

         IF ( lev == 2 ) THEN
            WRITE( out, * ) ' '
            WRITE( out, * ) '   A_type = ', SMT_get( prob%A%type )
            WRITE( out, * ) ' '
         END IF

         SELECT CASE ( TRIM( SMT_get( prob%A%type ) ) )
         CASE ( 'DENSE' )  ! Dense Jacobian

            IF ( stav ) THEN
               k = 0
               DO i = 1, prob%m
                  DO j = 1, prob%n
                     k = k + 1
                     IF ( prob%X_status( j ) == INACTIVE .OR. &
                          prob%C_status( i ) == INACTIVE      ) THEN
                        WRITE( out , 601 ) i, j, prob%A%val( k )
                     ELSE
                        WRITE( out , 600 ) i, j, prob%A%val( k )
                     END IF
                  END DO
               END DO
            ELSE
               k = 0
               DO i = 1, prob%m
                  DO j = 1, prob%n
                     k = k + 1
                     WRITE( out , 600 ) i, j, prob%A%val( k )
                  END DO
               END DO
            END IF

         CASE ( 'SPARSE_BY_ROWS' )  ! Sparse Jacobian

            IF ( stav ) THEN
               DO i = 1, prob%m
                  DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                     j = prob%A%col( k )
                     IF ( prob%X_status( j ) == INACTIVE .OR. &
                          prob%C_status( i ) == INACTIVE      ) THEN
                        WRITE( out , 601 ) i, j, prob%A%val( k )
                     ELSE
                        WRITE( out , 600 ) i, j, prob%A%val( k )
                     END IF
                  END DO
               END DO
            ELSE
               DO i = 1, prob%m
                  DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                     WRITE( out , 600 ) i, prob%A%col( k ), prob%A%val( k )
                  END DO
               END DO
            END IF

         CASE ( 'COORDINATE' ) ! Coordinate Jacobian

            IF ( stav ) THEN
               DO k = 1, prob%A%ne
                  i = prob%A%row( k )
                  j = prob%A%col( k )
                  IF ( prob%X_status( j ) == INACTIVE .OR. &
                       prob%C_status( i ) == INACTIVE      ) THEN
                     WRITE( out , 601 ) i, j, prob%A%val( k )
                  ELSE
                     WRITE( out , 600 ) i, j, prob%A%val( k )
                  END IF
               END DO
            ELSE
               DO k = 1, prob%A%ne
                  WRITE( out , 600 ) prob%A%row(k), prob%A%col(k), prob%A%val(k)
               END DO
            END IF

         END SELECT

      END IF

!     --------------------------------------------------------------------------
!                     Write the objective function.
!     --------------------------------------------------------------------------

      IF ( ALLOCATED( prob%X ) ) THEN
         WRITE( out, * ) ' '
         WRITE( out, 10 ) prob%q
         WRITE( out, * ) ' '
      ELSE
         WRITE( out, * ) ' '
      END IF

      WRITE( out, * ) ' '
      WRITE( out, 11 ) prob%f
      WRITE( out, * ) ' '

!     Write the gradient

      IF ( prob%n > 0 ) THEN

         stav = ALLOCATED( prob%X_status ) .AND. lev > 0

         WRITE( out, * ) ' '
         WRITE( out, * ) '   gradient '
         WRITE( out, * ) ' '

         IF ( lev > 0 ) THEN
            WRITE( out, * ) '   gradient_kind = ', prob%gradient_kind
            WRITE( out, * ) ' '
         END IF

         SELECT CASE ( prob%gradient_kind )

         CASE ( ALL_ZEROS )

            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 501 ) j, ZERO
                  CASE DEFAULT
                     WRITE( out, 500 ) j, ZERO
                  END SELECT
               END DO
            ELSE
               DO j = 1, prob%n
                  WRITE( out, 500 ) j, ZERO
               END DO
            END IF

         CASE ( ALL_ONES )

            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 501 ) j, ONE
                  CASE DEFAULT
                     WRITE( out, 500 ) j, ONE
                  END SELECT
               END DO
            ELSE
               DO j = 1, prob%n
                  WRITE( out, 500 ) j, ONE
               END DO
            END IF

         CASE DEFAULT

            IF ( stav ) THEN
               DO j = 1, prob%n
                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( INACTIVE )
                     WRITE( out, 501 ) j, prob%G( j )
                  CASE DEFAULT
                     WRITE( out, 500 ) j, prob%G( j )
                  END SELECT
               END DO
            ELSE
               DO j = 1, prob%n
                  WRITE( out, 500 ) j, prob%G( j )
               END DO
            END IF

         END SELECT

!        Write the Hessian

         gotH = ALLOCATED( prob%H%val )
         IF ( gotH  ) THEN

            SELECT CASE ( TRIM( SMT_get( prob%H%type ) ) )

            CASE ( 'DENSE' )  ! Dense Hessian

               WRITE( out, * ) ' '
               WRITE( out, * ) '   Hessian '
               WRITE( out, * ) ' '

               IF ( lev == 2 ) THEN
                  WRITE( out, * )                                              &
                       '   index of last free diagonal element          = ',   &
                       prob%h_diag_end_free
                  WRITE( out, * )                                              &
                       '   index of last nonnegative diagonal element   = ',   &
                       prob%h_diag_end_nonneg
                  WRITE( out, * )                                              &
                       '   index of last nonnegative diagonal element   = ',   &
                       prob%h_diag_end_nonpos
                  WRITE( out, * )                                              &
                       '   index of last lower bounded diagonal element = ',   &
                       prob%h_diag_end_lower
                  WRITE( out, * )                                              &
                       '   index of last range bounded diagonal element = ',   &
                       prob%h_diag_end_range
                  WRITE( out, * )                                              &
                       '   index of last upper bounded diagonal element = ',   &
                       prob%h_diag_end_upper
                  WRITE( out, * )                                              &
                       '   index of last fixed diagonal element         = ',   &
                       prob%h_diag_end_fixed
                  WRITE( out, * )' '
               END IF

               IF ( stav ) THEN
                  k = 0
                  DO i = 1, prob%n
                     DO j = 1, i
                        k = k + 1
                        IF ( prob%X_status( j ) == INACTIVE .OR. &
                             prob%X_status( i ) == INACTIVE       ) THEN
                           WRITE( out, 701 )  i, j, prob%H%val( k )
                        ELSE
                           WRITE( out, 700 )  i, j, prob%H%val( k )
                        END IF
                     END DO
                  END DO
               ELSE
                  k = 0
                  DO i = 1, prob%n
                     DO j = 1, i
                        k = k + 1
                        WRITE( out, 700 )  i, j, prob%H%val( k )
                     END DO
                  END DO
               END IF

            CASE ( 'SPARSE_BY_ROWS' )  ! Sparse Hessian

               IF( prob%H%ptr( prob%n + 1 ) > 1 ) THEN

                  WRITE( out, * ) ' '
                  WRITE( out, * ) '   Hessian '
                  WRITE( out, * ) ' '

                  IF ( lev == 2 ) THEN
                     WRITE( out, * )                                           &
                          '   index of last free diagonal element          = ',&
                          prob%h_diag_end_free
                     WRITE( out, * )                                           &
                          '   index of last nonnegative diagonal element   = ',&
                          prob%h_diag_end_nonneg
                     WRITE( out, * )                                           &
                          '   index of last nonnegative diagonal element   = ',&
                          prob%h_diag_end_nonpos
                     WRITE( out, * )                                           &
                         '    index of last lower bounded diagonal element = ',&
                         prob%h_diag_end_lower
                     WRITE( out, * )                                           &
                         '    index of last range bounded diagonal element = ',&
                         prob%h_diag_end_range
                     WRITE( out, * )                                           &
                         '    index of last upper bounded diagonal element = ',&
                         prob%h_diag_end_upper
                     WRITE( out, * )                                           &
                         '    index of last fixed diagonal element         = ',&
                         prob%h_diag_end_fixed
                     WRITE( out, * )' '
                  END IF

                  IF ( stav ) THEN
                     DO i = 1, prob%n
                        DO k = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                           j = prob%H%col( k )
                           IF ( prob%X_status( j ) == INACTIVE .OR. &
                                prob%X_status( i ) == INACTIVE      ) THEN
                              WRITE( out, 701 )  i, j, prob%H%val( k )
                           ELSE
                              WRITE( out, 700 )  i, j, prob%H%val( k )
                           END IF
                        END DO
                     END DO
                  ELSE
                     DO i = 1, prob%n
                        DO k = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                           WRITE( out, 700 )  i, prob%H%col( k ), prob%H%val(k)
                        END DO
                     END DO
                  END IF

               END IF

            CASE ( 'COORDINATE' ) ! Coordinate Hessian

               IF ( prob%H%ne > 0 ) THEN

                  WRITE( out, * ) ' '
                  WRITE( out, * ) '   Hessian '
                  WRITE( out, * ) ' '

                  IF ( lev == 2 ) THEN
                     WRITE( out, * )' '
                     WRITE( out, * ) '   H_ne = ', prob%H%ne
                     WRITE( out, * )                                           &
                          '   index of last free diagonal element          = ',&
                          prob%h_diag_end_free
                     WRITE( out, * )                                           &
                          '   index of last nonnegative diagonal element   = ',&
                          prob%h_diag_end_nonneg
                     WRITE( out, * )                                           &
                          '   index of last nonnegative diagonal element   = ',&
                          prob%h_diag_end_nonpos
                     WRITE( out, * )                                           &
                          '   index of last lower bounded diagonal element = ',&
                          prob%h_diag_end_lower
                     WRITE( out, * )                                           &
                          '   index of last range bounded diagonal element = ',&
                          prob%h_diag_end_range
                     WRITE( out, * )                                           &
                          '   index of last upper bounded diagonal element = ',&
                          prob%h_diag_end_upper
                     WRITE( out, * )                                           &
                          '   index of last fixed diagonal element         = ',&
                          prob%h_diag_end_fixed
                     WRITE( out, * )' '
                  END IF

                  IF ( stav ) THEN
                     DO k = 1, prob%H%ne
                        i = prob%H%row( k )
                        j = prob%H%col( k )
                        IF ( prob%X_status( j ) == INACTIVE .OR. &
                             prob%X_status( i ) == INACTIVE      ) THEN
                           WRITE( out , 701 )  i, j, prob%H%val( k )
                        ELSE
                           WRITE( out , 700 )  i, j, prob%H%val( k )
                        END IF
                     END DO
                  ELSE
                     DO k = 1, prob%H%ne
                        WRITE( out, 700 ) prob%H%row( k ), prob%H%col( k ),    &
                                          prob%H%val(k)
                     END DO
                  END IF
               END IF

            CASE ( 'DIAGONAL' ) ! Diagonal Hessian

               IF ( prob%H%ne > 0 ) THEN

                  WRITE( out, * ) ' '
                  WRITE( out, * ) '   Hessian '
                  WRITE( out, * ) ' '

                  IF ( lev == 2 ) THEN
                     WRITE( out, * )                                           &
                          '   index of last free diagonal element          = ',&
                          prob%h_diag_end_free
                     WRITE( out, * )                                           &
                          '   index of last nonnegative diagonal element   = ',&
                          prob%h_diag_end_nonneg
                     WRITE( out, * )                                           &
                          '   index of last nonnegative diagonal element   = ',&
                          prob%h_diag_end_nonpos
                     WRITE( out, * )                                           &
                          '   index of last lower bounded diagonal element = ',&
                          prob%h_diag_end_lower
                     WRITE( out, * )                                           &
                          '   index of last range bounded diagonal element = ',&
                          prob%h_diag_end_range
                     WRITE( out, * )                                           &
                          '   index of last upper bounded diagonal element = ',&
                          prob%h_diag_end_upper
                     WRITE( out, * )                                           &
                          '   index of last fixed diagonal element         = ',&
                          prob%h_diag_end_fixed
                     WRITE( out, * )' '
                  END IF

                  IF ( stav ) THEN
                     DO i = 1, prob%n
                        IF ( prob%X_status( i ) == INACTIVE ) THEN
                           WRITE( out , 701 )  i, i, prob%H%val( i )
                        ELSE
                           WRITE( out , 700 )  i, i, prob%H%val( i )
                        END IF
                     END DO
                  ELSE
                     DO i = 1, prob%n
                        WRITE( out, 700 ) i, i, prob%H%val( i )
                     END DO
                  END IF
               END IF

            END SELECT

         ELSE

!           Write the weights.

            IF ( lev > 0 ) THEN
               WRITE( out, * ) '   Hessian_kind = ', prob%Hessian_kind
               WRITE( out, * ) ' '
            END IF

            stav = ALLOCATED( prob%X_status )

            WRITE( out, * ) ' '
            WRITE( out, * ) '   weights '
            WRITE( out, * ) ' '

            SELECT CASE ( prob%Hessian_kind )

            CASE ( ALL_ZEROS )

               IF ( stav ) THEN
                  DO j = 1, prob%n
                     SELECT CASE ( prob%X_status( j ) )
                     CASE ( INACTIVE )
                        WRITE( out, 801 ) j, ZERO
                     CASE DEFAULT
                        WRITE( out, 800 ) j, ZERO
                     END SELECT
                  END DO
               ELSE
                  DO j = 1, prob%n
                     WRITE( out, 800 ) j, ZERO
                  END DO
               END IF

            CASE ( ALL_ONES )

               IF ( stav ) THEN
                  DO j = 1, prob%n
                     SELECT CASE ( prob%X_status( j ) )
                     CASE ( INACTIVE )
                        WRITE( out, 801 ) j, ONE
                     CASE DEFAULT
                        WRITE( out, 800 ) j, ONE
                     END SELECT
                  END DO
               ELSE
                  DO j = 1, prob%n
                     WRITE( out, 800 ) j, ONE
                  END DO
               END IF

            CASE DEFAULT

               IF ( stav ) THEN
                  DO j = 1, prob%n
                     SELECT CASE ( prob%X_status( j ) )
                     CASE ( INACTIVE )
                        WRITE( out, 801 ) j, prob%WEIGHT( j )
                     CASE DEFAULT
                        WRITE( out, 800 ) j, prob%WEIGHT( j )
                     END SELECT
                  END DO
               ELSE
                  DO j = 1, prob%n
                     WRITE( out, 800 ) j, prob%WEIGHT( j )
                  END DO
               END IF

            END SELECT

         END IF

      END IF

!     --------------------------------------------------------------------------
!                         Write the penalty parameters.
!     --------------------------------------------------------------------------

      IF ( lev > 0 ) THEN
         WRITE( out, * ) ' '
         WRITE( out, 900 ) prob%rho_b
         WRITE( out, 901 ) prob%rho_g
         WRITE( out, * ) ' '
      END IF

!     Indicate end of problem.

      WRITE( out, * ) ' '
      WRITE( out, * ) '   ============ END OF PROBLEM ================='
      WRITE( out, * ) ' '

      RETURN

!     Formats

10    FORMAT( 3x, ' current objective function value = ', ES12.4 )
11    FORMAT( 3x, ' objective function constant term = ', ES12.4 )
100   FORMAT( 3x, 'x(', i4, ') =', 3x, 3ES12.4 )
101   FORMAT( 3x, 'x(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
102   FORMAT( 3x, 'x(', i4, ') =', 3x, 2ES12.4, 3x, 'inactive'   )
103   FORMAT( 3x, 'x(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
200   FORMAT( 3x, 'z(', i4, ') =', 3x, 3ES12.4 )
201   FORMAT( 3x, 'z(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
202   FORMAT( 3x, 'z(', i4, ') =', 3x, 2ES12.4, 3x, 'inactive'   )
203   FORMAT( 3x, 'z(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
300   FORMAT( 3x, 'c(', i4, ') =', 3x, 3ES12.4 )
301   FORMAT( 3x, 'c(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
302   FORMAT( 3x, 'c(', i4, ') =', 3x, 2ES12.4, 3x, 'inactive'   )
303   FORMAT( 3x, 'c(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
400   FORMAT( 3x, 'y(', i4, ') =', 3x, 3ES12.4 )
401   FORMAT( 3x, 'y(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
402   FORMAT( 3x, 'y(', i4, ') =', 3x, 2ES12.4, 3x, 'inactive'   )
403   FORMAT( 3x, 'y(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
500   FORMAT( 3x, 'g(', i4, ') =', 3x,  ES12.4 )
501   FORMAT( 3x, 'g(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
600   FORMAT( 3x, 'A(', i4, ',', i4, ') = ', ES12.4 )
601   FORMAT( 3x, 'A(', i4, ',', i4, ') = ', ES12.4, 3x, 'inactive'   )
700   FORMAT( 3x, 'H(', i4, ',', i4, ') = ', ES12.4 )
701   FORMAT( 3x, 'H(', i4, ',', i4, ') = ', ES12.4, 3x, 'inactive'   )
800   FORMAT( 3x, 'w(', i4, ') =', 3x,  ES12.4 )
801   FORMAT( 3x, 'w(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
900   FORMAT( 3x, 'penalty parameter for the bound   constraints = ', ES12.4 )
901   FORMAT( 3x, 'penalty parameter for the general constraints = ', ES12.4 )

      END SUBROUTINE QPT_write_problem

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_write_to_sif( prob, probname, filename, out,              &
                                   use_X_status, use_C_status, infinity,       &
                                   no_H )

!     Translates the problem prob into SIF and writes the result on device out.

!     Arguments

      TYPE ( QPT_problem_type ), INTENT( IN ) :: prob

!            the problem to print out

      CHARACTER( 10 ), INTENT( IN ) :: probname

!            the name to give to the SIF version of the problem.

      CHARACTER( 16 ), INTENT( IN ) :: filename

      INTEGER, INTENT( IN ) :: out

!            the output device

      LOGICAL, INTENT( IN ) :: use_X_status

!            .TRUE. iff only variables whose X_status > ELIMINATED must
!            be considered. If .FALSE., all variables are considered and
!            prob%X_status is never referenced.

      LOGICAL, INTENT( IN ) :: use_C_status

!            .TRUE. iff only constraints whose C_status > ELIMINATED must
!            be considered. If .FALSE., all constraints are considered and
!            prob%C_status is never referenced.

      REAL ( KIND = wp ), INTENT( IN ) :: infinity

!            the absolute value that is equivalent to infinity

      LOGICAL, OPTIONAL, INTENT( IN ) :: no_H

!            present if there is no Hessian term to write

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local parameters

!     These are possible values for the variables *_unique which indicate
!     whether or not some default value is recognized for bounds,
!     starting point components, constants or ranges.

      INTEGER, PARAMETER :: NO      =  0
      INTEGER, PARAMETER :: YES     =  1
      INTEGER, PARAMETER :: FREE    =  2  ! defaults for variables (bounds)
      INTEGER, PARAMETER :: LOWER   =  3
      INTEGER, PARAMETER :: UPPER   =  4
      INTEGER, PARAMETER :: BOTH    =  5
      INTEGER, PARAMETER :: FIXED   =  6

!     The maximum number of distinct values to consider for establishing
!     defaults values

      INTEGER, PARAMETER :: NVALUES = 20

!     These are symbols for the side of the two column data specification
!     of the SIF format on which the next value is to be written.

      INTEGER, PARAMETER :: LEFT    =  0
      INTEGER, PARAMETER :: RIGHT   =  1

!     Numerical parameter

      REAL( KIND = wp ), PARAMETER :: TEN = 10.0_wp

!     Local variables

      INTEGER            :: iostat, nactx, nactc, nactr, side, imx, i, j, k,   &
                            s_unique, b_unique, c_unique, r_unique, imy( 1 ),  &
                            nval1, ival1( NVALUES ), cval1( NVALUES ),         &
                            nval2, ival2( NVALUES ), cval2( NVALUES ),         &
                            nval3, ival3( NVALUES ), cval3( NVALUES )
      LOGICAL            :: all_x_active, notfound, c_details, r_details,      &
                            s_details, l_details, u_details, g_details, written
      CHARACTER( 1 )     :: cobj, ccon
      CHARACTER( 2 )     :: f1
      CHARACTER( 8 )     :: date
      CHARACTER( 10 )    :: f2, f3, f5
      CHARACTER( 12 )    :: f4, f6
      CHARACTER( 80 )    :: fmt
      REAL ( KIND = wp ) :: a, xlj, xuj, x0, c0, r0, bl0, bu0, cli, cui,       &
                            val1( NVALUES ), val2( NVALUES ), val3( NVALUES )

!-------------------------------------------------------------------------------

!     Analyze the problem in order to detect if some quantities may be
!     set by using the DEFAULT keyword. Also determine the presence of
!     nontrivial terms in the Hessian and in the linear constraints.
!     Finally, count the number of active variables and the number of
!     active constraints.

!-------------------------------------------------------------------------------

!     Initialize the objective and constraint types.

      cobj  = 'C'
      ccon  = 'U'

!     Initialize the number of active variables, and the counts of distinct
!     values for lower and upper bounds.

      nactx     =  0
      nval1     =  0
      nval3     =  0
      b_unique  = NO

!     Initialize the flag for distinct values of the gradient.

      g_details = .FALSE.

!     Initialize the counts of distinct values for the starting point.

      s_unique  = NO
      nval2     =  0
      s_details = .TRUE.

!     Assume, for now, that no variable is eliminated.

      all_x_active = .TRUE.

!     Verify the variables.

      DO j = 1, prob%n

!        Avoid eliminated variables, if the variable status is active.
!        In this case, the gradient details are needed.

         IF ( use_X_status ) THEN
            IF ( prob%X_status( j ) <= ELIMINATED ) THEN
               all_x_active = .FALSE.
               g_details    = .TRUE.
               CYCLE
            END IF
         END IF

!        Count the active variables

         nactx = nactx + 1

!        See if the function contains linear or quadratic terms, in which case
!        the type of the objective function is updated.

         IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
           IF ( cobj == 'C' ) cobj = 'L'
         ELSE
           IF ( prob%G( j ) /= ZERO .AND. cobj == 'C' ) cobj = 'L'
         END IF
         IF ( .NOT. PRESENT( no_H ) ) THEN
            DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
               IF ( prob%H%val( k ) == ZERO ) CYCLE
               i = prob%H%col( k )
               IF ( use_X_status ) THEN
                  IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
               END IF
               IF ( i /= j ) THEN
                  cobj = 'Q'
               ELSE IF ( cobj == 'L' .OR. cobj == 'C' ) THEN
                  cobj = 'S'
               END IF
            END DO
         END IF

!        Update the indicator for the type of constraints, if a finite bound
!        is met.

         xlj = prob%X_l( j )
         xuj = prob%X_u( j )
         IF ( xlj > - infinity .OR. xuj < infinity ) ccon = 'B'

!        Count the number of different values of the lower and upper bounds.

         CALL QPT_accum_vals( nval1, cval1, ival1, val1, NVALUES, j, xlj )
         CALL QPT_accum_vals( nval3, cval3, ival3, val3, NVALUES, j, xuj )

!        See if there is a unique value for the gradient.

         IF ( prob%gradient_kind /= 0 .AND. prob%gradient_kind /= 1 ) THEN
           IF ( all_x_active ) THEN
              IF ( .NOT. g_details ) g_details = prob%G( j ) /= prob%G( 1 )
           END IF
         END IF

!        Count the number of different values of the starting point.

         CALL QPT_accum_vals( nval2, cval2, ival2, val2, NVALUES, j, prob%X(j) )

      END DO

!     Is there a suitable default for the lower bound?

      l_details = .TRUE.
      IF ( nactx > 0 .AND. nval1 <= NVALUES ) THEN
         imy       = MAXLOC( cval1( 1 : nval1 ) )
         imx       = imy( 1 )
         l_details = cval1( imx ) < nactx
         bl0       = prob%X_l( ival1( imx ) )
      END IF

!     Is there a suitable default for the upper bound?

      u_details = .TRUE.
      IF ( nactx > 0 .AND. nval3 <= NVALUES ) THEN
         imy       = MAXLOC( cval3( 1 : nval3 ) )
         imx       = imy( 1 )
         u_details = cval3( imx ) < nactx
         bu0       = prob%X_u( ival3( imx ) )
      END IF

!     Find the default for the bounds, if possible.

      IF ( nval1 <= NVALUES ) THEN
         IF ( nval3 <= NVALUES ) THEN
            IF ( bl0 == bu0 ) THEN                                 ! fixed
               b_unique  = FIXED
            ELSE
               IF ( bl0 > -infinity ) THEN
                  IF ( bu0 < infinity ) THEN                       ! range
                     b_unique  = BOTH
                  ELSE                                             ! lower bound
                     b_unique  = LOWER
                  END IF
               ELSE
                  IF ( bu0 < infinity ) THEN                       ! upper bound
                     b_unique  = UPPER
                  ELSE                                             ! free
                     b_unique  = FREE
                  END IF
               END IF
            END IF
         ELSE
            IF ( bl0 > - infinity ) THEN
               b_unique  = LOWER                                   ! lower bound
            END IF
         END IF
      ELSE
         IF ( nval3 <= NVALUES ) THEN
            IF ( bu0 < infinity ) THEN                             ! upper bound
               b_unique  = UPPER
            END IF
         END IF
      END IF

!     Is there a suitable default for the starting point?

      IF ( nactx > 0 .AND. nval2 <= NVALUES ) THEN
         imy       = MAXLOC( cval2( 1 : nval2 ) )
         imx       = imy( 1 )
         s_unique  = YES
         s_details = cval2( imx ) < nactx
         x0        = prob%X( ival2( imx ) )
      END IF

!     Initialize the number of active constraints and the count of distinct
!     constant values.

      nactc     = 0
      nval1     = 0
      c_unique  = NO
      c_details = .TRUE.

!     Initialize the number of active ranges and the count of their distinct
!     values.

      nactr     = 0
      nval2     = 0
      r_unique  = NO
      r_details = .TRUE.

!     Verify the constraints.

      IF ( prob%A%ptr( prob%m + 1 ) > 1 ) THEN
         DO i = 1, prob%m

!           Avoid eliminated constraints, if the constraints status is active.

            IF ( use_C_status ) THEN
               IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
            END IF

!           Count the active constraints.

            nactc = nactc + 1

!           See if a nontrivial linear constraint is met.

            DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               IF ( prob%A%val( k ) == ZERO ) CYCLE
               j = prob%A%col( k )
               IF ( use_X_status ) THEN
                  IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
               END IF
               ccon = 'L'
               EXIT
            END DO

!           Count the number of different values of the constants

            cli = prob%C_l( i )
            cui = prob%C_u( i )
            IF ( cli > -infinity ) THEN
               CALL QPT_accum_vals( nval1, cval1, ival1, val1, NVALUES, i, cli )
            ELSE IF ( cui < infinity ) THEN
               CALL QPT_accum_vals( nval1, cval1, ival1, val1, NVALUES, i, cui )
            END IF

!           Count the number of different values for the ranges.
!           Note that we only have to consider the upper ranges, since
!           any finite lower bound is considered as the constant associated
!           with the constraint.

            IF ( cli > -infinity .AND. cui < infinity .AND. cli < cui ) THEN
               nactr = nactr + 1
               CALL QPT_accum_vals( nval2, cval2, ival2, val2, NVALUES, i, &
                                    cui - cli )
            END IF
         END DO
      END IF

!     Is there a suitable default for the constants?

      c_details = .TRUE.
      IF ( nactc > 0 .AND. nval1 <= NVALUES ) THEN
         imy      = MAXLOC( cval1( 1 : nval1 ) )
         imx      = imy( 1 )
         k        = ival1( imx )
         IF ( prob%C_l( k ) > - infinity ) THEN
            c0 = prob%C_l( k )
         ELSE
            c0 = prob%C_u( k )
         END IF

!        Set the default only if it is consistent with the objective function

         IF ( c0 == -prob%f ) THEN
            c_unique  = YES
            c_details = cval1( imx ) < nactc
         END IF
      END IF

!     Is there a suitable default for the ranges?

      r_details = .TRUE.
      IF ( nactr > 0 .AND. nactr == prob%m .AND. nval2 <= NVALUES ) THEN
         r_unique  = YES
         imy       = MAXLOC( cval2( 1 : nval2 ) )
         imx       = imy( 1 )
         r0        = prob%C_u( ival2( imx ) )
         r_details = cval2( imx ) < nactr
      END IF

!-------------------------------------------------------------------------------

!                                 Header

!-------------------------------------------------------------------------------

!     Open the output file.

      OPEN( UNIT = out, FILE = filename, IOSTAT = iostat, STATUS = 'UNKNOWN' )
      IF ( iostat /= 0 ) RETURN

!     Write the problem name

      WRITE( out, 100 ) probname

!     Write a comment indicating the type of problem and its date of creation.

      CALL DATE_AND_TIME( DATE = date )
      SELECt CASE ( cobj )
      CASE ( 'Q' )
         SELECT CASE ( ccon )
         CASE ( 'L' )
            WRITE( out, 101 )
         CASE ( 'B' )
            WRITE( out, 102 )
         CASE ( 'U' )
            WRITE( out, 103 )
         END SELECT
      CASE ( 'S' )
         SELECT CASE ( ccon )
         CASE ( 'L' )
            WRITE( out, 104 )
         CASE ( 'B' )
            WRITE( out, 105 )
         CASE ( 'U' )
            WRITE( out, 106 )
         END SELECT
      CASE ( 'L' )
         WRITE( out, 107 )
      END SELECT
      WRITE( out, 110 )  date( 7:8 ), date( 5:6 ), date( 3:4 )

!     Write the classification.

      fmt = '( ''*   classification '',A1,A1,''R2-AN-'' '
      SELECT CASE( nactx )
      CASE ( 0:9 )
         fmt = TRIM( fmt ) // ',I1,' // '''-'''
      CASE ( 10:99 )
         fmt = TRIM( fmt ) // ',I2,' // '''-'''
      CASE ( 100:999 )
         fmt = TRIM( fmt ) // ',I3,' // '''-'''
      CASE ( 1000:9999 )
         fmt = TRIM( fmt ) // ',I4,' // '''-'''
      CASE ( 10000:99999 )
         fmt = TRIM( fmt ) // ',I5,' // '''-'''
      CASE DEFAULT
         fmt = TRIM( fmt ) // ',I6,' // '''-'''
      END SELECT
      SELECT CASE ( nactc )
      CASE ( 0:9 )
         fmt = TRIM( fmt ) // ',I1)'
      CASE ( 10:99 )
         fmt = TRIM( fmt ) // ',I2)'
      CASE ( 100:999 )
         fmt = TRIM( fmt ) // ',I3)'
      CASE ( 1000:9999 )
         fmt = TRIM( fmt ) // ',I4)'
      CASE ( 10000:99999 )
         fmt = TRIM( fmt ) // ',I5)'
      CASE DEFAULT
         fmt = TRIM( fmt ) // ',I6)'
      END SELECT
      WRITE( out, fmt ) cobj, ccon, nactx, nactc

!     Avoid too short loops to describe the variables.

      all_x_active = all_x_active .AND. nactx >  3
      g_details    = g_details    .OR.  nactx <= 3

!     Write the necessary constants for the loops, when appropriate.

      IF ( prob%gradient_kind == 0 ) THEN
        IF ( all_x_active .OR. .NOT. g_details  ) THEN
           WRITE( out, '( /, '' IE 1                   '', I10 )' ) 1
           WRITE( out, '(    '' IE N                   '', I10 )' ) prob%n
        END IF
      ELSE IF ( prob%gradient_kind == 1 ) THEN
         WRITE( out, '( /, '' IE 1                   '', I10 )' ) 1
         WRITE( out, '(    '' IE N                   '', I10 )' ) prob%n
      ELSE
        IF ( all_x_active                                  .OR. &
             ( .NOT. g_details .AND. prob%G( 1 ) /= ZERO )    ) THEN
           WRITE( out, '( /, '' IE 1                   '', I10 )' ) 1
           WRITE( out, '(    '' IE N                   '', I10 )' ) prob%n
        END IF
      END IF

!-------------------------------------------------------------------------------

!                             Variables

!-------------------------------------------------------------------------------

      WRITE( out, '( /, ''VARIABLES'', / )' )

      IF ( all_x_active ) THEN
         WRITE( out, '( '' DO J         1                        N'')' )
         WRITE( out, '( '' X  X(J)'' )' )
         WRITE( out, '( '' OD J'' )' )
      ELSE
         DO j = 1, prob%n
            IF ( use_X_status ) THEN
               IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
            END IF
            CALL QPT_write_field( 'X', j, f2 )
            WRITE( out, 1000 ) '  ', f2
         END DO
      END IF

!-------------------------------------------------------------------------------

!                             The groups

!-------------------------------------------------------------------------------

      WRITE( out, '( /, ''GROUPS'' )' )

!     The objective function

      IF ( nactx > 0 ) WRITE( out, '( '' '' )' )

      IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 )              &
        g_details = .FALSE.
      IF ( g_details ) THEN
         side     = LEFT
         notfound = .TRUE.           ! all gradient components are zero so far
         DO j = 1, prob%n
            IF ( use_X_status ) THEN
               IF ( prob%x_status( j ) <= ELIMINATED ) CYCLE
            END IF
            a = prob%G( j )
            IF ( a == ZERO ) CYCLE
            notfound = .FALSE.       ! a nontrivial gradient component is found
            IF ( side == LEFT ) THEN
               CALL QPT_write_field( 'X', j, f3 )
               CALL QPT_trim_real( a, f4 )
               side = RIGHT
            ELSE
               CALL QPT_write_field( 'X', j, f5 )
               CALL QPT_trim_real( a, f6 )
               WRITE( out, 1000 ) 'N ', 'OBJ       ', f3, f4, f5, f6
               side = LEFT
            END IF
         END DO
         IF ( side == RIGHT ) WRITE( out, 1000 ) 'N ', 'OBJ       ', f3, f4
         IF ( notfound ) WRITE( out, 1000 ) 'N ', 'OBJ       '
      ELSE
         IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
           IF ( prob%gradient_kind == 1 ) THEN
              CALL QPT_trim_real( ONE, f4 )
              WRITE( out, '( '' DO J         1                        N'')' )
              WRITE( out, 1000 ) 'XN', 'OBJ       ', 'X(J)      ', f4
              WRITE( out, '( '' OD J'' )' )
           ELSE
              WRITE( out, 1000 ) 'N ', 'OBJ       '
           END IF
         ELSE
           IF ( prob%G( 1 ) /= ZERO ) THEN
              CALL QPT_trim_real( prob%G( 1 ), f4 )
              WRITE( out, '( '' DO J         1                        N'')' )
              WRITE( out, 1000 ) 'XN', 'OBJ       ', 'X(J)      ', f4
              WRITE( out, '( '' OD J'' )' )
           ELSE
              WRITE( out, 1000 ) 'N ', 'OBJ       '
           END IF
         END IF
      END IF

!     The constraints

      IF ( ccon == 'L' ) THEN
         WRITE( out, '( '' '' )' )
         DO i = 1, prob%m
            IF ( use_C_status ) THEN
               IF (  prob%C_status( i ) <= ELIMINATED ) CYCLE
            END IF

!           Determine the constraint type.

            IF ( prob%C_l( i ) > -infinity ) THEN
               IF ( prob%C_u( i ) == prob%C_l( i ) ) THEN
                  f1 = 'E '
               ELSE
                  f1 = 'G '
               END IF
            ELSE IF ( prob%C_u( i ) < infinity ) THEN
               f1 = 'L '
            ELSE
               CYCLE
            END IF
            CALL QPT_write_field( 'C', i, f2 )

!           Write the coefficients.

            written = .FALSE.
            side = LEFT
            DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               j = prob%A%col( k )
               IF ( use_X_status ) THEN
                  IF ( prob%x_status( j ) <= ELIMINATED ) CYCLE
               END IF
               a = prob%A%val( k )
               IF ( a  == ZERO ) CYCLE
               written = .TRUE.
               IF ( side == LEFT ) THEN
                  CALL QPT_write_field( 'X', j, f3 )
                  CALL QPT_trim_real( a, f4 )
                  side = RIGHT
               ELSE
                  CALL QPT_write_field( 'X', j, f5 )
                  CALL QPT_trim_real( a, f6 )
                  WRITE( out, 1000 ) f1, f2, f3, f4, f5, f6
                  side = LEFT
               END IF
            END DO
            IF ( side == RIGHT ) WRITE( out, 1000 ) f1, f2, f3, f4

!           Write a dummy coefficient if the constraint is empty.

            IF ( .NOT. written ) THEN
               CALL QPT_write_field( 'X', 1, f3 )
               CALL QPT_trim_real( ZERO, f4 )
               WRITE( out, 1000)  f1, f2, f3, f4
            END IF
         END DO
      END IF

!-------------------------------------------------------------------------------

!                             The constants

!-------------------------------------------------------------------------------

      IF ( nactc > 0 .OR. prob%f /= ZERO ) THEN

         WRITE( out, '( /, ''CONSTANTS'', / )' )

!        The linear constraints

!        The default is set: write it.

         IF ( c_unique == YES ) THEN
            IF ( c0 /= ZERO ) THEN
               CALL QPT_trim_real( c0, f4 )
               WRITE( out, 1000 ) '  ', probname, '''DEFAULT'' ', f4
               WRITE( out, '( '' '' )' )
            END IF
         END IF

!        The constant objective function term
!        (negated to reflect SIF definition of constants).

         IF (  ( c_unique == NO  .AND.  prob%f /= ZERO ) .OR. &
               ( c_unique == YES .AND. -prob%f /= c0   )      ) THEN
            CALL QPT_trim_real( -prob%f, f4 )
            WRITE( out, 1000 ) '  ', probname, 'OBJ       ', f4
         END IF

!        No default

         IF ( c_details ) THEN
            side = LEFT
            DO i = 1, prob%m
               IF ( use_C_status ) THEN
                  IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
               END IF
               IF ( prob%C_l( i ) > -infinity ) THEN
                  a = prob%C_l( i )
               ELSE IF ( prob%C_u( i ) < infinity ) THEN
                  a = prob%C_u( i )
               ELSE
                  CYCLE
               END IF
               IF ( c_unique == YES ) THEN
                  IF ( a == c0 ) CYCLE
               END IF
               IF ( side == LEFT ) THEN
                  CALL QPT_write_field( 'C', i, f3 )
                  CALL QPT_trim_real( a, f4 )
                  side = RIGHT
               ELSE
                  CALL QPT_write_field( 'C', i, f5 )
                  CALL QPT_trim_real( a, f6 )
                  WRITE( out, 1000 ) '  ', probname, f3, f4, f5, f6
                  side = LEFT
               END IF
            END DO
            IF ( side == RIGHT ) WRITE( out, 1000 ) '  ', probname, f3, f4
         END IF
      END IF

!-------------------------------------------------------------------------------

!                             The ranges

!-------------------------------------------------------------------------------

      IF ( nactr > 0 ) THEN

         WRITE( out, '( /, ''RANGES'', / )' )

!        The default is set.

         IF ( r_unique == YES ) THEN
            CALL QPT_trim_real( r0, f4 )
            WRITE( out, 1000 ) '  ', probname, '''DEFAULT'' ', f4
         END IF

!        No default

         IF ( r_details ) THEN
            side = LEFT
            DO i = 1, prob%m
               IF ( use_C_status ) THEN
                  IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
               END IF
               IF ( prob%C_l( i ) > -infinity ) THEN
                  IF ( prob%C_l( i ) == prob%C_u( i ) ) CYCLE
                  IF ( prob%C_u( i ) < infinity ) THEN
                     a = prob%C_u( i ) - prob%C_l( i )
                  ELSE
                     CYCLE
                  END IF
               ELSE
                  CYCLE
               END IF
               IF ( r_unique == YES ) THEN
                  IF ( a == r0 ) CYCLE
               END IF
               IF ( side == LEFT ) THEN
                  CALL QPT_write_field( 'C', i, f3 )
                  CALL QPT_trim_real( a, f4 )
                  side = RIGHT
               ELSE
                  CALL QPT_write_field( 'C', i, f5 )
                  CALL QPT_trim_real( a, f6 )
                  WRITE( out, 1000 ) '  ', probname, f3, f4, f5, f6
                  side = LEFT
               END IF
            END DO
            IF ( side == RIGHT ) WRITE( out, 1000 ) '  ', probname, f3, f4
         END IF
      END IF

!-------------------------------------------------------------------------------

!                             The bounds

!-------------------------------------------------------------------------------

      IF ( nactx > 0 ) THEN

         written = .FALSE.                      ! section title not yet written

         SELECT CASE ( b_unique )

!        Write the default if all variables are free.

         CASE ( FREE )
            WRITE( out, '( /, ''BOUNDS'', / )' )
            written = .TRUE.
            WRITE( out, 1000 ) 'FR', probname, '''DEFAULT'' '

!        Write the default if all variables are fixed.

         CASE ( FIXED )
            WRITE( out, '( /, ''BOUNDS'', / )' )
            written = .TRUE.                   ! section title just written
            CALL QPT_trim_real( bl0, f4 )
            WRITE( out, 1000 ) 'FX', probname, '''DEFAULT'' ', f4

!        Write the default if the lower bound has a non zero default.

         CASE ( LOWER )
            IF ( bl0 /= ZERO ) THEN
               WRITE( out, '( /, ''BOUNDS'', / )' )
               written = .TRUE.                ! section title just written
               CALL QPT_trim_real( bl0, f4 )
               WRITE( out, 1000 ) 'LO', probname, '''DEFAULT'' ', f4
            END IF

!        Write the default if the upper bound has a finite default.

         CASE ( UPPER )
            IF ( bu0 < infinity ) THEN
               WRITE( out, '( /, ''BOUNDS'', / )' )
               written = .TRUE.                ! section title just written
               CALL QPT_trim_real( bu0, f4 )
               WRITE( out, 1000 ) 'UP', probname, '''DEFAULT'' ', f4
            END IF

!        Write the defaults if both bounds have defaults.

         CASE ( BOTH )
            IF ( bl0 /= ZERO ) THEN
               WRITE( out, '( /, ''BOUNDS'', / )' )
               written = .TRUE.                ! section title just written
               CALL QPT_trim_real( bl0, f4 )
               WRITE( out, 1000 ) 'LO', probname, '''DEFAULT'' ', f4
            END IF
            IF ( bu0 < infinity ) THEN
               IF ( .NOT. written ) THEN
                  WRITE( out, '( /, ''BOUNDS'', / )' )
                  written  = .TRUE.            ! section title just written
               END IF
               CALL QPT_trim_real( bu0, f4 )
               WRITE( out, 1000 ) 'UP', probname, '''DEFAULT'' ', f4
            END IF

         END SELECT

!        Not all variables are at the default value(s).

         IF ( l_details .OR. u_details ) THEN

!           Write the section title, if not already done.

            IF ( .NOT. written ) WRITE( out, '( /, ''BOUNDS'', / )' )

!           Write the remaining bounds.

            DO j = 1, prob%n
               IF ( use_X_status ) THEN
                  IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
               END IF
               xlj = prob%X_l( j )
               xuj = prob%X_u( j )
               CALL QPT_write_field( 'X', j, f3 )
               IF ( xlj > - infinity ) THEN
                  IF ( xuj < infinity ) THEN
                     IF ( xlj == xuj ) THEN                           ! fixed
                        IF ( b_unique /= FIXED .OR. xlj /= bl0 ) THEN
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'FX', probname, f3, f4
                        END IF
                     ELSE                                             ! range
                        SELECT CASE ( b_unique )
                        CASE ( FIXED )
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        CASE ( LOWER )
                           IF ( xlj /= bl0 ) THEN
                              CALL QPT_trim_real( xlj, f4 )
                              WRITE( out, 1000 ) 'LO', probname, f3, f4
                           END IF
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        CASE ( BOTH )
                           IF ( xlj /= bl0 ) THEN
                              CALL QPT_trim_real( xlj, f4 )
                              WRITE( out, 1000 ) 'LO', probname, f3, f4
                           END IF
                           IF ( xuj /= bu0 ) THEN
                              CALL QPT_trim_real( xuj, f4 )
                              WRITE( out, 1000 ) 'UP', probname, f3, f4
                           END IF
                        CASE ( UPPER )
                           IF ( xlj /= ZERO ) THEN
                              CALL QPT_trim_real( xlj, f4 )
                              WRITE( out, 1000 ) 'LO', probname, f3, f4
                           END IF
                           IF ( xuj /= bu0 ) THEN
                              CALL QPT_trim_real( xuj, f4 )
                              WRITE( out, 1000 ) 'UP', probname, f3, f4
                           END IF
                        CASE ( FREE )
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        CASE DEFAULT
                           IF ( xlj /= ZERO ) THEN
                              CALL QPT_trim_real( xlj, f4 )
                              WRITE( out, 1000 ) 'LO', probname, f3, f4
                           END IF
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        END SELECT
                     END IF
                  ELSE                                                ! lower
                     SELECT CASE ( b_unique )
                     CASE ( FIXED )
                        IF ( xlj < xuj ) THEN
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                           CALL QPT_trim_real( MAX( TEN * infinity,            &
                                                    TEN ** 21 ), f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        END IF
                     CASE ( LOWER )
                        IF ( xlj /= bl0 ) THEN
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                        END IF
                     CASE ( BOTH )
                        IF ( xlj /= bl0 ) THEN
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                        END IF
                     CASE ( FREE )
                        CALL QPT_trim_real( xlj, f4 )
                        WRITE( out, 1000 ) 'LO', probname, f3, f4
                     CASE DEFAULT
                        IF ( xlj /= ZERO ) THEN
                           CALL QPT_trim_real( xlj, f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                        END IF
                     END SELECT
                  END IF
               ELSE
                  IF ( xuj < infinity ) THEN                          ! upper
                     SELECT CASE ( b_unique )
                     CASE ( FIXED )
                        IF ( xlj < xuj ) THEN
                           CALL QPT_trim_real( - MAX( TEN * infinity,          &
                                                      TEN ** 21 ), f4 )
                           WRITE( out, 1000 ) 'LO', probname, f3, f4
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        END IF
                     CASE ( UPPER )
                        IF ( xuj /= bu0 ) THEN
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        END IF
                     CASE ( BOTH )
                        IF ( xuj /= bu0 ) THEN
                           CALL QPT_trim_real( xuj, f4 )
                           WRITE( out, 1000 ) 'UP', probname, f3, f4
                        END IF
                     CASE ( FREE )
                        CALL QPT_trim_real( xuj, f4 )
                        WRITE( out, 1000 ) 'UP', probname, f3, f4
                     CASE DEFAULT
                        CALL QPT_trim_real( xuj, f4 )
                        WRITE( out, 1000 ) 'UP', probname, f3, f4
                     END SELECT
                  ELSE                                                ! free
                     IF ( b_unique /= FREE ) &
                        WRITE( out, 1000 ) 'FR', probname, f3
                  END IF
               END IF
            END DO
         END IF
      END IF

!-------------------------------------------------------------------------------

!                             The starting point

!-------------------------------------------------------------------------------


      IF ( nactx > 0 ) THEN

         IF ( s_unique == YES .AND. x0 /= ZERO ) THEN
            WRITE( out, '( /, ''START POINT'', / )' )
            CALL QPT_trim_real( x0, f4 )
            WRITE( out, 1000 ) 'V ', probname, '''DEFAULT'' ', f4
         ELSE IF ( s_details ) THEN
            WRITE( out, '( /, ''START POINT'', / )' )
         END IF

         IF ( s_details ) THEN
            side = LEFT
            DO j = 1, prob%n
               IF ( use_X_status ) THEN
                  IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
               END IF
               IF ( s_unique == YES ) THEN
                  IF ( prob%X( j ) == x0 ) CYCLE
               END IF
               IF ( side == LEFT ) THEN
                  CALL QPT_write_field( 'X', j, f3 )
                  CALL QPT_trim_real( prob%X( j ), f4 )
                  side = RIGHT
               ELSE
                  CALL QPT_write_field( 'X', j, f5 )
                  CALL QPT_trim_real( prob%X( j ), f6 )
                  WRITE( out, 1000 ) 'V ', probname, f3, f4, f5, f6
                  side = LEFT
               END IF
            END DO
            IF ( side == RIGHT ) WRITE( out, 1000 ) 'V ', probname, f3, f4
         END IF
      END IF

!-------------------------------------------------------------------------------

!                         The quadratic terms

!-------------------------------------------------------------------------------

      IF ( .NOT. PRESENT( no_H ) ) THEN
         IF ( cobj == 'Q' .OR. cobj == 'S' ) THEN
            WRITE( out, '( /, ''QUADRATIC'', / )' )
            DO j = 1, prob%n
               IF ( use_X_status ) THEN
                  IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
               END IF
               CALL QPT_write_field( 'X', j, f2 )
               side = LEFT
               DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
                  a = prob%H%val( k )
                  IF ( a == ZERO ) CYCLE
                  i = prob%H%col( k )
                  IF ( use_X_status ) THEN
                     IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
                  END IF
                  IF ( side == LEFT ) THEN
                     CALL QPT_write_field( 'X', i, f3 )
                     CALL QPT_trim_real( a, f4 )
                     side = RIGHT
                  ELSE
                     CALL QPT_write_field( 'X', i, f5 )
                     CALL QPT_trim_real( a, f6 )
                     WRITE( out, 1000 ) '  ', f2, f3, f4, f5, f6
                     side = LEFT
                  END IF
               END DO
               IF ( side == RIGHT ) WRITE( out, 1000 ) '  ', f2, f3, f4
            END DO
         END IF
      END IF

!-------------------------------------------------------------------------------

!                                  The end

!-------------------------------------------------------------------------------

      WRITE( out, '( /, ''ENDATA'' )' )
      CLOSE( out )

      RETURN

!     Formats

100   FORMAT( '***************************', / , &
              '* SET UP THE INITIAL DATA *', / , &
              '***************************', //, &
              'NAME          ', A8,          //, &
              '*   Problem :',               / , &
              '*   *********',               /   )
101   FORMAT( '*   A quadratic programming problem' )
102   FORMAT( '*   A bound constrained quadratic programming problem' )
103   FORMAT( '*   An unconstrained quadratic problem' )
104   FORMAT( '*   A linearly constrained linear least-squares problem' )
105   FORMAT( '*   A bound constrained linear least-squares problem' )
106   FORMAT( '*   An unconstrained linear least-squares problem' )
107   FORMAT( '*   A linear programming problem' )
110   FORMAT( '*   (automatically generated by GALAHAD on ',      &
              A2, '/', A2, '/', A2, ')', / )
1000  FORMAT( 1x, A2, 1x, A10, A10, A12, 3x, A10, A12 )

      END SUBROUTINE QPT_write_to_sif

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_write_field( c, i, field )

!     Writes a CHARACTER( 10 ) field consisting of the character c followed
!     by the integer i and padded on the right with blanks

      CHARACTER( 1 ), INTENT( IN ) :: c

!            the first character of the field

      INTEGER, INTENT( IN ) :: i

!            the integer to write in the field

      CHARACTER( 10 ), INTENT( OUT ) :: field

!            the resulting field.

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

      SELECT CASE ( i )
      CASE ( 0:9 )
         WRITE( field, '( A1, I1, 8x )' ) c, i
      CASE ( 10:99 )
         WRITE( field, '( A1, I2, 7x )' ) c, i
      CASE ( 100:999 )
         WRITE( field, '( A1, I3, 6x )' ) c, i
      CASE ( 1000:9999 )
         WRITE( field, '( A1, I4, 5x )' ) c, i
      CASE ( 10000:99999 )
         WRITE( field, '( A1, I5, 4x )' ) c, i
      CASE ( 100000:999999 )
         WRITE( field, '( A1, I6, 3x )' ) c, i
      CASE ( 1000000:9999999 )
         WRITE( field, '( A1, I7, 2x )' ) c, i
      CASE DEFAULT
         WRITE( field, '( A1, I8, 1x )' ) c, i
      END SELECT

      RETURN

      END SUBROUTINE QPT_write_field

!===============================================================================

      SUBROUTINE QPT_trim_real( r, field )

!     Transform the real r into a neat(?) character( 12 ) field

      REAL ( KIND = wp ), INTENT( IN ) :: r

!            the real to transform,

      CHARACTER( 12 ), INTENT( OUT ) :: field

!            the resulting string.

!     Programming: Ph. Toint, June 2001.
!
!===============================================================================

!     Local variables

      INTEGER :: p, l
      REAL( KIND = wp ), PARAMETER :: ONE   = 1.0_wp
      REAL( KIND = wp ), PARAMETER :: TEN   = 10.0_wp
      REAL( KIND = wp ), PARAMETER :: TENM4 = TEN ** ( -4 )
      REAL( KIND = wp ), PARAMETER :: TEN2  = TEN ** 2
      REAL( KIND = wp ), PARAMETER :: TEN3  = TEN ** 3
      REAL( KIND = wp ), PARAMETER :: TEN4  = TEN ** 4
      REAL( KIND = wp ), PARAMETER :: TEN5  = TEN ** 5
      REAL( KIND = wp ), PARAMETER :: TEN6  = TEN ** 6
      REAL( KIND = wp ), PARAMETER :: TEN7  = TEN ** 7
      REAL( KIND = wp ), PARAMETER :: TEN8  = TEN ** 8
      REAL( KIND = wp ), PARAMETER :: TEN9  = TEN ** 9
      REAL( KIND = wp ), PARAMETER :: TEN10 = TEN ** 10

!     Write the real into a string.

      IF ( r >= 0.0_wp ) THEN
         IF ( r == 0.0_wp .OR. ( r >= TENM4 .AND. r < ONE  ) ) THEN
            WRITE( field, '(F12.10)' ) r
         ELSE IF ( r >= ONE  .AND. r < TEN  ) THEN
            WRITE( field, '(F12.10)' ) r
         ELSE IF ( r >= TEN  .AND. r < TEN2 ) THEN
            WRITE( field, '(F12.9)' ) r
         ELSE IF ( r >= TEN2 .AND. r < TEN3 ) THEN
            WRITE( field, '(F12.8)' ) r
         ELSE IF ( r >= TEN3 .AND. r < TEN4 ) THEN
            WRITE( field, '(F12.7)' ) r
         ELSE IF ( r >= TEN4 .AND. r < TEN5 ) THEN
            WRITE( field, '(F12.6)' ) r
         ELSE IF ( r >= TEN5 .AND. r < TEN6 ) THEN
            WRITE( field, '(F12.5)' ) r
         ELSE IF ( r >= TEN6 .AND. r < TEN7 ) THEN
            WRITE( field, '(F12.4)' ) r
         ELSE IF ( r >= TEN7 .AND. r < TEN8 ) THEN
            WRITE( field, '(F12.3)' ) r
         ELSE IF ( r >= TEN8 .AND. r < TEN9 ) THEN
            WRITE( field, '(F12.2)' ) r
         ELSE IF ( r >= TEN9 .AND. r < TEN10 ) THEN
            WRITE( field, '(F12.1)' ) r
!        ELSE IF ( r >= TEN9 .AND. r < TEN10 ) THEN
!           WRITE( field, '(F12.0)' ) r
         ELSE
            WRITE( field, '(ES12.4)' ) r
         END IF
      ELSE
         IF ( r <= - TENM4 .AND. r > - ONE ) THEN
            WRITE( field, '(F12.9)' ) r
         ELSE IF ( r <= - ONE  .AND. r > - TEN  ) THEN
            WRITE( field, '(F12.8)' ) r
         ELSE IF ( r <= - TEN  .AND. r > - TEN2 ) THEN
            WRITE( field, '(F12.7)' ) r
         ELSE IF ( r <= - TEN2 .AND. r > - TEN3 ) THEN
            WRITE( field, '(F12.6)' ) r
         ELSE IF ( r <= - TEN3 .AND. r > - TEN4 ) THEN
            WRITE( field, '(F12.5)' ) r
         ELSE IF ( r <= - TEN4 .AND. r > - TEN5 ) THEN
            WRITE( field, '(F12.4)' ) r
         ELSE IF ( r <= - TEN5 .AND. r > - TEN6 ) THEN
            WRITE( field, '(F12.3)' ) r
         ELSE IF ( r <= - TEN6 .AND. r > - TEN7 ) THEN
            WRITE( field, '(F12.2)' ) r
         ELSE IF ( r <= - TEN7 .AND. r > - TEN8 ) THEN
            WRITE( field, '(F12.1)' ) r
!        ELSE IF ( r <= - TEN8 .AND. r > - TEN9 ) THEN
!           WRITE( field, '(F12.1)' ) r
!        ELSE IF ( r <= - TEN8 .AND. r > - TEN9 ) THEN
!           WRITE( field, '(F12.0)' ) r
         ELSE
            WRITE( field, '(ES12.5)' ) r
         END IF
      END IF

!     Remove trailing E+00.

      p = INDEX( field, 'E+00' )
      IF ( p > 0 ) field = field( 1 : p - 1 )

!     Remove useless zeros.

      p = INDEX( field, 'E' )
      l = LEN_TRIM( field )
      IF ( p > 0 ) THEN
         p = p - 1
      ELSE
         p = l
      END IF
      DO
         IF ( field( p : p ) == '0' ) THEN
            IF ( l == p ) THEN
               field = field( 1 : p - 1 )
               p = p - 1
               l = p
            ELSE
               field = field( 1 : p - 1 ) // field( p + 1 : l )
               p = p - 1
               l = l - 1
            END IF
         ELSE
            EXIT
         END IF
      END DO

!     Adjust on the left.

      field = ADJUSTR( field )

      RETURN

      END SUBROUTINE QPT_trim_real

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_accum_vals( nval, cval, ival, val, NVALUES, idx, value )

!     Accumulate counts of the different values met in a vector, up to a
!     maximum of NVALUES. The successive distinct values are stored in
!     val( 1:nval ), the position of their first occurrence in the vector
!     in ival( 1:nval ), the number of times they have been found so far in
!     cval( 1:nval ), where nval <= NVALUES.  The new value "value", whose
!     position in the vector is idx, is compared to all the previously distinct
!     values found so far: if it corresponds to one of the previous values, the
!     occurrence count of that value is incremented by one; otherwise, it is
!     stored in val, cval and ival are updated, and nval is incremented by one,
!     unless nval > NVALUES (in which case no action is taken except
!     incrementing nval to NVALUES + 1).

!     Arguments:

      INTEGER, INTENT( INOUT ) :: nval

!            input : the number of different values found so far in the vector
!            output: unmodified, if nval is larger than NVALUES on input, or
!                    if value is equal to one of the previously detected
!                    values stored in val, or incremented by one otherwise.

      INTEGER, INTENT( IN ) :: NVALUES

!            the maximum number of distinct values that are to be memorized
!            and whose occurrence must be counted,

      INTEGER, DIMENSION( NVALUES ), INTENT( INOUT ) :: cval

!            input : the number of times each of the values (from 1 to
!                    MIN( NVALUES, nval ) ) has been met so far in the vector
!            output : if value = val( i ), the cval( i ) is incremented by one,

      INTEGER, DIMENSION( NVALUES ), INTENT( INOUT ) :: ival

!            input : the position, in the vector, of the first occurrence of
!                    the value val( i ),
!            output: same, except that a new component may be added in
!                    value is not equal to any of the val( k ) for k from
!                    1 to nval(on input),

      REAL ( KIND = wp ), DIMENSION( NVALUES ), INTENT( INOUT ) :: val

!            input : the vector of distinct values found so far in the vector.

      INTEGER, INTENT( IN ) :: idx

!            the position, in the vector, of the value value,

      REAL ( KIND = wp ), INTENT( IN ) :: value

!            the new value to compare to already found distinct values in the
!            vector.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: k

      IF ( nval > NVALUES ) RETURN
      DO k = 1, nval
         IF ( value == val( k ) ) THEN
            cval( k ) = cval( k ) + 1
            RETURN
         END IF
      END DO
      nval = nval + 1
      IF ( nval <= NVALUES ) THEN
         cval( nval ) = 1
         ival( nval ) = idx
         val(  nval ) = value
      END IF

      RETURN

      END SUBROUTINE QPT_accum_vals

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_A_from_D_to_S( prob, exitcode )

!     Transforms the matrix A from dense storage to sparse-by-row format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( OUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, j, k, iostat, p

!     Allocate the pointer to the beginning of each row.

      ALLOCATE( prob%A%ptr( prob%m + 1 ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Count the number of nonzeros in A and set up the vector of pointers to
!     the beginning of each row.

      k = 0
      prob%A%ptr( 1 ) = 1
      DO i = 1, prob%m
         p = 0
         DO j = 1, prob%n
            k = k + 1
            IF ( prob%A%val( k ) /= ZERO ) p = p + 1
         END DO
         prob%A%ptr( i + 1 ) = prob%A%ptr( i ) + p
      END DO
      prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1

!     Allocate the pointer for the column indices.

      ALLOCATE( prob%A%col( prob%A%ne ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Remove the zeros from A and set up the column numbers.

      k = 0
      p = 0
      DO i = 1, prob%m
         DO j = 1, prob%n
            k = k + 1
            IF ( prob%A%val( k ) /= ZERO ) THEN
               p = p + 1
               prob%A%val( p ) = prob%A%val( k )
               prob%A%col( p ) = j
            END IF
         END DO
      END DO

!     Update the type of matrix.

      CALL QPT_put_A( prob%A%type, 'SPARSE_BY_ROWS' )

      RETURN

      END SUBROUTINE QPT_A_from_D_to_S

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_A_from_S_to_D( prob, exitcode )

!     Transforms the matrix A from sparse-by-row format to dense format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( OUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, j, nextj, k, pos, last, is
      REAL ( KIND = wp ) :: val, tmp
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: seen

!     Allocate marker workspace

      ALLOCATE( seen( prob%n ), STAT = is )
      IF ( is /= 0 ) THEN
          exitcode = MEMORY_FULL
          RETURN
      END IF

!     Sort the entries, row by row, starting from the last.

      last = prob%A%ptr( prob%m + 1 ) - 1
      DO i = prob%m, 1, -1
         seen = .FALSE.
         k    = prob%A%ptr( i )
         is   = ( i - 1 ) * prob%n
         DO
            IF ( k >= prob%A%ptr( i + 1 ) ) EXIT
            j = prob%A%col( k )
            IF ( j > 0 ) THEN
               val = prob%A%val( k )
               prob%A%col( k ) = -1
               DO
                  pos = is + j
                  tmp = prob%A%val( pos )
                  prob%A%val( pos ) = val
                  seen( pos ) = .TRUE.
                  IF ( pos > last ) EXIT
                  nextj = prob%A%col( pos )
                  IF ( nextj < 0 ) EXIT
                  val = tmp
                  prob%A%col( pos ) = -1
                  j = nextj
               END DO
            END IF
            k = k + 1
         END DO

!        Zero the entries that haven't been assigned a value.

         DO j = 1, prob%n
            IF ( seen( j ) ) CYCLE
            prob%A%val( is + j ) = ZERO
         END DO
      END DO

!     Deallocate the workspace.

      DEALLOCATE( seen )

!     Update the type of matrix.

      CALL QPT_put_A( prob%A%type, 'DENSE' )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_A_from_S_to_D

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_A_from_C_to_S( prob, exitcode )

!     Transform the coordinate storage to sparse by row for the matrix A.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER  :: k, i, ii, nnz, iostat

      IF ( prob%m <= 0 ) RETURN

!     Allocate the pointers to the beginning of each row.

      ALLOCATE( prob%A%ptr( prob%m + 1 ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Count the number of nonzero in each row.

      prob%A%ptr( :prob%m + 1 ) = 0
      DO k = 1, prob%A%ne
         IF ( prob%A%val( k ) /= ZERO ) THEN
            i = prob%A%row( k )
            prob%A%ptr( i ) = prob%A%ptr( i ) + 1
         END IF
      END DO

!     Assign the pointers to the beginning of each row.

      ii = 1
      DO i = 1, prob%m + 1
         k = prob%A%ptr( i )
         prob%A%ptr( i ) = ii
         ii = ii + k
      END DO

!     Build the permutation of the elements of A in the vector that contains
!     its row numbers.

      nnz = prob%A%ne
      DO k = 1, prob%A%ne
         IF ( prob%A%val( k ) /= ZERO ) THEN
            i  = prob%A%row( k )
            ii = prob%A%ptr( i )
            prob%A%row( k ) = ii
            prob%A%ptr( i ) = ii + 1
         ELSE
            prob%A%row( k ) = nnz
            nnz = nnz - 1
         END IF
      END DO

!     Rebuild the pointers to the beginning of each row.

      DO i = prob%m, 2, -1
         prob%A%ptr( i ) = prob%A%ptr( i - 1 )
      END DO
      prob%A%ptr( 1 ) = 1

!     Apply the permutation to the elements of A and their column numbers.

      CALL SORT_inplace_permute( prob%A%ne , prob%A%row, &
                                 x = prob%A%val, ix = prob%A%col  )

!     Deallocate the row vector.

      DEALLOCATE( prob%A%row )

!     Update the matrix type

      CALL QPT_put_A( prob%A%type, 'SPARSE_BY_ROWS' )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_A_from_C_to_S

!==============================================================================
!===============================================================================

      SUBROUTINE QPT_A_from_S_to_C( prob, exitcode )

!     Transforms the matrix A from sparse-by-row format to coordinate format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, k, iostat

!     Allocate the pointers for the row numbers.

      ALLOCATE( prob%A%row( prob%A%ne ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Assign the row numbers.

      DO i = 1, prob%m
         DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            prob%A%row( k ) = i
         END DO
      END DO
      prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
      CALL QPT_put_A( prob%A%type, 'COORDINATE' )

      DEALLOCATE( prob%A%ptr )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_A_from_S_to_C

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_H_from_S_to_D( prob, exitcode )

!     Transforms the lower triangle of the matrix H from sparse-by-row format
!     to dense format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, j, nextj, k, pos, last, is
      REAL ( KIND = wp ) :: val, tmp
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: seen

!     Allocate marker workspace.

      ALLOCATE( seen( prob%n ), STAT = is )
      IF ( is /= 0 ) THEN
          exitcode = MEMORY_FULL
          RETURN
      END IF

!     Sort the entries, row by row, starting from the last.

      last = prob%H%ptr( prob%n + 1 ) - 1
      DO i = prob%n, 1, -1
         seen = .FALSE.
         k    = prob%H%ptr( i )
         is   = ( ( i - 1 ) * i ) / 2
         DO
            IF ( k >= prob%H%ptr( i + 1 ) ) EXIT
            j = prob%H%col( k )
            IF ( j > 0 ) THEN
               val = prob%H%val( k )
               prob%H%col( k ) = -1
               DO
                  pos = is + j
                  tmp = prob%H%val( pos )
                  prob%H%val( pos ) = val
                  seen( pos ) = .TRUE.
                  IF ( pos > last ) EXIT
                  nextj = prob%H%col( pos )
                  IF ( nextj < 0 ) EXIT
                  val = tmp
                  prob%H%col( pos ) = -1
                  j = nextj
               END DO
            END IF
            k = k + 1
         END DO

!        Zero the entries that haven't been assigned a value.

         DO j = 1, i
            IF ( seen( j ) ) CYCLE
            prob%H%val( is + j ) = ZERO
         END DO
      END DO

!     Deallocate the workspace.

      DEALLOCATE( seen )

!     Update the type of matrix.

      CALL QPT_put_H( prob%H%type, 'DENSE' )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_H_from_S_to_D

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_H_from_D_to_S( prob, exitcode )

!     Transforms the matrix H from dense (lower triangular) storage to
!     sparse-by-row format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, j, k, iostat, p

!     Allocate the pointer to the beginning of each row.

      ALLOCATE( prob%H%ptr( prob%n + 1 ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Count the number of nonzeros in H and set up the vector of pointers to
!     the beginning of each row.

      k = 0
      prob%H%ptr( 1 ) = 1
      DO i = 1, prob%n
         p = 0
         DO j = 1, i
            k = k + 1
            IF ( prob%H%val( k ) /= ZERO ) p = p + 1
         END DO
         prob%H%ptr( i + 1 ) = prob%H%ptr( i ) + p
      END DO
      prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1

!     Allocate the pointer for the column indices.

      ALLOCATE( prob%H%col( prob%H%ne ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Remove the zeros from A and set up the column numbers.

      k = 0
      p = 0
      DO i = 1, prob%n
         DO j = 1, i
            k = k + 1
            IF ( prob%H%val( k ) /= ZERO ) THEN
               p = p + 1
               prob%H%val( p ) = prob%H%val( k )
               prob%H%col( p ) = j
            END IF
         END DO
      END DO

!     Update the type of matrix.

      CALL QPT_put_H( prob%H%type, 'SPARSE_BY_ROWS' )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_H_from_D_to_S

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_H_from_S_to_Di( prob, exitcode )

!     Transforms the lower triangle of the (diagonal) matrix H from
!     sparse-by-row format to diagonal format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            NOT_DIAGONAL: the matrix is not diagonal

!     Programming: Nick Gould, April 2005, based on Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i

!     Sort the entries, row by row, starting from the last.

      IF ( prob%H%ptr( 1 ) /= 1 ) THEN
        exitcode = NOT_DIAGONAL
        RETURN
      END IF
      DO i = 1, prob%n
         IF ( prob%H%ptr( i + 1 ) /= i + 1 .OR. prob%H%col( i ) /= i ) THEN
           exitcode = NOT_DIAGONAL
           RETURN
         END IF
      END DO

!     Update the type of matrix.

      CALL QPT_put_H( prob%H%type, 'DIAGONAL' )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_H_from_S_to_Di

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_H_from_Di_to_S( prob, exitcode )

!     Transforms the (diagonal) matrix H from diagonal storage to
!     sparse-by-row format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Nick Gould, April 2005, based on Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, iostat

!     Allocate the pointer to the beginning of each row.

      ALLOCATE( prob%H%ptr( prob%n + 1 ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Allocate the pointer for the column indices.

      ALLOCATE( prob%H%col( prob%n ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Set the row pointers and column numbers

      DO i = 1, prob%n
        prob%H%ptr( i ) = i
        prob%H%col( i ) = i
      END DO
      prob%H%ptr( prob%n + 1 ) = prob%n + 1

!     Update the type of matrix.

      CALL QPT_put_H( prob%H%type, 'SPARSE_BY_ROWS' )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_H_from_Di_to_S

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_H_from_C_to_S( prob, exitcode )

!     Transform the coordinate storage for the matrix H to sparse by row.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, September 2000

!===============================================================================

!     Local variables

      INTEGER  :: k, i, j, ii, nnz, iostat

      IF ( prob%n <= 0 ) RETURN

!     Allocate the pointers to the beginning of each row.

      ALLOCATE( prob%H%ptr( prob%n + 1 ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Count the number of nonzeros in each subdiagonal row.

      prob%H%ptr( :prob%n + 1 ) = 0
      DO k = 1, prob%H%ne
         IF ( prob%H%val( k ) /= ZERO ) THEN
            i = prob%H%row( k )
            j = prob%H%col( k )
            IF ( j <= i ) THEN
               prob%H%ptr( i ) = prob%H%ptr( i ) + 1
            ELSE
               prob%H%ptr( j ) = prob%H%ptr( j ) + 1
            END IF
         END IF
      END DO

!     Assign the pointers to the beginning of each row.

      ii = 1
      DO i = 1, prob%n + 1
         k = prob%H%ptr( i )
         prob%H%ptr( i ) = ii
         ii = ii + k
      END DO

!     Build the permutation of the elements of H in the vector that contains
!     its row numbers.

      nnz = prob%H%ne
      DO k = 1, prob%H%ne

!        Insert nonzero entries in their appropriate row.

         IF ( prob%H%val( k ) /= ZERO ) THEN
            i  = prob%H%row( k )
            j  = prob%H%col( k )
            IF ( j <= i ) THEN
               ii = prob%H%ptr( i )
               prob%H%row( k ) = ii
               prob%H%ptr( i ) = ii + 1
            ELSE
               ii = prob%H%ptr( j )
               prob%H%row( k ) = ii
               prob%H%ptr( j ) = ii + 1
            END IF

!        Insert zero entries at the (hidden) end of the array.

         ELSE
            prob%H%row( k ) = nnz
            nnz = nnz - 1
         END IF
      END DO

!     Rebuild the pointers to the beginning of each row.

      DO i = prob%n , 2, -1
         prob%H%ptr( i ) = prob%H%ptr( i - 1 )
      END DO
      prob%H%ptr( 1 ) = 1

!     Apply the permutation to the elements of H and their column numbers.

      CALL SORT_inplace_permute( prob%H%ne , prob%H%row,         &
                                 x = prob%H%val, ix =prob%H%col  )

!     Update the matrix type.

      CALL QPT_put_H( prob%H%type, 'SPARSE_BY_ROWS' )

!     Deallocate the space occupied by the (now useless) row indices.

      DEALLOCATE( prob%H%row )

!     Indicate successful exit

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_H_from_C_to_S

!===============================================================================
!===============================================================================

      SUBROUTINE QPT_H_from_S_to_C( prob, exitcode )

!     Transforms the matrix H from sparse-by-row format to coordinate format.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!            the problem whose matrix is considered.

      INTEGER, INTENT( INOUT ) :: exitcode

!            this function return one of the values
!            OK         : successful transformation,
!            MEMORY_FULL: the transformation could not be carried out because
!                         the necessary memory allocation failed.

!     Programming: Ph. Toint, June 2001.

!===============================================================================

!     Local variables

      INTEGER :: i, k, iostat

!     Allocate the pointers for the row numbers.

      ALLOCATE( prob%H%row( prob%H%ne ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         exitcode = MEMORY_FULL
         RETURN
      END IF

!     Assign the row numbers.

      DO i = 1, prob%n
         DO k = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            prob%H%row( k ) = i
         END DO
      END DO
      prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1
      CALL QPT_put_H( prob%H%type, 'COORDINATE' )

      DEALLOCATE( prob%H%ptr )

!     Indicate successful exit.

      exitcode = OK

      RETURN

      END SUBROUTINE QPT_H_from_S_to_C

!===============================================================================

   END MODULE GALAHAD_QPT_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*   END GALAHAD QPT  M O D U L E  *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
