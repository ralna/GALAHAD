! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LLST_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LLST_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, zero = 0.0_rp_
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 5000, n = 2 * m + 1
   INTEGER ( KIND = ip_ ) :: i, pass, problem, nn
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   REAL ( KIND = rp_ ), DIMENSION( m ) :: B
   REAL ( KIND = rp_ ) :: radius

   TYPE ( LLST_data_type ) :: data
   TYPE ( LLST_control_type ) :: control
   TYPE ( LLST_inform_type ) :: inform
   TYPE ( SMT_type ) :: A, S

   B = one                               ! The term b is a vector of ones
   A%m = m ; A%n = n ; A%ne = 3 * m      ! A^T = ( I : Diag(1:n) : e )
   CALL SMT_put( A%type, 'COORDINATE', i )
   ALLOCATE( A%row( 3 * m ), A%col( 3 * m ), A%val( 3 * m ) )
   DO i = 1, m
     A%row( i ) = i ; A%col( i ) = i ; A%val( i ) = one
     A%row( m + i ) = i ; A%col( m + i ) = m + i
     A%val( m + i ) = REAL( i, rp_ )
     A%row( 2 * m + i ) = i ; A%col( 2 * m + i ) = n
     A%val( 2 * m + i ) = one
   END DO
   S%m = n ; S%n = n ; S%ne = n    ! S = diag(1:n)**2
   CALL SMT_put( S%type, 'DIAGONAL', i )
   ALLOCATE( S%val( n ) )
   DO i = 1, n
     S%val( i ) = REAL( i * i, rp_ )
   END DO

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exit tests ====== ', / )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
   DO problem = 1, 2
     DO pass = 1, 5
       CALL LLST_initialize( data, control, inform )
       control%print_level = 10
!      control%itmax = 50
!      control%extra_vectors = 100
!      control%error = 23 ; control%out = 23 ; control%print_level = 10
       control%sbls_control%symmetric_linear_solver = "sytr  "
       control%sbls_control%definite_linear_solver = "sytr  "
       radius = one
       IF ( pass == 2 ) radius = 10.0_rp_
       IF ( pass == 3 ) radius = 0.0001_rp_
       IF ( pass == 4 ) THEN
         inform%status = 5
         radius = 0.0001_rp_
         control%prefix = '"LLST: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        radius = 10.0_rp_
       END IF
       IF ( pass == 5 ) THEN
!        if(problem==2)stop
         control%prefix = '"LLST: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        radius = 10.0_rp_
       END IF

       IF ( problem == 1 ) THEN
         CALL LLST_solve( m, n, radius, A, B, X, data, control, inform )
       ELSE
         CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
       END IF
       WRITE( 6, "( ' problem ', I1, ' pass = ', I1,                           &
      &  ' LLST_solve exit status = ', I6 )" ) problem, pass, inform%status
       CALL LLST_terminate( data, control, inform ) ! delete workspace
     END DO
   END DO

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exit tests ====== ', / )" )

! Initialize control parameters

!  DO pass = 1, 6
   DO pass = 1, 5
      radius = one
      CALL LLST_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      IF ( pass == 1 ) nn = 0
      IF ( pass == 2 ) radius = - one
      IF ( pass == 3 ) CALL SMT_put( A%type, 'UNCOORDINATE', i )
      IF ( pass == 4 ) CALL SMT_put( S%type, 'UNDIAGONAL', i )
!      IF ( pass == 1 ) control%equality_problem = .TRUE.
      IF ( pass == 5 ) THEN
        control%max_factorizations = 1
        radius = 100.0_rp_
      IF ( pass == 6 ) THEN
        DO i = 1, n
          S%val( i ) = - REAL( i, rp_ )
        END DO
      END IF
      END IF

!  Iteration to find the minimizer

      IF ( pass /= 1 ) THEN
        CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
      ELSE
        CALL LLST_solve( 0, nn, radius, A, B, X, data, control, inform )
      END IF
      IF ( pass == 3 ) CALL SMT_put( A%type, 'COORDINATE', i )
      IF ( pass == 4 ) CALL SMT_put( S%type, 'DIAGONAL', i )
      WRITE( 6, "( ' pass ', I3, ' LLST_solve exit status = ', I6 )" )         &
             pass, inform%status
      CALL LLST_terminate( data, control, inform ) !  delete internal workspace
   END DO
   CLOSE( unit = 23 )
   WRITE( 6, "( /, ' tests completed' )" )

   END PROGRAM GALAHAD_LLST_test_program
