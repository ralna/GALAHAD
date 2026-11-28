! THIS VERSION: GALAHAD 5.6 - 2026-11-28 AT 14:20 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LLST_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LLST_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, zero = 0.0_rp_
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 10, n = 2 * m + 1
!  INTEGER ( KIND = ip_ ), PARAMETER :: m = 100, n = 2 * m + 1
   INTEGER ( KIND = ip_ ) :: i, l, data_storage_type, pass, with_s, nn, smt_stat
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   REAL ( KIND = rp_ ), DIMENSION( m ) :: B
   REAL ( KIND = rp_ ) :: radius
   CHARACTER ( len = 1 ) :: st

   TYPE ( LLST_data_type ) :: data
   TYPE ( LLST_control_type ) :: control
   TYPE ( LLST_inform_type ) :: inform
   TYPE ( SMT_type ) :: A, S, A_dense, S_dense
   TYPE ( SMT_type ) :: A4, S2
   REAL ( KIND = rp_ ), DIMENSION( 2 ) :: X2, X2_star
   REAL ( KIND = rp_ ), DIMENSION( 4 ) :: B4
   REAL ( KIND = rp_ ) :: lambda_star, tol

   A%m = m ; A%n = n ; A%ne = 3 * m    ! A = ( I : Diag(1:n) : e )
   ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ), A%ptr( m + 1 ) )
   l = 1
   DO i = 1, m
     A%ptr( i ) = l
     A%row( l ) = i ; A%col( l ) = i ; A%val( l ) = one
     l = l + 1
     A%row( l ) = i ; A%col( l ) = m + i ;  A%val( l ) = REAL( i, rp_ )
     l = l + 1
     A%row( l ) = i ; A%col( l ) = n ;  A%val( l ) = one
     l = l + 1
   END DO
   A%ptr( m + 1 ) = l
   ALLOCATE( A_dense%val( m * n ) )
   A_dense%val = zero
   l = 0
   DO i = 1, m
     A_dense%val( l + i ) = one
     A_dense%val( l + m + i ) = REAL( i, rp_ )
     A_dense%val( l + n ) = one
     l = l + n
   END DO
   S%m = n ; S%n = n ; S%ne = n        ! S = diag(1:n)**2
   ALLOCATE( S%row( S%ne ), S%col( S%ne ), S%val( S%ne ), S%ptr( n + 1 ) )
   DO i = 1, n
     S%row( i ) = i ; S%col( i ) = i ; S%ptr( i ) = i
     S%val( i ) = REAL( i * i, rp_ )
   END DO
   S%ptr( n + 1 ) = n + 1
   ALLOCATE( S_dense%val( n * ( n + 1 ) / 2 ) )
   S_dense%val = zero
   l = 0
   DO i = 1, n
     S_dense%val( l + i ) = REAL( i * i, rp_ )
     l = l + i
   END DO
   B = one   !  b is a vector of ones

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exit tests ====== ', / )" )

   CALL SMT_put( A%type, 'COORDINATE', i )
   CALL SMT_put( S%type, 'DIAGONAL', i )

! Initialize control parameters

!  DO pass = 1, 6
   DO pass = 1, 5
     radius = one
     CALL LLST_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
     IF ( pass == 1 ) nn = 0
     IF ( pass == 2 ) radius = - one
     IF ( pass == 3 ) CALL SMT_put( A%type, 'UNCOORDINATE', i )
     IF ( pass == 4 ) CALL SMT_put( S%type, 'UNDIAGONAL', i )
!     IF ( pass == 1 ) control%equality_problem = .TRUE.
     IF ( pass == 5 ) THEN
       control%max_factorizations = 1
       radius = 0.01_rp_
     END IF
     IF ( pass == 6 ) THEN
       DO i = 1, n
         S%val( i ) = - REAL( i, rp_ )
       END DO
     END IF

!  Iteration to find the minimizer

     IF ( pass == 4 .OR. pass == 6 ) THEN
       CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
     ELSE IF ( pass /= 1 ) THEN
       CALL LLST_solve( m, n, radius, A, B, X, data, control, inform )
     ELSE
       CALL LLST_solve( 0_ip_, nn, radius, A, B, X, data, control, inform )
     END IF
     IF ( pass == 2 ) radius = one
     IF ( pass == 3 ) CALL SMT_put( A%type, 'COORDINATE', i )
     IF ( pass == 4 ) CALL SMT_put( S%type, 'DIAGONAL', i )
     IF ( pass == 5 ) control%max_factorizations = - 1
     IF ( pass == 6 ) THEN
       DO i = 1, n
         S%val( i ) = REAL( i * i, rp_ )
         S%val( i ) = one
       END DO
     END IF
     WRITE( 6, "( ' pass ', I3, ' LLST_solve exit status = ', I6 )" )          &
            pass, inform%status
     CALL LLST_terminate( data, control, inform ) !  delete internal workspace
   END DO
   DEALLOCATE( A%type, S%type )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

!  DO data_storage_type = 2, 2
   DO data_storage_type = 1, 4
     CALL LLST_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%print_level = 1
!    control%sls_control%print_level_solver = 3
!    control%sls_control%print_level = 3
     radius = one
     IF ( data_storage_type == 1 ) THEN          ! sparse co-ordinate storage
       st = 'C'
       CALL SMT_put( A%type, 'COORDINATE', smt_stat )
       CALL SMT_put( S%type, 'COORDINATE', smt_stat )
     ELSE IF ( data_storage_type == 2 ) THEN     ! sparse row-wise storage
       st = 'R'
       CALL SMT_put( A%type, 'SPARSE_BY_ROWS', smt_stat )
       CALL SMT_put( S%type, 'SPARSE_BY_ROWS', smt_stat )
     ELSE IF ( data_storage_type == 3 ) THEN      ! dense storage
       st = 'D'
       CALL SMT_put( A_dense%type, 'DENSE', smt_stat )
       CALL SMT_put( S_dense%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == 4 ) THEN      ! co-ordinate A, diagonal S
       st = 'I'
       CALL SMT_put( A%type, 'COORDINATE', smt_stat )
       CALL SMT_put( S%type, 'DIAGONAL', smt_stat )
     END IF
     IF ( data_storage_type == 3 ) THEN
       CALL LLST_solve( m, n, radius, A_dense, B, X, data, control, inform,    &
                        S = S_dense )
       DEALLOCATE( A_dense%type, S_dense%type )
     ELSE
       CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
       DEALLOCATE( A%type, S%type )
     END IF
     WRITE( 6, "( ' storage type ', A1, ' LLST_solve exit status = ', I0,      &
    & ' ||r|| = ', F5.2 )" )  st, inform%status, inform%r_norm
   END DO

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exit tests ====== ', / )" )

   CALL SMT_put( A%type, 'COORDINATE', i )
   CALL SMT_put( S%type, 'DIAGONAL', i )

! Initialize control parameters

   DO with_s = 0, 1
     DO pass = 1, 5
       CALL LLST_initialize( data, control, inform )
       CALL WHICH_sls( control )
       control%print_level = 1
!      control%itmax = 50
!      control%extra_vectors = 100
       control%error = 23 ; control%out = 23 ; control%print_level = 10
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
!        if(with_s==2)stop
         control%prefix = '"LLST: "     '
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
!        radius = 10.0_rp_
       END IF

       IF ( with_s == 0 ) THEN
         CALL LLST_solve( m, n, radius, A, B, X, data, control, inform )
       ELSE
         CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
       END IF
       WRITE( 6, "( ' with S = ', I1, ' pass = ', I1,                          &
      &  ' LLST_solve exit status = ', I0, ' ||r|| = ', F5.2 )" )              &
         with_s, pass, inform%status, inform%r_norm
       CALL LLST_terminate( data, control, inform ) ! delete workspace
     END DO
   END DO

!  ==========================================================================
!  the Schur-complement factorization of K(lambda) with lambda = 0, for
!  A = ( I ; diag( 1, 2 ) ). Passes 1 and 2: b = ( 0.9, 0.8, 0.9, 2 ) and the
!  radii 1.5 and 10, for which the solution is the least-squares solution
!  x = ( 0.9, 0.96 ), with lambda = 0. Pass 3: b = ( 0.9, 1.2, 0.9, 1.2 ),
!  radius 1 and S = diag( 1, 4 ) in co-ordinate storage, for which
!  x = ( 0.6, 0.4 ) and lambda = 1
!  ==========================================================================

   WRITE( 6, "( /, ' ==== Schur complement with lambda = 0 ==== ', / )" )
   CALL SMT_put( A4%type, 'COORDINATE', i )
   A4%m = 4 ; A4%n = 2 ; A4%ne = 4
   ALLOCATE( A4%row( 4 ), A4%col( 4 ), A4%val( 4 ) )
   A4%row = (/ 1, 2, 3, 4 /) ; A4%col = (/ 1, 2, 1, 2 /)
   A4%val = (/ one, one, one, 2.0_rp_ /)
   CALL SMT_put( S2%type, 'COORDINATE', i )
   S2%m = 2 ; S2%n = 2 ; S2%ne = 2
   ALLOCATE( S2%row( 2 ), S2%col( 2 ), S2%val( 2 ) )
   S2%row = (/ 1, 2 /) ; S2%col = (/ 1, 2 /) ; S2%val = (/ one, 4.0_rp_ /)
   tol = SQRT( EPSILON( one ) )
   DO pass = 1, 3
     CALL LLST_initialize( data, control, inform )
     control%definite_linear_solver = 'potr'
     control%SBLS_control%symmetric_linear_solver = 'sytr'
     control%SBLS_control%definite_linear_solver = 'potr'
     control%SBLS_control%factorization = 1
     IF ( pass == 1 .OR. pass == 2 ) THEN
       IF ( pass == 1 ) THEN
         radius = 1.5_rp_
       ELSE
         radius = 10.0_rp_
       END IF
       B4 = (/ 0.9_rp_, 0.8_rp_, 0.9_rp_, 2.0_rp_ /)
       X2_star = (/ 0.9_rp_, 0.96_rp_ /) ; lambda_star = zero
       CALL LLST_solve( 4_ip_, 2_ip_, radius, A4, B4, X2, data, control,      &
                        inform )
     ELSE
       radius = one
       B4 = (/ 0.9_rp_, 1.2_rp_, 0.9_rp_, 1.2_rp_ /)
       X2_star = (/ 0.6_rp_, 0.4_rp_ /) ; lambda_star = one
       CALL LLST_solve( 4_ip_, 2_ip_, radius, A4, B4, X2, data, control,      &
                        inform, S = S2 )
     END IF
     WRITE( 6, "( ' pass ', I1, ' LLST_solve exit status = ', I0 )" )         &
       pass, inform%status
     IF ( inform%status /= 0 .OR. MAXVAL( ABS( X2 - X2_star ) ) > tol .OR.    &
          ABS( inform%multiplier - lambda_star ) > tol ) THEN
       WRITE( 6, "( ' x = ', 2ES12.4, ', lambda = ', ES12.4, ' FAILED' )" )   &
         X2, inform%multiplier
       STOP 1
     END IF
     CALL LLST_terminate( data, control, inform ) ! delete workspace
   END DO
   DEALLOCATE( A4%row, A4%col, A4%val, A4%type )
   DEALLOCATE( S2%row, S2%col, S2%val, S2%type )

   DEALLOCATE( A%row, A%col, A%val, A%ptr, A%type, A_dense%val )
   DEALLOCATE( S%row, S%col, S%val, S%ptr, S%type, S_dense%val )
   CLOSE( unit = 23 )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( LLST_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%definite_linear_solver = definite_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_LLST_test_program
