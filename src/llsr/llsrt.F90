! THIS VERSION: GALAHAD 5.1 - 2024-11-23 AT 15:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LLSR_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LLSR_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, zero = 0.0_rp_
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 10, n = 2 * m + 1
!  INTEGER ( KIND = ip_ ), PARAMETER :: m = 100, n = 2 * m + 1
   INTEGER ( KIND = ip_ ) :: i, l, p, pass
   INTEGER ( KIND = ip_ ) :: data_storage_type, with_s, smt_stat
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   REAL ( KIND = rp_ ), DIMENSION( m ) :: B
   REAL ( KIND = rp_ ) :: power, weight
   CHARACTER ( len = 1 ) :: st

   TYPE ( LLSR_data_type ) :: data
   TYPE ( LLSR_control_type ) :: control
   TYPE ( LLSR_inform_type ) :: inform
   TYPE ( SMT_type ) :: A, S, A_dense, S_dense

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
     power = 3.0_rp_ ; weight = one
     CALL LLSR_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
     IF ( pass == 2 ) weight = - one
     IF ( pass == 3 ) CALL SMT_put( A%type, 'UNCOORDINATE', i )
     IF ( pass == 4 ) CALL SMT_put( S%type, 'UNDIAGONAL', i )
     IF ( pass == 5 ) THEN
       control%max_factorizations = 1
       weight = 100.0_rp_
     END IF
     IF ( pass == 6 ) THEN
       DO i = 1, n
         S%val( i ) = - REAL( i, rp_ )
       END DO
     END IF

!  Iteration to find the minimizer

     IF ( pass == 4 .OR. pass == 6 ) THEN
       CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform,   &
                        S = S )
     ELSE IF ( pass /= 1 ) THEN
       CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform )
     ELSE
       CALL LLSR_solve( 0_ip_, 0_ip_, power, weight, A, B, X, data, control,   &
                        inform )
     END IF
     IF ( pass == 2 ) weight = one
     IF ( pass == 3 ) CALL SMT_put( A%type, 'COORDINATE', i )
     IF ( pass == 4 ) CALL SMT_put( S%type, 'DIAGONAL', i )
     IF ( pass == 5 ) control%max_factorizations = - 1
     IF ( pass == 6 ) THEN
       DO i = 1, n
         S%val( i ) = REAL( i * i, rp_ )
         S%val( i ) = one
       END DO
     END IF
     WRITE( 6, "( ' pass ', I3, ' LLSR_solve exit status = ', I6 )" )          &
            pass, inform%status
     CALL LLSR_terminate( data, control, inform ) !  delete internal workspace
   END DO
   DEALLOCATE( A%type, S%type )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

!  DO data_storage_type = 2, 2
   DO data_storage_type = 1, 4
     CALL LLSR_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%print_level = 1
!    control%sls_control%print_level_solver = 3
!    control%sls_control%print_level = 3
     power = 3.0_rp_ ; weight = one
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
!      CALL LLSR_solve( m, n, power, weight, A_dense, B, X, data, control,     &
!                       inform )
       CALL LLSR_solve( m, n, power, weight, A_dense, B, X, data, control,     &
                        inform, S = S_dense )
       DEALLOCATE( A_dense%type, S_dense%type )
     ELSE
!      CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform )
       CALL LLSR_solve( m, n, power, weight, A, B, X, data, control, inform,   &
                        S = S )
       DEALLOCATE( A%type, S%type )
     END IF
     WRITE( 6, "( ' storage type ', A1, ' LLSR_solve exit status = ', I0,      &
    & ' ||r|| = ', F5.2 )" )  st, inform%status, inform%r_norm
   END DO

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exit tests ====== ', / )" )

   CALL SMT_put( A%type, 'COORDINATE', i )
   CALL SMT_put( S%type, 'DIAGONAL', i )

! Initialize control parameters

   DO p = 2, 4
!  DO p = 3, 4
     IF ( p == 2 ) THEN
       power = 2.0_rp_
     ELSE IF ( p == 3 ) THEN
       power = 3.0_rp_
     ELSE
!      power = 4.0_rp_
       power = 3.5_rp_
     END IF
     DO with_s = 0, 1
       DO pass = 1, 4
         CALL LLSR_initialize( data, control, inform )
         CALL WHICH_sls( control )
!        control%itmax = 50
!        control%extra_vectors = 100
         control%error = 23 ; control%out = 23 ; control%print_level = 10
!        control%print_level = 100
         weight = one
         IF ( pass == 2 ) weight = 0.1_rp_
         IF ( pass == 3 ) weight = 1000.0_rp_
         IF ( pass == 4 ) THEN
           inform%status = 5
           weight = 1000.0_rp_
           control%prefix = '"LLSR: "     '
!          control%error = 6 ; control%out = 6 ; control%print_level = 1
!          weight = 0.1_rp_
         END IF

         IF ( with_s == 0 ) THEN
           CALL LLSR_solve( m, n, power, weight, A, B, X, data, control,       &
                            inform )
         ELSE
           CALL LLSR_solve( m, n, power, weight, A, B, X, data, control,       &
                            inform, S = S )
         END IF
         WRITE( 6, "( ' with power = ', I1, ', S = ', I1, ' & pass = ', I1,    &
        &  ', LLSR_solve exit status = ', I0, ', ||r|| = ', F5.2 )" )          &
           p, with_s, pass, inform%status, inform%r_norm
         CALL LLSR_terminate( data, control, inform ) ! delete workspace
       END DO
     END DO
   END DO
   DEALLOCATE( A%row, A%col, A%val, A%ptr, A%type, A_dense%val )
   DEALLOCATE( S%row, S%col, S%val, S%ptr, S%type, S_dense%val )
   CLOSE( unit = 23 )

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( LLSR_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%definite_linear_solver = definite_linear_solver
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_LLSR_test_program
