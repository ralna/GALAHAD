   PROGRAM GALAHAD_TRS_LARGE_EXAMPLE  !  GALAHAD 4.1 - 2022-12-16 AT 14:10 GMT.
#include "galahad_modules.h"
   USE GALAHAD_KINDS_precision
   USE GALAHAD_TRS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
   INTEGER ( KIND = ip_ ) :: i, s, l, n2, logn, n, h_ne
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C, X
   TYPE ( SMT_type ) :: H
   TYPE ( TRS_data_type ) :: data
   TYPE ( TRS_control_type ) :: control
   TYPE ( TRS_inform_type ) :: inform
   REAL ( KIND = rp_ ) :: f = 0.0_rp_           ! constant term, f
!  REAL ( KIND = rp_ ) :: radius = 0.001_rp_    ! trust-region radius (small)
   REAL ( KIND = rp_ ) :: radius = 1.0_rp_      ! trust-region radius (medium)
!  REAL ( KIND = rp_ ) :: radius = 100000.0_rp_ ! trust-region radius (large)
!$ INTEGER:: OMP_GET_MAX_THREADS
   CALL SMT_put( H%type, 'COORDINATE', s )    ! Specify co-ordinate for H
!$ WRITE(6, "( ' max threads = ', I0 )" ) OMP_GET_MAX_THREADS( )

!  DO logn = 6, 14
!  DO logn = 14, 14
!  DO logn = 13, 13
!  DO logn = 8, 8
   DO logn = 11, 11
!  DO logn = 17, 17
     IF ( MOD( logn, 2 ) == 1 ) THEN
       n = 3.1622772 * 10 ** ( logn / 2 )
       IF ( 2 * ( n / 2 ) /= n ) n = n + 1
     ELSE
       n = 10 ** ( logn / 2 )
     END IF
!    n = 10 ** logn
!    n = 2 ** logn
     h_ne = 4 * n - 6
     WRITE( 6, "( ' n, h_ne ', I0, 1X, I0 )" ) n, h_ne
     ALLOCATE( C( n ), X( n ) )
     ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) ) ; H%ne = h_ne
     n2 = n / 2
     DO l = 1, n
       C( l ) = - 0.5_rp_
       H%row( l ) = l ;  H%col( l ) = l
       IF ( l /= 1 .AND. l /= n2 .AND. l /= n ) THEN
         H%val( l ) = 6.0_rp_
       ELSE
         H%val( l ) = 2.0_rp_ * n + 10.0_rp_
       END IF
     END DO

     l = n
     DO i = 2, n
       l = l + 1
       H%row( l ) = i ; H%col( l ) = 1
       IF ( i /= n2 .AND. i /= n ) THEN
         H%val( l ) = 2.0_rp_
       ELSE
         H%val( l ) = 4.0_rp_
       END IF
     END DO
     DO i = 2, n - 1
       l = l + 1
       H%row( l ) = n ; H%col( l ) = i
       IF ( i /= n2 ) THEN
         H%val( l ) = 2.0_rp_
       ELSE
         H%val( l ) = 4.0_rp_
       END IF
     END DO
     DO i = 2, n2 - 1
       l = l + 1
       H%row( l ) = n2 ; H%col( l ) = i ; H%val( l ) = 2.0_rp_
     END DO
     DO i = n2 + 1, n - 1
       l = l + 1
       H%row( l ) = i ; H%col( l ) = n2 ; H%val( l ) = 2.0_rp_
     END DO

     CALL TRS_initialize( data, control )       ! Initialize control parameters
     control%print_level = 3
!    control%equality_problem = .TRUE.
!    control%taylor_max_degree = 3
!    control%IR_control%itref_max = 2
     control%initial_multiplier = 0.0_rp_
     control%use_initial_multiplier = .TRUE.
     control%SLS_control%ordering = 2
!    control%problem = 99
     control%definite_linear_solver = 'ma87'
     control%SLS_control%initial_pool_size = 6400000

     WRITE( 6, * ) ' ** solver ', control%definite_linear_solver
     CALL TRS_solve( n, radius, f, C, H, X, data, control, inform )
     IF ( inform%status == 0 ) THEN !  Successful return
       WRITE( 6, "( ' Solution and Lagrange multiplier =', 2ES12.4 )" )        &
         inform%obj, inform%multiplier
       WRITE( 6, "( 1X, I0,' factorizations, time = ', F9.2 )" )               &
         inform%factorizations, inform%time%total
     ELSE  !  Error returns
       WRITE( 6, "( ' TRS_solve exit status = ', I0 ) " ) inform%status
     END IF
     WRITE( 6, * ) ' factors ', inform%SLS_inform%entries_in_factors
     CALL TRS_terminate( data, control, inform )  ! delete internal workspace
     DEALLOCATE( X, C, H%row, H%col, H%val )

   END DO
   END PROGRAM GALAHAD_TRS_LARGE_EXAMPLE
