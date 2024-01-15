   PROGRAM GALAHAD_TRS_EXAMPLE  !  GALAHAD 4.1 - 2022-12-16 AT 14:10 GMT.
#include "galahad_modules.h"
   USE GALAHAD_KINDS_precision
   USE GALAHAD_TRS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, h_ne = 4 ! problem dimensions
   INTEGER ( KIND = ip_ ) :: i, s
   REAL ( KIND = rp_ ), DIMENSION( n ) :: C, X
   TYPE ( SMT_type ) :: H, M
   TYPE ( TRS_data_type ) :: data
   TYPE ( TRS_control_type ) :: control
   TYPE ( TRS_inform_type ) :: inform
   LOGICAL, PARAMETER :: use_m = .FALSE.
   REAL ( KIND = rp_ ) :: f = 0.0_rp_           ! constant term, f
!  REAL ( KIND = rp_ ) :: radius = 0.001_rp_    ! trust-region radius (small)
   REAL ( KIND = rp_ ) :: radius = 1.0_rp_      ! trust-region radius (medium)
!  REAL ( KIND = rp_ ) :: radius = 100000.0_rp_ ! trust-region radius (large)
   IF ( use_m ) THEN
!    CALL SMT_put( M%type, 'DIAGONAL', s )      ! Specify diagonal for M
!    ALLOCATE( M%val( n ) )                     !
!    M%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)       ! M-norm, M
     CALL SMT_put( M%type, 'COORDINATE', s )    ! Specify co-ordinate for H
     ALLOCATE( M%val( h_ne ), M%row( h_ne ), M%col( h_ne ) )
     M%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 0.9_rp_ /) ! M-norm, M
     M%row = (/ 1, 2, 3, 2 /)                     ! NB lower triangle
     M%col = (/ 1, 2, 3, 1 /) ; M%ne = h_ne
   END IF
   CALL SMT_put( H%type, 'COORDINATE', s )    ! Specify co-ordinate for H
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /) ! Hessian, H
!  H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 1.0_rp_ /) ! Hessian, H
   H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   H%col = (/ 1, 2, 3, 1 /) ; H%ne = h_ne
   DO i = 1, 3
     IF ( i == 1 ) THEN             !  (normal case)
       WRITE( 6, "( ' Normal case:' )" )
       C = (/ 5.0_rp_, 0.0_rp_, 4.0_rp_ /)
     ELSE IF ( i == 2 ) THEN        !  (hard case)
       WRITE( 6, "( ' Hard case:' )" )
       C = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
     ELSE IF ( i == 3 ) THEN        !  (almost hard case)
       WRITE( 6, "( ' Almost hard case:' )" )
       C = (/ 0.0_rp_, 2.0_rp_, 0.0001_rp_ /)
     END IF
     CALL TRS_initialize( data, control )       ! Initialize control parameters
     control%print_level = 1
!    control%equality_problem = .TRUE.
     control%taylor_max_degree = 2
!    control%IR_control%itref_max = 2
!    control%initial_multiplier = 9.403124237432849_rp_
     IF ( use_m ) THEN                          ! solve problem
       CALL TRS_solve( n, radius, f, C, H, X, data, control, inform, M = M )
     ELSE
       CALL TRS_solve( n, radius, f, C, H, X, data, control, inform )
     END IF
     IF ( inform%status == 0 ) THEN !  Successful return
       WRITE( 6, "(1X,I0,' factorizations. Solution and Lagrange multiplier =',&
      &    2ES12.4 )" ) inform%factorizations, inform%obj, inform%multiplier
     ELSE  !  Error returns
       WRITE( 6, "( ' TRS_solve exit status = ', I0 ) " ) inform%status
     END IF
     CALL TRS_terminate( data, control, inform )  ! delete internal workspace
   END DO
   END PROGRAM GALAHAD_TRS_EXAMPLE
