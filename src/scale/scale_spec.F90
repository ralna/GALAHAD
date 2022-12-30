! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SCALE_EXAMPLE
   USE GALAHAD_KINDS
   USE GALAHAD_SCALE_precision
   USE GALAHAD_SMT_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SCALE_trans_type ) :: trans
   TYPE ( SCALE_data_type ) :: data
   TYPE ( SCALE_control_type ) :: control        
   TYPE ( SCALE_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: s, scale
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4 
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%n = n ; p%m = m ; p%f = 1.0_rp_               ! dimensions & obj constant
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)           ! objective gradient
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)                  ! constraint lower bound
   p%C_u = (/ 2.0_rp_, 2.0_rp_ /)                  ! constraint upper bound
   p%X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)        ! variable upper bound
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp    ! typical values for x, y & z
   p%C = 0.0_rp_                                   ! c = A * x
!  sparse co-ordinate storage format
   CALL SMT_put( p%H%type, 'COORDINATE', s )      ! specify co-ordinate 
   CALL SMT_put( p%A%type, 'COORDINATE', s )      ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ /) ! Hessian H
   p%H%row = (/ 1, 2, 2, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 1, 3 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete - compute and apply scale factors
   CALL SCALE_initialize( data, control, inform ) ! Initialize controls
   control%infinity = infinity                    ! Set infinity
   scale = 7                                      ! Sinkhorn-Knopp scaling
   CALL SCALE_get( p, scale, trans, data, control, inform ) ! Get scalings
   IF ( inform%status == 0 ) THEN                 !  Successful return
     WRITE( 6, "( ' variable scalings : ', /, ( 5ES12.4 ) )" ) trans%X_scale
     WRITE( 6, "( ' constraint scalings : ', /, ( 5ES12.4 ) )" ) trans%C_scale
   ELSE                                           !  Error returns
     WRITE( 6, "( ' SCALE_get exit status = ', I6 ) " ) inform%status
   END IF
   CALL SCALE_apply( p, trans, data, control, inform )
   IF ( inform%status == 0 ) THEN                 !  Successful return
     WRITE( 6, "( ' scaled A : ', /, ( 5ES12.4 ) )" ) p%A%val
   ELSE                                           !  Error returns
     WRITE( 6, "( ' SCALE_get exit status = ', I6 ) " ) inform%status
   END IF
   CALL SCALE_terminate( data, control, inform, trans )  !  delete workspace
   END PROGRAM GALAHAD_SCALE_EXAMPLE
