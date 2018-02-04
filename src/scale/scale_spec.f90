! THIS VERSION: GALAHAD 2.4 - 17/01/2011 AT 15:30 GMT.
   PROGRAM GALAHAD_SCALE_EXAMPLE
   USE GALAHAD_SCALE_double                       ! double precision version
   USE GALAHAD_SMT_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )      ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SCALE_trans_type ) :: trans
   TYPE ( SCALE_data_type ) :: data
   TYPE ( SCALE_control_type ) :: control        
   TYPE ( SCALE_inform_type ) :: inform
   INTEGER :: s, scale
   INTEGER, PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4 
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%n = n ; p%m = m ; p%f = 1.0_wp               ! dimensions & obj constant
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)             ! objective gradient
   p%C_l = (/ 1.0_wp, 2.0_wp /)                   ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)                   ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)         ! variable upper bound
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp     ! typical values for x, y & z
   p%C = 0.0_wp                                   ! c = A * x
!  sparse co-ordinate storage format
   CALL SMT_put( p%H%type, 'COORDINATE', s )      ! specify co-ordinate 
   CALL SMT_put( p%A%type, 'COORDINATE', s )      ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_wp, 2.0_wp, 1.0_wp, 3.0_wp /) ! Hessian H
   p%H%row = (/ 1, 2, 2, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 1, 3 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
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
