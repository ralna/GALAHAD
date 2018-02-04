! THIS VERSION: GALAHAD 2.5 - 16/07/2011 AT 10:00 GMT.
   PROGRAM GALAHAD_FDH_EXAMPLE
   USE GALAHAD_FDH_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( FDH_data_type ) :: data
   TYPE ( FDH_control_type ) :: control        
   TYPE ( FDH_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 5, nz = 9
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   INTEGER :: i, status
   INTEGER :: DIAG( n ), ROW( nz )
   REAL ( KIND = wp ) ::  X1( n ), X2( n ), STEPSIZE( n ), G( n )
   REAL ( KIND = wp ) ::  H( nz )
   TYPE ( NLPT_userdata_type ) :: userdata
   INTERFACE
     SUBROUTINE GRAD( status, X, userdata, G )   
     USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE GRAD
   END INTERFACE
! start problem data
   ROW = (/ 1, 4, 2, 3, 3, 4, 4, 5, 5 /)      ! Record the sparsity pattern of
   DIAG = (/ 1, 3, 5, 7, 9 /)                 ! the Hessian (lower triangle)
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space for the parameter
   userdata%real( 1 ) = p                     ! Record the parameter, p
   STEPSIZE  = (/ 0.000001_wp, 0.000001_wp, 0.000001_wp,                       &
                  0.000001_wp, 0.000001_wp /) ! Set difference stepsizes
! estimate the Hessian at X1 by internal evaluation 
   X1 = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   CALL FDH_initialize( data, control, inform )
   CALL FDH_analyse( n, nz, ROW, DIAG, data, control, inform )
   IF ( inform%status /= 0 ) THEN             ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from FDH_analyse' )" ) &
       inform%status ; STOP
   END IF
   CALL GRAD( status, X1( : n ), userdata, G( : n ) )
   CALL FDH_estimate( n, nz, ROW, DIAG, X1, G, STEPSIZE, H,                    &
                      data, control, inform, userdata, eval_G = GRAD )
   IF ( inform%status == 0 ) THEN              ! Success
     WRITE( 6, "( ' At 1st point, nonzeros in Hessian matrix are ', /,         &
    &             ( 5ES12.4 ) )" ) ( H( i ), i = 1, nz )
   ELSE                                        ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from FDH_estimate' )" )&
       inform%status ; STOP
   END IF
! estimate the Hessian at X2 by reverse communication
   X2 = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)
   CALL GRAD( status, X2( : n ), userdata, G( : n ) )
10 CONTINUE  
   CALL FDH_estimate( n, nz, ROW, DIAG, X2, G, STEPSIZE, H,                    &
                      data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN              ! Success
     WRITE( 6, "( /, ' At 2nd point, nonzeros in Hessian matrix are ', /,      &
    &             ( 5ES12.4 ) )" ) ( H( i ), i = 1, nz )
   ELSE IF ( inform%status > 0 ) THEN          ! Reverse communication required
     CALL GRAD( data%eval_status, data%X( : n ), userdata, data%G( : n ) )
     GO TO 10
   ELSE                                        ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from FDH_estimate' )" )&
       inform%status ; STOP
   END IF
   CALL FDH_terminate( data, control, inform ) ! Delete internal workspace
   END PROGRAM GALAHAD_FDH_EXAMPLE
! internal subroutine to evaluate the gradient of the objective
   SUBROUTINE GRAD( status, X, userdata, G )   
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   G( 1 ) = 3.0_wp * ( X( 1 ) + userdata%real( 1 ) ) ** 2 + X( 4 )
   G( 2 ) = 3.0_wp * X( 2 ) ** 2 + X( 3 )
   G( 3 ) = 3.0_wp * X( 3 ) ** 2 + X( 2 ) + X( 4 )
   G( 4 ) = 3.0_wp * X( 4 ) ** 2 + X( 1 ) + X( 3 ) + X( 5 )
   G( 5 ) = 3.0_wp * X( 5 ) ** 2 + X( 4 )
   status = 0
   END SUBROUTINE GRAD
