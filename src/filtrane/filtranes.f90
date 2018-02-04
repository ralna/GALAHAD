! THIS VERSION: GALAHAD 2.1 - 13/02/2008 AT 09:40 GMT.
PROGRAM GALAHAD_FILTRANE_EXAMPLE
  USE GALAHAD_NLPT_double      ! the problem type
  USE GALAHAD_FILTRANE_double  ! the FILTRANE solver
  USE GALAHAD_SYMBOLS
  IMPLICIT NONE
  INTEGER, PARAMETER                :: wp = KIND( 1.0D+0 )
  INTEGER,           PARAMETER      :: ispec = 55      ! SPECfile device number
  INTEGER,           PARAMETER      :: iout = 6        ! stdout and stderr
  REAL( KIND = wp ), PARAMETER      :: INFINITY = (10.0_wp)**19
  TYPE( NLPT_problem_type     )     :: problem 
  TYPE( FILTRANE_control_type )     :: FILTRANE_control
  TYPE( FILTRANE_inform_type  )     :: FILTRANE_inform
  TYPE( FILTRANE_data_type    )     :: FILTRANE_data
  INTEGER                           :: J_size
  REAL( KIND = wp ), DIMENSION( 3 ) :: H1
! Set the problem up.
  problem%n        = 2
  ALLOCATE( problem%x( problem%n )  , problem%x_status( problem%n ),           &
            problem%x_l( problem%n ), problem%x_u( problem%n ),                &
            problem%g( problem%n )  , problem%z( problem%n )  )
  problem%m        = 2
  ALLOCATE( problem%equation( problem%m ),                                     &
            problem%c( problem%m ) , problem%c_l( problem%m ),                 &
            problem%c_u( problem%m), problem%y( problem%m ) )
  problem%J_ne     = 4
  J_size           = problem%J_ne + problem%n 
  ALLOCATE( problem%J_val( J_size), problem%J_row( J_size ),                   &
            problem%J_col( J_size ) )
  problem%J_type   = GALAHAD_COORDINATE
  problem%infinity = INFINITY
  problem%x        = (/  1.0D0,  1.0D0 /)
  problem%x_l      = (/ -2.0D0, -2.0D0 /)
  problem%x_u      = (/  2.0D0,  2.0D0 /)
  problem%c_l      = (/  0.0D0,  0.0D0 /)
  problem%c_u      = (/  0.0D0,  0.0D0 /)
  problem%equation = (/ .TRUE., .TRUE. /)
           
! Initialize FILTRANE.
  CALL FILTRANE_initialize( FILTRANE_control, FILTRANE_inform, FILTRANE_data )
! Read the FILTRANE spec file (not necessary in this example, as the default
! settings are mostly suitable).
!  OPEN( ispec, file = 'FILTRANE.SPC', form = 'FORMATTED', status = 'OLD' )
!  CALL FILTRANE_read_specfile( ispec, FILTRANE_control, FILTRANE_inform )
!  CLOSE( ispec )
! Nevertheless... ask for some output:
  FILTRANE_control%print_level = GALAHAD_TRACE
! Now apply the solver in the reverse communication loop.
  DO 
     CALL FILTRANE_solve( problem, FILTRANE_control, FILTRANE_inform,          &
                          FILTRANE_data )
     SELECT CASE ( FILTRANE_inform%status )
     CASE ( 1, 2 ) ! constraints values and Jacobian
        problem%c( 1 ) = 3.0D0 * problem%x( 1 ) ** 2 +                         &
                         2.0D0 * problem%x( 2 ) ** 3 +                         &
                         problem%x( 1 ) * problem%x( 2 )
        problem%c( 2 ) = problem%x( 1 ) + problem%x( 2 )
        problem%J_val( 1 ) = 6.0D0 * problem%x( 1 ) + problem%x( 2 ) 
        problem%J_val( 2 ) = 1.0D0
        problem%J_val( 3 ) = 6.0D0 * problem%x( 2 ) ** 2 + problem%x( 1 )
        problem%J_val( 4 ) = 1.0D0
        problem%J_row( 1 : 4 ) = (/ 1, 2, 1, 2 /)
        problem%J_col( 1 : 4 ) = (/ 1, 1, 2, 2 /)
     CASE ( 3 : 5 ) ! constraints values only
        problem%c( 1 ) = 3.0D0 * problem%x( 1 ) ** 2 +                         &
                         2.0D0 * problem%x( 2 ) ** 3 +                         &
                         problem%x( 1 ) * problem%x( 2 )
        problem%c( 2 ) = problem%x( 1 ) + problem%x( 2 )
     CASE ( 6 ) ! Jacobian only
        problem%J_val( 1 ) = 6.0D0 * problem%x( 1 )  + problem%x( 2 ) 
        problem%J_val( 2 ) = 1.0D0
        problem%J_val( 3 ) = 6.0D0 * problem%x( 2 ) ** 2 + problem%x( 1 )
        problem%J_val( 4 ) = 1.0D0
     CASE ( 15, 16 ) ! product times the Hessian of the Lagrangian
!       Note that H2, the Hessian of C2 is identically zero, since this 
!       constraint is linear. Hence the terms in y(2)*H2 disappear.
        IF ( FILTRANE_data%RC_newx ) THEN
           H1( 1 ) = 6.0D0
           H1( 2 ) = 1.0D0
           H1( 3 ) = 12.0D0 * problem%x( 2 )
        END IF
        FILTRANE_data%RC_Mv( 1 ) =                                             &
                        problem%y( 1 ) * H1( 1 ) * FILTRANE_data%RC_v( 1 ) +   &
                        problem%y( 1 ) * H1( 2 ) * FILTRANE_data%RC_v( 2 ) 
        FILTRANE_data%RC_Mv( 2 ) =                                             &
                        problem%y( 1 ) * H1( 2 ) * FILTRANE_data%RC_v( 1 ) +   &
                        problem%y( 1 ) * H1( 3 ) * FILTRANE_data%RC_v( 2 ) 
     CASE DEFAULT
        EXIT
     END SELECT
  END DO ! end of the reverse communication loop
! Terminate FILTRANE.
  FILTRANE_control%print_level = GALAHAD_SILENT
  CALL FILTRANE_terminate( FILTRANE_control, FILTRANE_inform, FILTRANE_data )
! Output results.
  WRITE( iout, 1000 )
  WRITE( iout, 1001 ) 
  WRITE( iout, 1000 )
  WRITE( iout, 1002 ) problem%x( 1 )
  WRITE( iout, 1003 ) problem%x( 2 )
  WRITE( iout, 1000 ) 
  WRITE( iout, 1004 ) problem%c( 1 )
  WRITE( iout, 1005 ) problem%c( 2 )
  WRITE( iout, 1000 )
  WRITE( iout, 1006 ) FILTRANE_inform%status
  WRITE( iout, 1007 ) problem%f
  WRITE( iout, 1008 ) FILTRANE_inform%nbr_iterations,                          &
                      FILTRANE_inform%nbr_cg_iterations
  WRITE( iout, 1009 ) FILTRANE_inform%nbr_c_evaluations
  WRITE( iout, 1010 ) FILTRANE_inform%nbr_J_evaluations
! Cleanup the problem.
  CALL NLPT_cleanup( problem )
  STOP
! Formats
1000 FORMAT(/)
1001 FORMAT(' Problem : GALEXAMPLE')
1002 FORMAT(' X1 = ',1PE20.12)
1003 FORMAT(' X2 = ',1PE20.12)
1004 FORMAT(' C1 = ',1PE20.12)
1005 FORMAT(' C2 = ',1PE20.12)
1006 FORMAT(' Exit condition number    = ',i10)
1007 FORMAT(' Objective function value =',1PE20.12)
1008 FORMAT(' Number of iterations     = ',i10,/,                              &
            ' Number of CG iterations  = ',i10)
1009 FORMAT(' Number of constraints evaluations =',i6)
1010 FORMAT(' Number of Jacobian evaluations    =',i6)
END PROGRAM GALAHAD_FILTRANE_EXAMPLE
