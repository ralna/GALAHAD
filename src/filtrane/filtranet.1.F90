! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

! This version: 26 V 2003
! Programming: Ph. Toint

  USE GALAHAD_NLPT_double      ! the problem type
  USE GALAHAD_FILTRANE_double  ! the FILTRANE solver
  USE GALAHAD_SYMBOLS
  IMPLICIT NONE
  INTEGER, PARAMETER                :: wp = KIND( 1.0D+0 )
  INTEGER,           PARAMETER      :: ispec = 55      ! SPECfile device number
  INTEGER,           PARAMETER      :: iout = 6        ! stdout and stderr
  REAL( KIND = wp ), PARAMETER      :: INFINITY = (10.0_wp)**19
  TYPE( NLPT_problem_type     )     :: problem 
  TYPE( FILTRANE_control_type )     :: control
  TYPE( FILTRANE_inform_type  )     :: inform
  TYPE( FILTRANE_data_type    )     :: data
  INTEGER                           :: J_size, test
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
  problem%x        = (/   1.0D0 ,  1.0D0 /)
  problem%x_l      = (/  -2.0D0 , -2.0D0 /)
  problem%x_u      = (/   2.0D0 ,  2.0D0 /)
  problem%c_l      = (/   0.0D0 ,  0.0D0 /)
  problem%c_u      = (/   0.0D0 ,  0.0D0 /)
  problem%equation = (/  .TRUE. , .TRUE. /)
  NULLIFY( problem%H_ptr, problem%J_ptr, problem%gL, problem%linear,           &
           problem%H_val, problem%H_row, problem%H_col, problem%H_ptr )

! Initialize FILTRANE.

  CALL FILTRANE_initialize( control, inform, data )

! Read the FILTRANE spec file.

  OPEN( ispec, file = 'FILTRANE_t.SPC', form = 'FORMATTED', status = 'OLD' )
  CALL FILTRANE_read_specfile( ispec, control, inform )
  CLOSE( ispec )

! Loop on the test cases.

  test = 0
  DO 

!    Now apply the solver in the reverse communication loop.

     DO 

        CALL FILTRANE_solve( problem, control, inform, data )

!        write(*,*) ' >>> inform%status =', inform%status

        SELECT CASE ( inform%status )

        CASE ( 1, 2 ) ! constraints values and Jacobian

           CALL GFT_compute_c( problem%x, problem%c )
           CALL GFT_compute_J( problem%x, problem%J_val, problem%J_row,        &
                               problem%J_col )

        CASE ( 3:5 ) ! constraints values only

           CALL GFT_compute_c( problem%x, problem%c )

        CASE ( 6 ) ! Jacobian only

           CALL GFT_compute_J( problem%x, problem%J_val, problem%J_row,        &
                               problem%J_col )

        CASE ( 7 ) ! Jacobian times v

           CALL GFT_compute_J_times_v( problem%x, data%RC_v, data%RC_Mv,.FALSE.)

        CASE ( 8:11 ) ! Jacobian transpose times v
  
           CALL GFT_compute_J_times_v( problem%x, data%RC_v, data%RC_Mv, .TRUE.)

        CASE ( 12:14 ) ! preconditioning

           CALL GFT_prec( data%RC_Pv )

        CASE ( 15, 16 ) ! product times the Hessian of the Lagrangian
!          Note that H2, the Hessian of C2 is identically zero, since this 
!          constraint is linear. Hence the terms in y(2)*H2 disappear.

           IF ( data%RC_newx ) THEN
              H1( 1 ) = 60.0D0
              H1( 2 ) = 1.0D0
              H1( 3 ) = 12.0D0 * problem%x( 2 )
           END IF
           data%RC_Mv( 1 ) = problem%y( 1 ) * H1( 1 ) * data%RC_v( 1 ) +       &
                             problem%y( 1 ) * H1( 2 ) * data%RC_v( 2 ) 
           data%RC_Mv( 2 ) = problem%y( 1 ) * H1( 2 ) * data%RC_v( 1 ) +       &
                             problem%y( 1 ) * H1( 3 ) * data%RC_v( 2 ) 

        CASE DEFAULT

           EXIT

        END SELECT

     END DO ! end of the reverse communication loop

     IF( test == 26 ) WRITE( iout, 1001 ) problem%f

!    Restore the data.

     problem%n        = 2
     problem%m        = 2
     problem%J_ne     = 4
     problem%J_type   = GALAHAD_COORDINATE
     problem%infinity = INFINITY
     IF ( .NOT. ALLOCATED ( problem%x )   ) ALLOCATE( problem%x( problem%n ) )
     problem%x        = (/ 1.0D0, 1.0D0 /)
     IF ( .NOT. ALLOCATED ( problem%x_l ) ) ALLOCATE( problem%x_l( problem%n ))
     problem%x_l      = (/ -2.0D0, -2.0D0 /)
     IF ( .NOT. ALLOCATED ( problem%x_u ) ) ALLOCATE( problem%x_u( problem%n ))
     problem%x_u      = (/  2.0D0,  2.0D0 /)
     IF ( .NOT. ALLOCATED ( problem%c )   ) ALLOCATE( problem%c( problem%m ))
     IF ( .NOT. ALLOCATED ( problem%c_l ) ) ALLOCATE( problem%c_l( problem%m ))
     problem%c_l      = (/ 0.0D0, 0.0D0 /)
     IF ( .NOT. ALLOCATED ( problem%c_u ) ) ALLOCATE( problem%c_u( problem%m ))
     problem%c_u      = (/ 0.0D0, 0.0D0 /)
     IF ( .NOT. ALLOCATED ( problem%equation ) )                              &
        ALLOCATE( problem%equation( problem%m ) )
     problem%equation = (/  .TRUE. , .TRUE. /)
     IF ( .NOT. ALLOCATED ( problem%x_status ) )                              &
        ALLOCATE( problem%x_status( problem%n ) )
     IF ( .NOT. ALLOCATED ( problem%y )     ) ALLOCATE( problem%y( problem%m ))
     IF ( .NOT. ALLOCATED ( problem%g )     ) ALLOCATE( problem%g( problem%n ))
     IF ( .NOT. ALLOCATED ( problem%J_val ) ) ALLOCATE( problem%J_val( J_size))
     IF ( .NOT. ALLOCATED ( problem%J_row ) ) ALLOCATE( problem%J_row( J_size))
     IF ( .NOT. ALLOCATED ( problem%J_col ) ) ALLOCATE( problem%J_col( J_size))

!    Re-initialize FILTRANE.

     CALL FILTRANE_initialize( control, inform, data )
     control%print_level = GALAHAD_TRACE

!    Select the next test case.

     test = test + 1

     WRITE( iout, 1000 )

     SELECT CASE( test )
     CASE ( 1 )
        WRITE( iout, 1002 ) test, 'Impossible n'
        problem%n = 0
     CASE ( 2 ) 
        WRITE( iout, 1002 ) test, 'Impossible m'
        problem%m = -1
     CASE ( 3 ) 
        WRITE( iout, 1002 ) test, 'Impossible s%stage'
        data%stage = 89
     CASE ( 4 )
        WRITE( iout, 1002 ) test, 'x unallocated'
        DEALLOCATE( problem%x )
     CASE ( 5 )
        WRITE( iout, 1002 ) test, 'x_l unallocated'
        DEALLOCATE( problem%x_l )
     CASE ( 6 )
        WRITE( iout, 1002 ) test, 'x_u unallocated'
        DEALLOCATE( problem%x_u )
     CASE ( 7 )
        WRITE( iout, 1002 ) test, 'x_status unallocated'
        DEALLOCATE( problem%x_status )
     CASE ( 8 )
        WRITE( iout, 1002 ) test, 'c unallocated'
        DEALLOCATE( problem%c )
     CASE ( 9 )
        WRITE( iout, 1002 ) test, 'c_l unallocated'
        DEALLOCATE( problem%c_l )
     CASE ( 10 )
        WRITE( iout, 1002 ) test, 'c_u unallocated'
        DEALLOCATE( problem%c_u )
     CASE ( 11 )
        WRITE( iout, 1002 ) test, 'equation unallocated'
        DEALLOCATE( problem%equation )
     CASE ( 12 )
        WRITE( iout, 1002 ) test, 'y unallocated'
        DEALLOCATE( problem%y )
     CASE ( 13 )
        WRITE( iout, 1002 ) test, 'g unallocated'
        DEALLOCATE( problem%g )
     CASE ( 14 )
        WRITE( iout, 1002 ) test, 'J_val unallocated'
        DEALLOCATE( problem%J_val )
     CASE ( 15 )
        WRITE( iout, 1002 ) test, 'J_row unallocated'
        DEALLOCATE( problem%J_row )
     CASE ( 16 )
        WRITE( iout, 1002 ) test, 'J_col unallocated'
        DEALLOCATE( problem%J_col )
     CASE ( 17 )
        WRITE( iout, 1002 ) test, 'Impossible nbr_groups'
        control%grouping = GALAHAD_USER_DEFINED
        control%nbr_groups = -1
     CASE ( 18 )
        WRITE( iout, 1002 ) test, 'control%group unallocated'
        control%grouping = GALAHAD_USER_DEFINED
        control%nbr_groups = 1
     CASE ( 19 )
        WRITE( iout, 1002 ) test, 'Impossible control%group(1)'
        control%grouping = GALAHAD_USER_DEFINED
        control%nbr_groups = 1
        ALLOCATE( control%group( 4 ) )
        control%group( 1 ) = 3026
        control%group( 2 ) = 1
        control%group( 3 ) = 1
        control%group( 4 ) = 1
     CASE ( 20 )
        WRITE( iout, 1002 ) test, 'Correct USER_DEFINED groups'
        control%grouping = GALAHAD_USER_DEFINED
        control%nbr_groups = 1
        control%group( 1 ) = 1
        control%group( 2 ) = 1
        control%group( 3 ) = 1
        control%group( 4 ) = 1
     CASE ( 21 )
        DEALLOCATE( control%group ) 
        WRITE( iout, 1002 ) test, 'No checkpoint file'
        control%restart_from_checkpoint = .TRUE.
     CASE ( 22 )
        WRITE( iout, 1002 ) test, 'Checkpointing after iteration 2 and stop'
        control%checkpoint_freq = 2
        control%max_iterations = 2
     CASE ( 23 )
        WRITE( iout, 1002 ) test,                                              &
             'Restart from checkpoint and use 1 automatic group'
        control%restart_from_checkpoint = .TRUE.
        control%grouping = GALAHAD_AUTOMATIC
        control%nbr_groups = 1
     CASE ( 24 )
        WRITE( iout, 1002 ) test,'Balance 2 automatic groups for signed filter,'
        WRITE( iout, 1003 )      'user-defined preconditioner'
        control%grouping = GALAHAD_AUTOMATIC
        control%balance_group_values = .TRUE.
        control%nbr_groups = 2
        control%filter_sign_restriction = .TRUE.
        control%prec_used = GALAHAD_USER_DEFINED
     CASE ( 25 )
        WRITE( iout, 1002 ) test,                                              &
             'Use Newton model,banded preconditioner and large radius'
        WRITE( iout, 1003 ) 'and print between iterations 4 and 8'
        control%model_type = GALAHAD_NEWTON
        control%prec_used  = GALAHAD_BANDED
        control%start_print = 4
        control%stop_print  = 8
        control%initial_radius = 2.0_wp
     CASE ( 26 ) 
        WRITE( iout, 1002 ) test, 'Keep best point and stop at iteration 3'
        control%save_best_point = .TRUE.
        control%max_iterations  = 3
     CASE ( 27 )
        WRITE( iout, 1002 ) test,                                              &
                     'stop on unpreconditioned gradient, banded preconditioner,'
        WRITE( iout, 1003 ) 'filter_increment = 1'
        control%stop_on_prec_g = .FALSE.
        control%prec_USED = GALAHAD_BANDED
        control%filter_size_increment = 1
     CASE ( 28 )
        WRITE( iout, 1002 ) test,                                              &
                          'no weak acceptance, no removal of dominated entries,'
        WRITE( iout, 1003 ) 'best-reduction models, external Jacobian products'
        control%weak_accept_power = -1
        control%remove_dominated = .FALSE.
        control%model_criterion = GALAHAD_BEST_REDUCTION
        DEALLOCATE( problem%J_val, problem%J_row , problem%J_col )
        control%external_J_products = .TRUE.
     CASE ( 29 )
        WRITE( iout, 1002 ) test,'bounds only'
        problem%x_l      = (/  10.0D0 , 10.0D0 /)
        problem%x_u      = (/  15.0D0 , 15.0D0 /)
        problem%m = 0
        DEALLOCATE( problem%c, problem%c_l, problem%c_u, problem%equation )
        DEALLOCATE( problem%y )
        DEALLOCATE( problem%J_val, problem%J_row , problem%J_col )
     CASE ( 30 )
        EXIT
     END SELECT

     WRITE( iout, 1000 )

  END DO ! end of the test loop

! Terminate FILTRANE. 

! control%print_level = GALAHAD_SILENT
  CALL FILTRANE_terminate( control, inform, data )

! Cleanup the problem.

  CALL NLPT_cleanup( problem )

  STOP

! Formats

1000 FORMAT(/,' ====================================================',         &
              '====================',/)
1001 FORMAT(' f = ',1pE20.12)
1002 FORMAT(1x,i2,')',5x,a)
1003 FORMAT(9x,a)
CONTAINS

SUBROUTINE GFT_compute_c( x, c )
REAL( KIND = wp ), DIMENSION( 2 ), INTENT(  IN ) :: x
REAL( KIND = wp ), DIMENSION( 2 ), INTENT( OUT ) :: c
c( 1 ) = 30.0D0 * x( 1 ) ** 2 + 2.0D0 * x( 2 ) ** 3 + x( 1 ) * x( 2 )
c( 2 ) = x( 1 ) + x( 2 )
RETURN
END SUBROUTINE GFT_compute_c

SUBROUTINE GFT_compute_J( x, J_val, J_row, J_col )
REAL( KIND = wp ), DIMENSION( 2 ), INTENT(  IN ) :: x
REAL( KIND = wp ), DIMENSION( 4 ), INTENT( OUT ) :: J_val
INTEGER          , DIMENSION( 4 ), INTENT( OUT ) :: J_row
INTEGER          , DIMENSION( 4 ), INTENT( OUT ) :: J_col
J_val( 1 ) = 60.0D0 * x( 1 )  + problem%x( 2 ) 
J_val( 2 ) = 1.0D0
J_val( 3 ) = 6.0D0 * problem%x( 2 ) ** 2 + problem%x( 1 )
J_val( 4 ) = 1.0D0
J_row      = (/ 1, 2, 1, 2 /)
J_col      = (/ 1, 1, 2, 2 /)
RETURN
END SUBROUTINE GFT_compute_J

SUBROUTINE GFT_compute_J_times_v( x, v, Jv, trans )
REAL( KIND = wp ), DIMENSION( 2 ), INTENT(  IN  ) :: x, v
REAL( KIND = wp ), DIMENSION( 2 ), INTENT(  OUT ) :: Jv
LOGICAL, INTENT( IN ) :: trans
Jv = 0
IF ( trans ) THEN
   Jv( 1 ) = ( 60.0D0 * x( 1 )  + problem%x( 2 ) ) * v( 1 ) + v( 2 ) 
   Jv( 2 ) = ( 6.0D0 * problem%x( 2 ) ** 2 + problem%x( 1 ) ) * v( 1 ) + v( 2 )
ELSE
   Jv( 1 ) = ( 60.0D0 * x( 1 )  + problem%x( 2 ) ) * v( 1 ) +                  &
             ( 6.0D0 * problem%x( 2 ) ** 2 + problem%x( 1 ) ) * v( 2 )
   Jv( 2 ) = v( 1 ) + v( 2 )
END IF
RETURN
END SUBROUTINE GFT_compute_J_times_v

SUBROUTINE GFT_prec( x )
REAL( KIND = wp ), DIMENSION( 2 ), INTENT(  INOUT  ) :: x
x( 1 ) = x ( 1 ) / 15.0D0
RETURN
END SUBROUTINE GFT_prec

END PROGRAM GALAHAD_FILTRANE_TEST

