! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 17:00 GMT.

PROGRAM FILTRANE_MAIN

!-------------------------------------------------------------------------------
!   U s e d   m o d u l e s   a n d   s y m b o l s
!-------------------------------------------------------------------------------

   USE GALAHAD_NLPT_double      ! the NLP problem type

   USE GALAHAD_FILTRANE_double  ! the FILTRANE solver

   USE GALAHAD_SYMBOLS,                                                        &
      OK                          => GALAHAD_SUCCESS,                          &
      MEMORY_FULL                 => GALAHAD_MEMORY_FULL,                      &
      SILENT                      => GALAHAD_SILENT,                           &
      TRACE                       => GALAHAD_TRACE,                            &
!     ACTION                      => GALAHAD_ACTION,                           &
!     DETAILS                     => GALAHAD_DETAILS,                          &
!     DEBUG                       => GALAHAD_DEBUG,                            &
!     CRAZY                       => GALAHAD_CRAZY,                            &
      COORDINATE                  => GALAHAD_COORDINATE,                       &
      FIXED                       => GALAHAD_FIXED,                            &
      RANGE                       => GALAHAD_RANGE,                            &
      LOWER                       => GALAHAD_LOWER,                            &
      UPPER                       => GALAHAD_UPPER,                            &
      FREE                        => GALAHAD_FREE,                             &
      USER_DEFINED                => GALAHAD_USER_DEFINED,                     &
      NONE                        => GALAHAD_NONE

!-------------------------------------------------------------------------------
!   A c c e s s 
!-------------------------------------------------------------------------------

  IMPLICIT NONE

!  PRIVATE :: OK, MEMORY_FULL_ SILENT, TRACE, ACTION,                          &
!             DETAILS, DEBUG, CRAZY, COORDINATE, FIXED, LOWER, UPPER,          &
!             RANGE, FREE, USER_DEFINED, NONE

!-------------------------------------------------------------------------------
!   P r e c i s i o n
!-------------------------------------------------------------------------------

  INTEGER, PARAMETER :: sp = KIND( 1.0 )
  INTEGER, PARAMETER :: dp = KIND( 1.0D+0 )
  INTEGER, PARAMETER :: wp = dp

!-------------------------------------------------------------------------------
!   D e c l a r a t i o n s
!-------------------------------------------------------------------------------

  TYPE( NLPT_problem_type     ) :: problem
  TYPE( FILTRANE_control_type ) :: FILTRANE_control
  TYPE( FILTRANE_inform_type  ) :: FILTRANE_inform
  TYPE( FILTRANE_data_type    ) :: FILTRANE_data
  
  INTEGER, PARAMETER :: ispec = 55      ! SPECfile  device number
  INTEGER, PARAMETER :: isif  = 56      ! OUTSDIF.d device number
  INTEGER, PARAMETER :: iout = 6        ! stderr and stdout
  INTEGER, PARAMETER :: io_buffer = 11

  REAL( KIND = wp ), PARAMETER :: INFINITY = (10.0_wp)**19

  INTEGER :: iostat

!-------------------------------------------------------------------------------
!  T h e   w o r k s
!-------------------------------------------------------------------------------

! Local variable

  INTEGER              :: status, n_bounded, m_not_equal, nnzj, J_ne_plus_n,   &
                          CUTEst_inform, cutest_status
  REAL, DIMENSION( 7 ) :: CUTEst_calls
  REAL, DIMENSION( 2 ) :: CUTEst_time

! Open the SIF description file

  OPEN( isif, file = 'OUTSDIF.d', form = 'FORMATTED', status = 'OLD',          &
        IOSTAT = iostat )
  IF ( iostat > 0 ) THEN
     WRITE( iout, 201 )  isif
     STOP
  END IF
  REWIND( isif )

! Setup the current CUTEst problem

  CALL CUTEst_initialize( problem, isif, iout, io_buffer, CUTEst_inform )
  CLOSE( isif )
  IF ( CUTEst_inform /= OK ) STOP

! Update J_ne to take the peculiarity of CUTEst into account.

  J_ne_plus_n = problem%J_ne + problem%n

  CALL NLPT_write_stats( problem, iout )

! Initialize FILTRANE

  CALL FILTRANE_initialize( FILTRANE_control, FILTRANE_inform, FILTRANE_data )

! Read the FILTRANE spec file

  OPEN( ispec, file = 'FILTRANE.SPC', form = 'FORMATTED', status = 'OLD',      &
        IOSTAT = iostat )
  IF ( iostat > 0 ) THEN
     WRITE( iout, 205 )  ispec
     STOP
  END IF
  CALL FILTRANE_read_specfile( ispec, FILTRANE_control, FILTRANE_inform )
  CLOSE( ispec )

! Check the preconditioning and external product options, as these are
! not desired when using the CUTEst interface.

  IF ( FILTRANE_control%prec_used == USER_DEFINED ) THEN
     FILTRANE_control%prec_used = NONE
  END IF

  IF ( FILTRANE_control%external_J_products ) THEN
     FILTRANE_control%external_J_products = .FALSE.
  END IF

! Apply the solver in a reverse communication loop

  DO 
  
     CALL FILTRANE_solve( problem, FILTRANE_control, FILTRANE_inform,          &
                          FILTRANE_data )

     SELECT CASE ( FILTRANE_inform%status )
  
     CASE ( 1, 2 )
        CALL CUTEST_ccfsg( cutest_status, problem%n, problem%m, problem%x,     &
                           problem%c, nnzj, J_ne_plus_n, problem%J_val,        &
                           problem%J_col, problem%J_row, .TRUE. )
        IF ( cutest_status /= 0 ) GO TO 910
  
     CASE ( 3:5 )
        CALL CUTEST_ccfsg( cutest_status, problem%n, problem%m, problem%x,     &
                           problem%c, nnzj, J_ne_plus_n, problem%J_val,        &
                           problem%J_col, problem%J_row, .FALSE. )
        IF ( cutest_status /= 0 ) GO TO 910
  
     CASE ( 6 )
        CALL CUTEST_csgr( cutest_status, problem%n, problem%m,                 &
                          problem%x, problem%y,.FALSE.,                        &
                          nnzj, J_ne_plus_n, problem%J_val,                    &
                          problem%J_col, problem%J_row )
        IF ( cutest_status /= 0 ) GO TO 910
  
     CASE ( 7 )

        WRITE( iout, 206 )
        EXIT

     CASE ( 8:11 )
  
        WRITE( iout, 206 )
        EXIT

     CASE ( 12:14 )

        WRITE( iout, 207 )
        EXIT

     CASE ( 15, 16 )
        CALL CUTEST_chprod( cutest_status, problem%n, problem%m,               &
                            .NOT. FILTRANE_data%RC_newx,                       &
                            problem%x, problem%y, FILTRANE_data%RC_v,          &
                            FILTRANE_data%RC_Mv )
        IF ( cutest_status /= 0 ) GO TO 910

     CASE DEFAULT
  
        EXIT
  
     END SELECT
  
  END DO ! end of the reverse communication loop

! Get the CUTEst statistics.

  CALL CUTEST_creport( cutest_status, CUTEst_calls, CUTEst_time )
  IF ( cutest_status /= 0 ) GO TO 910

! Terminate FILTRANE.

  FILTRANE_control%print_level = SILENT
  CALL FILTRANE_terminate( FILTRANE_control, FILTRANE_inform, FILTRANE_data )

! Output results

  CALL NLPT_write_problem( problem, iout, TRACE )
!  CALL NLPT_write_problem( problem, iout, DETAILS )

  WRITE( iout, 202  )FILTRANE_inform%nbr_iterations,                           &
                     FILTRANE_inform%nbr_cg_iterations
  WRITE( iout, 203 ) FILTRANE_inform%nbr_c_evaluations
  WRITE( iout, 204 ) FILTRANE_inform%nbr_J_evaluations
  WRITE( iout, 200 ) problem%pname, problem%n, problem%m,                      &
                     FILTRANE_inform%nbr_c_evaluations,                        &
                     FILTRANE_inform%nbr_J_evaluations,                        &
                     FILTRANE_inform%status, problem%f,                        &
                     CUTEst_time( 1 ), CUTEst_time( 2 )
  WRITE( iout, 100 ) problem%pname, 'FILTRANE', FILTRANE_inform%nbr_iterations,&
                     FILTRANE_inform%nbr_cg_iterations, problem%f,             &
                     FILTRANE_inform%status, CUTEst_time( 1 ), CUTEst_time( 2 ), &
                     CUTEst_time( 1 ) + CUTEst_time( 2 )

! Clean up the problem space

  CALL NLPT_cleanup( problem )
  CALL CUTEST_cterminate( cutest_status )
  STOP

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )          &
       cutest_status
     inform_status = - 98
     STOP

! Formats

100 FORMAT( /, ' Problem: ', A10, //,                                          &
               '                         CG      objective',                   &
               '          < ------ time ----- > ', /,                          &
               ' Method  iterations  iterations    value  ',                   &
               '   status setup   solve   total', /,                           &
               ' ------  ----------  ----------  ---------',                   &
               '   ------ -----    ----   -----  ',/,                          &
                A8, 2I10, 3X, ES12.4, I6, 0P, 3F8.2 ) 
200 FORMAT( /, 24('*'), ' CUTEst statistics ', 24('*') //                       &
            ,' Code used               :  FILTRANE',    /                      &
            ,' Problem                 :  ', A10,    /                         &
            ,' # variables             =      ', I10 /                         &
            ,' # constraints           =      ', I10 /                         &
            ,' # constraints functions =      ', I10 /                         &
            ,' # constraints gradients =      ', I10 /                         &
             ' Exit code               =      ', I10 /                         &
            ,' Final f                 = ', 1pE15.7 /                          &
            ,' Set up time             =        ', 0P, F8.2, ' seconds' /      &
             ' Solve time              =        ', 0P, F8.2, ' seconds' //     &
            66('*') / )
201 FORMAT(1x,'ERROR: could not open file OUTSDIF.d as unit ',i2)
202 FORMAT(1x,'Number of iterations = ',i6,' Number of CG iterations = ',i10)
203 FORMAT(1x,'Number of constraints evaluations = ',i10)
204 FORMAT(1x,'Number of Jacobian evaluations    = ',i10)
205 FORMAT(1x,'ERROR: could not open file FILTRANE.SPC as unit ',i2)
206 FORMAT(1x,'ERROR: Jacobian products are requested to be internal.')
207 FORMAT(1x,'ERROR: preconditioner is requested to be internal.')

CONTAINS

!===============================================================================

      SUBROUTINE CUTEst_initialize( problem, isif, errout, io_buffer,          &
                                    inform_status )

!     Initializes the problem from its CUTEst description.

!     Arguments

      TYPE ( NLPT_problem_type ), INTENT( OUT ) :: problem
 
!            the problem;

      INTEGER, INTENT( IN ) :: isif

!            the device file for the OUTSDIF.d file

      INTEGER, INTENT( IN ) :: errout

!            the device number for error disagnostics;

      INTEGER, INTENT( IN ) :: io_buffer

!            CUTEst internal read/writes

      INTEGER, INTENT( OUT ) :: inform_status

!            the exit code.

!     Programming: Ph. Toint, November 2002.
!
!===============================================================================

! Local variables

  INTEGER :: i, j, iostat, J_size, n_free, cutest_status

!-------------------------------------------------------------------------------
! Initialize the exit status.
!-------------------------------------------------------------------------------

  inform_status = OK

!-------------------------------------------------------------------------------
! Set the infinity value.
!-------------------------------------------------------------------------------

  problem%infinity = INFINITY

! --------------------------------------------------------------------------
! Get the problem's dimensions
! --------------------------------------------------------------------------
  
  CALL CUTEST_cdimen( cutest_status, isif, problem%n, problem%m )
  IF ( cutest_status /= 0 ) GO TO 910
  
! --------------------------------------------------------------------------
! Allocate the problem's structure
! --------------------------------------------------------------------------
  
  ALLOCATE( problem%x( problem%n ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 300 ) problem%n
     RETURN
  END IF
  
  ALLOCATE( problem%x_l( problem%n ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 301 )  problem%n
     RETURN
  END IF
  
  ALLOCATE( problem%x_u( problem%n ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 302 ) problem%n
     RETURN
  END IF
  
  ALLOCATE( problem%x_status( problem%n ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 303 ) problem%n
     RETURN
  END IF

  ALLOCATE( problem%vnames( problem%n ) )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 304 ) problem%n
     RETURN
  END IF
  
  ALLOCATE( problem%equation( problem%m ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 305 ) problem%m
     RETURN
  END IF
  
  ALLOCATE( problem%linear( problem%m ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 306 ) problem%m
     RETURN
  END IF
  
  ALLOCATE( problem%c( problem%m ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 307 ) problem%m
     RETURN
  END IF
  
  ALLOCATE( problem%c_l( problem%m ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 308 ) problem%m
     RETURN
  END IF
  
  ALLOCATE( problem%c_u( problem%m ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 309 ) problem%m
     RETURN
  END IF
  
  ALLOCATE( problem%y( problem%m ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 310 ) problem%m
     RETURN
  END IF
  
  ALLOCATE( problem%cnames( problem%m ) )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 311 ) problem%m
     RETURN
  END IF
  
! --------------------------------------------------------------------------
! CUTEst setup
! --------------------------------------------------------------------------
  
  CALL CUTEST_csetup( cutest_status, isif, errout, io_buffer,                  &
                      problem%n, problem%m, problem%x,                         &
                      problem%x_l, problem%x_u,                                &
                      problem%y, problem%c_l, problem%c_u,                     &
                      problem%equation, problem%linear, 0, 0, 0 )
  IF ( cutest_status /= 0 ) GO TO 910
  
  CALL CUTEST_cnames( cutest_status, problem%n, problem%m,                    &
                      problem%pname, problem%vnames, problem%cnames )
  IF ( cutest_status /= 0 ) GO TO 910
  
! --------------------------------------------------------------------------
! Allocate the Jacobian space.
! --------------------------------------------------------------------------
  
  CALL CUTEST_cdimsj( cutest_status, J_size )
  IF ( cutest_status /= 0 ) GO TO 910
  problem%J_ne = J_size - problem%n

  ALLOCATE( problem%J_val( J_size ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 312 )  J_size
     RETURN
  END IF
  
  ALLOCATE( problem%J_col( J_size ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 313 ) J_size
     RETURN
  END IF
  
  ALLOCATE( problem%J_row( J_size ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 314 ) J_size
     RETURN
  END IF
  problem%J_type = COORDINATE
  
!---------------------------------------------------------------------------
!    The gradient
!---------------------------------------------------------------------------
  
  ALLOCATE( problem%g( problem%n ), STAT = iostat )
  IF ( iostat /= 0 ) THEN
     inform_status = MEMORY_FULL
     WRITE( errout, 315 ) problem%n
     RETURN
   END IF

!-------------------------------------------------------------------------------
! Analyze the problem variables.
!-------------------------------------------------------------------------------

  n_free  = 0
  DO j = 1, problem%n
     IF ( problem%x_l( j ) <= - INFINITY .AND. problem%x_u( j ) >= INFINITY )  &
        n_free  = n_free + 1
  END DO

!-------------------------------------------------------------------------------
! If they are not all free, allocate the vector of dual variables
!-------------------------------------------------------------------------------

  IF ( n_free < problem%n  ) THEN

     ALLOCATE( problem%z( problem%n ), STAT = iostat )
     IF ( iostat /= 0 ) THEN
        inform_status = MEMORY_FULL
        WRITE( errout, 316 ) problem%n
        RETURN
     END IF
     problem%z = 0.0_wp

!-------------------------------------------------------------------------------
! Deallocate some useless vectors, if no bound is present
!-------------------------------------------------------------------------------

  ELSE
     NULLIFY( problem%z )
  END IF

!-------------------------------------------------------------------------------
! Deallocate some useless vectors, if no contraint is present
!-------------------------------------------------------------------------------

  IF ( problem%m <= 0 ) THEN
     DEALLOCATE( problem%y, problem%c, problem%c_l, problem%c_u,               &
                 problem%equation, problem%linear, problem%cnames )     
  END IF
 
!-------------------------------------------------------------------------------
! Nullify the derivative pointers for the Hessian
!-------------------------------------------------------------------------------

  NULLIFY( problem%H_val, problem%H_row, problem%H_col, problem%H_ptr,         &
           problem%gL)
  RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )          &
       cutest_status
     inform_status = - 98
     RETURN

!     Formats

101   FORMAT( 1x, i4, 1x, a10, 3x,  ES12.4 )
102   FORMAT( 1x, i4, 1x, a10, 3x, 2ES12.4 )
103   FORMAT( 1x, i4, 1x, a10, 3x, 3ES12.4 )
202   FORMAT( 1x, i4, 1x, a10, 3x, 2ES12.4, ' linear' )
203   FORMAT( 1x, i4, 1x, a10, 3x, 3ES12.4, ' linear' )
300   FORMAT( 1x, 'ERROR: no memory for allocating x(',i6, ')')
301   FORMAT( 1x, 'ERROR: no memory for allocating x_l(',i6, ')')
302   FORMAT( 1x, 'ERROR: no memory for allocating x_u(',i6, ')')
303   FORMAT( 1x, 'ERROR: no memory for allocating x_status(',i6, ')')
304   FORMAT( 1x, 'ERROR: no memory for allocating vnames(',i6, ')')
305   FORMAT( 1x, 'ERROR: no memory for allocating equation(',i6, ')')
306   FORMAT( 1x, 'ERROR: no memory for allocating linear(',i6, ')')
307   FORMAT( 1x, 'ERROR: no memory for allocating c(',i6, ')')
308   FORMAT( 1x, 'ERROR: no memory for allocating c_l(',i6, ')')
309   FORMAT( 1x, 'ERROR: no memory for allocating c_u(',i6, ')')
310   FORMAT( 1x, 'ERROR: no memory for allocating y(',i6, ')')
311   FORMAT( 1x, 'ERROR: no memory for allocating CUTEST_cnames( cutest_status, ',i6, ')')
312   FORMAT( 1x, 'ERROR: no memory for allocating J_val(',i10, ')')
313   FORMAT( 1x, 'ERROR: no memory for allocating J_col(',i10, ')')
314   FORMAT( 1x, 'ERROR: no memory for allocating J_row(',i10, ')')
315   FORMAT( 1x, 'ERROR: no memory for allocating g(',i6, ')')
316   FORMAT( 1x, 'ERROR: no memory for allocating z(',i6, ')')

   END SUBROUTINE CUTEst_initialize

!===============================================================================
!===============================================================================

END PROGRAM FILTRANE_main

