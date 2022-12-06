! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 16:40 GMT.

   MODULE GALAHAD_USEFILTRANE_double

!-------------------------------------------------------------------------------
!   U s e d   m o d u l e s   a n d   s y m b o l s
!-------------------------------------------------------------------------------

   USE GALAHAD_SYMBOLS,                                                        &
      OK                          => GALAHAD_SUCCESS,                          &
      MEMORY_FULL                 => GALAHAD_MEMORY_FULL,                      &
      SILENT                      => GALAHAD_SILENT,                           &
      TRACE                       => GALAHAD_TRACE,                            &
      DETAILS                     => GALAHAD_DETAILS,                          &
      COORDINATE                  => GALAHAD_COORDINATE,                       &
      USER_DEFINED                => GALAHAD_USER_DEFINED,                     &
      NONE                        => GALAHAD_NONE

   USE GALAHAD_NLPT_double      ! the NLP problem type

   USE GALAHAD_SPECFILE_double  ! the specfile tools

   USE GALAHAD_FILTRANE_double  ! the FILTRANE solver

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

   IMPLICIT NONE

   PRIVATE :: OK, MEMORY_FULL, SILENT, TRACE, DETAILS, COORDINATE,             &
              USER_DEFINED, NONE


   PUBLIC  :: USE_FILTRANE

 CONTAINS

!===============================================================================

   SUBROUTINE USE_FILTRANE( isif )

!  Reads a SIF problem and applies FILTRANE to it.

!  Argument
!  --------

   INTEGER, INTENT( IN ) :: isif

!         the number of the device on which the SIF problem file is opened.

!  Programming:  Ph. Toint and N. Gould, June 2003.

!===============================================================================


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
  TYPE( SPECFILE_item_type    ), DIMENSION( 7 ) :: specs

  INTEGER, PARAMETER :: ispec = 55      ! SPECfile device number
  INTEGER, PARAMETER :: iout = 6        ! stdout
  INTEGER, PARAMETER :: io_buffer = 11

  REAL( KIND = wp ), PARAMETER :: INFINITY = (10.0_wp)**19

  INTEGER :: iostat
  LOGICAL :: filexx
  LOGICAL :: is_specfile

  INTEGER :: soldev = 57                ! solution file device number
  INTEGER :: sumdev = 58                ! summary file device number
  LOGICAL :: full_sol  = .FALSE.
  LOGICAL :: write_sol = .FALSE.
  LOGICAL :: write_sum = .FALSE.
  INTEGER :: ierrout = 6                ! stderr
  CHARACTER ( LEN = 30 ) :: solfilename  = 'FILTRANE.sol'
  CHARACTER ( LEN = 30 ) :: sumfilename  = 'FILTRANE.sum'
  CHARACTER ( LEN = 16 ) :: specfilename = 'RUNFILTRANE.SPC'
  CHARACTER ( LEN = 16 ) :: algo_name    = 'RUNFILTRANE'

!-------------------------------------------------------------------------------
!  T h e   w o r k s
!-------------------------------------------------------------------------------

! Local variable

  INTEGER              :: nnzj, J_ne_plus_n, cutest_status
  REAL ( KIND = wp ), DIMENSION( 7 ) :: CUTEst_calls
  REAL ( KIND = wp ), DIMENSION( 2 ) :: CUTEst_time

! Setup the current CUTEst problem

  CALL CUTEst_initialize( problem, isif, iout, io_buffer, CUTEst_status )
  IF ( cutest_status /= 0 ) GO TO 910

! Update J_ne to take the peculiarity of CUTEst into account.

  J_ne_plus_n = problem%J_ne + problem%n

! Initialize FILTRANE

  CALL FILTRANE_initialize( FILTRANE_control, FILTRANE_inform, FILTRANE_data )

! Open the specfile for RUNFILT and FILTRANE

  INQUIRE( FILE = specfilename, EXIST = is_specfile )
  IF ( is_specfile ) THEN
    OPEN( ispec, file = specfilename, form = 'FORMATTED', status = 'OLD',      &
          IOSTAT = iostat )
    IF ( iostat > 0 ) THEN
       WRITE( iout, 205 ) specfilename, ispec, iostat
       STOP
    END IF

! Define the keywords.

    specs( 1 )%keyword = 'print-full-solution'
    specs( 2 )%keyword = 'write-solution'
    specs( 3 )%keyword = 'solution-file-name'
    specs( 4 )%keyword = 'solution-file-device'
    specs( 5 )%keyword = 'write-result-summary'
    specs( 6 )%keyword = 'result-summary-file-name'
    specs( 7 )%keyword = 'result-summary-file-device'

! Read the specfile for RUNFILTRANE.

    CALL SPECFILE_read( ispec, algo_name, specs, 7, ierrout )

! Interpret the result

    CALL SPECFILE_assign_logical( specs( 1 ), full_sol   , ierrout )
    CALL SPECFILE_assign_logical( specs( 2 ), write_sol  , ierrout )
    CALL SPECFILE_assign_string ( specs( 3 ), solfilename, ierrout )
    CALL SPECFILE_assign_integer( specs( 4 ), soldev     , ierrout )
    CALL SPECFILE_assign_logical( specs( 5 ), write_sum  , ierrout )
    CALL SPECFILE_assign_string ( specs( 6 ), sumfilename, ierrout )
    CALL SPECFILE_assign_integer( specs( 7 ), sumdev     , ierrout )

! Read the specfile for FILTRANE and close it.

    CALL FILTRANE_read_specfile( ispec, FILTRANE_control, FILTRANE_inform )
    CLOSE( ispec )
  END IF

! Check the preconditioning and external product options, as these are
! not desired when using the CUTEst interface.

  IF ( FILTRANE_control%prec_used == USER_DEFINED ) THEN
     FILTRANE_control%prec_used = NONE
     WRITE( ierrout, 300 )
  END IF

  IF ( FILTRANE_control%external_J_products ) THEN
     FILTRANE_control%external_J_products = .FALSE.
     WRITE( ierrout, 301 )
  END IF

! Apply the solver in a reverse communication loop.

  DO

     CALL FILTRANE_solve( problem, FILTRANE_control, FILTRANE_inform,          &
                          FILTRANE_data )

     SELECT CASE ( FILTRANE_inform%status )

     CASE ( 1, 2 )
        CALL CUTEST_ccfsg( cutest_status, problem%n, problem%m, problem%x,     &
                           problem%c, nnzj, J_ne_plus_n, problem%J_val,        &
                           problem%J_col, problem%J_row, .TRUE. )
       IF ( cutest_status /= 0 ) GO TO 910

     CASE ( 3 : 5 )
          CALL CUTEST_ccfsg( cutest_status, problem%n, problem%m, problem%x,   &
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

! Output results.

  IF ( full_sol ) THEN
     CALL NLPT_write_problem( problem, iout, DETAILS )
  ELSE
     CALL NLPT_write_problem( problem, iout, TRACE )
  END IF

  WRITE( iout, 202  )FILTRANE_inform%nbr_iterations,                           &
                     FILTRANE_inform%nbr_cg_iterations
  WRITE( iout, 200 ) problem%pname, problem%n, problem%m,                      &
                     FILTRANE_inform%nbr_c_evaluations,                        &
                     FILTRANE_inform%nbr_J_evaluations,                        &
                     FILTRANE_inform%status, problem%f,                        &
                     CUTEst_time( 1 ), CUTEst_time( 2 )
  WRITE( iout, 100 ) 'FILTRANE', FILTRANE_inform%nbr_iterations,               &
                     FILTRANE_inform%nbr_cg_iterations, problem%f,             &
                     FILTRANE_inform%status, CUTEst_time( 1 ), CUTEst_time(2), &
                     CUTEst_time( 1 ) + CUTEst_time( 2 )

! If required, write the solution to a file

  IF ( write_sol .AND. soldev > 0 ) THEN
     INQUIRE( FILE = solfilename, EXIST = filexx )
     IF ( filexx ) THEN
        OPEN( soldev, FILE = solfilename, FORM = 'FORMATTED',                  &
              STATUS = 'OLD', IOSTAT = iostat )
     ELSE
        OPEN( soldev, FILE = solfilename, FORM = 'FORMATTED',                  &
              STATUS = 'NEW', IOSTAT = iostat )
     END IF
     IF ( iostat /= 0 ) THEN
        WRITE( iout, 205 ) solfilename, soldev, iostat
     ELSE
        CALL NLPT_write_problem( problem, soldev, DETAILS )
        CLOSE( soldev )
     END IF
  END IF

! If required, write a result summary to a file

  IF ( write_sum .AND.sumdev > 0 ) THEN
     INQUIRE( FILE = sumfilename, EXIST = filexx )
     IF ( filexx ) THEN
        OPEN( sumdev, FILE = sumfilename, FORM = 'FORMATTED',                  &
              STATUS = 'OLD', IOSTAT = iostat )
     ELSE
        OPEN( sumdev, FILE = sumfilename, FORM = 'FORMATTED',                  &
              STATUS = 'NEW', IOSTAT = iostat )
     END IF
     IF ( iostat /= 0 ) THEN
        WRITE( iout, 205 ) sumfilename, sumdev, iostat
     ELSE
        WRITE( sumdev, 208 ) problem%pname, problem%n, problem%m,              &
                             FILTRANE_inform%nbr_iterations,                   &
                             FILTRANE_inform%nbr_cg_iterations, problem%f,     &
                             FILTRANE_inform%status, CUTEst_time( 2 )
        CLOSE( sumdev )
     END IF
  END IF

! Clean up the problem space

   CALL NLPT_cleanup( problem )
   CALL CUTEST_cterminate( cutest_status )
   RETURN

 910 CONTINUE
     WRITE( iout, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
       cutest_status
     RETURN

! Formats

100 FORMAT( /, '                         CG      objective',                   &
               '          < ------ time ----- > ', /,                          &
               ' Method  iterations  iterations    value  ',                   &
               '   status setup   solve   total', /,                           &
               ' ------  ----------  ----------  ---------',                   &
               '   ------ -----    ----   -----  ',/,                          &
                A8, 2I10, 3X, ES12.4, I6, 0P, 3F8.2 )
200 FORMAT( /, 24('*'), ' CUTEst statistics ', 24('*') //                      &
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
202 FORMAT(1x,'Number of iterations = ',i6,' Number of CG iterations = ',i10)
205 FORMAT(1x,'ERROR: could not open file ',a,' as unit ',i2,' (IOSTAT = ', i6,&
           ')' )
206 FORMAT(1x,'ERROR: Jacobian products are requested to be internal.')
207 FORMAT(1x,'ERROR: preconditioner is requested to be internal.')
208 FORMAT(a10,2x,i10,1x,i10,1x,i10,1x,i10,1x,1pe11.3,1x,i3,1x,0p,f8.2)
300 FORMAT(1x,'WARNING: usefiltrane does not support USER_DEFINED ',           &
              'preconditioners.',/,'         Abandoning preconditioning.')
301 FORMAT(1x,'WARNING: usefiltrane does not support external Jacobian ',      &
              'products ',/, '         Using internal products.')

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

  INTEGER :: j, iostat, J_size, n_free, cutest_status

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

  CALL CUTEST_cnames( cutest_status, problem%n, problem%m,                     &
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

  END IF

!-------------------------------------------------------------------------------
! Deallocate some useless vectors, if no contraint is present
!-------------------------------------------------------------------------------

  IF ( problem%m <= 0 ) THEN
     DEALLOCATE( problem%y, problem%c, problem%c_l, problem%c_u,               &
                 problem%equation, problem%linear, problem%cnames )
  END IF
  RETURN

 910 CONTINUE
     WRITE( errout, "( ' CUTEst error, status = ', i0, ', stopping' )" )       &
       cutest_status
     inform_status = - 98
     RETURN

!     Formats

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
311   FORMAT( 1x, 'ERROR: no memory for allocating ',                          &
                  'CUTEST_cnames( cutest_status, ',i6, ')')
312   FORMAT( 1x, 'ERROR: no memory for allocating J_val(',i10, ')')
313   FORMAT( 1x, 'ERROR: no memory for allocating J_col(',i10, ')')
314   FORMAT( 1x, 'ERROR: no memory for allocating J_row(',i10, ')')
315   FORMAT( 1x, 'ERROR: no memory for allocating g(',i6, ')')
316   FORMAT( 1x, 'ERROR: no memory for allocating z(',i6, ')')

   END SUBROUTINE CUTEst_initialize

!===============================================================================
!===============================================================================

   END SUBROUTINE USE_FILTRANE

   END MODULE GALAHAD_USEFILTRANE_double

