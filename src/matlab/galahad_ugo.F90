#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.0 - 2022-03-14 AT 16:05 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_UGO
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  find a global bound-constrained minimizer of a twice differentiable objective
!  function f(x) of a real variable x over the finite interval [x_l,x_u]
!
!  Simple usage -
!
!  to find the minimizer
!   [ x, f, g, h, inform ]
!    = galahad_ugo( x_l, x_u, eval_fgh, control )
!   [ x, f, g, h, inform, z ]
!    = galahad_ugo( x_l, x_u, eval_fgh )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!    = galahad_ugo( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, f, g, h, inform ]
!    = galahad_ugo( 'existing', x_l, x_u, eval_fgh, control )
!   [ x, f, g, h, inform ]
!    = galahad_ugo( 'existing', x_l, x_u, eval_fgh )
!
!  to remove data structures after solution
!   galahad_ugo( 'final' )
!
!  Usual Input -
!       x_l: the finite lower bound x_l
!       x_u: the finite upper bound x_u
!    eval_fgh: a user-provided subroutine named eval_fgh.m for which
!              [f,g,h,status] = eval_fgh(x)
!            returns the value of objective function f and its first
!            derivative g = f'(x) at x. Additionally, if
!            control.second_derivative_available is true, also returns
!            the value of the second derivative h = f''(x) at x; h need
!            not be set otherwise. status should be set to 0 if the
!            evaluations succeed, and a non-zero value if an evaluation fails.
!
!  Optional Input -
!     control: a structure containing control parameters.
!              The components are of the form control.value, where
!              value is the name of the corresponding component of
!              the derived type UGO_control_type as described in
!              the manual for the fortran 90 package GALAHAD_UGO.
!              See: http://galahad.rl.ac.uk/galahad-www/doc/ugo.pdf
!
!  Usual Output -
!          x: the estimated global minimizer
!          f: the objective function value at x
!          g: the first derivative of the objective function value at x
!          h: the second derivative of the objective function value at x
!             when control.second_derivative_available is true
!
!  Optional Output -
!    control: see above. Returned values are the defaults
!     inform: a structure containing information parameters
!             The components are of the form inform.value, where
!             value is the name of the corresponding component of the
!             derived type UGO_inform_type as described in the manual
!             for the fortran 90 package GALAHAD_UGO. The component
!             inform.time is itself a structure, holding the
!             components of the derived types UGO_time_type.
!             See: http://galahad.rl.ac.uk/galahad-www/doc/ugo.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.0, March 14th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
!     USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_UGO_MATLAB_TYPES
      USE GALAHAD_UGO_double
      USE GALAHAD_USERDATA_double
      IMPLICIT NONE
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

! ------------------------- Do not change -------------------------------

!  Keep the above subroutine, argument, and function declarations for use
!  in all your fortran mex files.
!
      INTEGER * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )

      INTEGER, PARAMETER :: slen = 30
      LOGICAL :: mxIsChar, mxIsStruct
      mwSize :: mxGetString, mxIsNumeric
      mwPointer :: mxGetPr, mxCreateDoubleMatrix
      INTEGER :: mexCallMATLABWithTrap

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i
      INTEGER * 4 :: i4
      INTEGER, PARAMETER :: int4 = KIND( i4 )

      mwSize :: xl_arg, xu_arg
      mwSize :: efgh_arg, con_arg, c_arg
      mwSize :: x_arg, f_arg, g_arg, h_arg, i_arg, s_len

      mwPointer :: xl_in, xu_in, xl_pr, xu_pr, con_in
      mwPointer :: x_pr, f_pr, f_in, g_pr, g_in, h_pr, h_in, s_in, s_pr
      mwPointer input_x( 1 ), output_fgh( 4 )
      mwSize :: status
      mwSize :: m_mwsize, n_mwsize

      CHARACTER ( len = 80 ) :: char_output_unit, filename
!     CHARACTER ( len = 80 ) :: debug = REPEAT( ' ', 80 )
      CHARACTER ( len = 80 ) :: eval_fgh = REPEAT( ' ', 80 )
      LOGICAL :: opened, file_exists, initial_set = .FALSE.
      INTEGER :: iores
      CHARACTER ( len = 8 ) :: mode
      TYPE ( UGO_pointer_type ) :: UGO_pointer

!  arguments for UGO

      REAL ( KIND = wp ) :: x_l, x_u, x, f, g, h
      TYPE ( GALAHAD_userdata_type ) :: userdata
      TYPE ( UGO_data_type ), SAVE :: data
      TYPE ( UGO_control_type ), SAVE :: control
      TYPE ( UGO_inform_type ) :: inform

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_ugo requires at least 1 input argument' )

!     write(debug,"(I0)") nrhs
!     CALL mexErrMsgTxt( TRIM(debug))
      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 4 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_ugo' )
          xl_arg = 2 ; xu_arg = 3 ; efgh_arg = 4
          IF ( nrhs == 4 ) THEN
            con_arg = - 1
          ELSE IF ( nrhs == 5 ) THEN
            IF ( mxIsStruct( prhs( 5 ) ) ) THEN
              con_arg = 5
            END IF
          ELSE
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_ugo' )
          END IF
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 4 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_ugo' )
        xl_arg = 1 ; xu_arg = 2 ; efgh_arg = 3

        IF (  nrhs == 4 ) THEN
          con_arg = - 1
        ELSE IF (  nrhs == 5 ) THEN
          con_arg = 5
        ELSE
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_ugo' )
        END IF
      END IF

      x_arg = 1 ; f_arg = 2 ; g_arg = 3 ; h_arg = 4
      IF ( nlhs == 4 ) THEN
        i_arg = - 1
      ELSE IF ( nlhs == 5 ) THEN
        i_arg = 5
      ELSE IF ( nlhs > 5 ) THEN
        CALL mexErrMsgTxt( ' galahad_ugo provides at most 5 output arguments' )
      END IF

!  Initialize the internal structures for tru

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL UGO_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  if required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL UGO_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  check that UGO_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  find the name of the eval_fgh routine and ensure it exists

        i = mxGetString( prhs( efgh_arg ), eval_fgh, 80 )
        INQUIRE( file = TRIM( eval_fgh ) // '.m', exist = file_exists )
        IF ( .NOT. file_exists )                                               &
          CALL mexErrMsgTxt( TRIM( ' function evaluation file ' //             &
                              TRIM( eval_fgh ) // '.m does not exist' ) )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( con_arg > 0 ) THEN
          con_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( con_in ) )                                    &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL UGO_matlab_control_set( con_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( char_output_unit, "( I0 )" ) control%error
          filename = "output_ugo." // TRIM( char_output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_ugo." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL UGO_matlab_inform_create( plhs( i_arg ), UGO_pointer )

!  import the problem data

!  Input x_l

        xl_in = prhs( xl_arg )
        IF ( mxIsNumeric( xl_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a scalar x_l ' )
        xl_pr = mxGetPr( xl_in )
        CALL MATLAB_copy_from_ptr( xl_pr, x_l )

!  Input x_u

        xu_in = prhs( xu_arg )
        IF ( mxIsNumeric( xu_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a scalar x_u ' )
        xu_pr = mxGetPr( xu_in )
        CALL MATLAB_copy_from_ptr( xu_pr, x_u )

!  set for initial entry

        inform%status = 1
        m_mwsize = 1 ; n_mwsize = 1
        input_x( 1 ) = mxCreateDoubleMatrix( m_mwsize, n_mwsize, 0 )

!  loop to solve problem

!       CALL mexWarnMsgTxt( ' start loop' )
        DO
!         CALL mexWarnMsgTxt( ' enter solve' )
          CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data,         &
                          userdata )
!         CALL mexWarnMsgTxt( ' end solve' )
          SELECT CASE ( inform%status )

!  obtain the objective function and its first derivative

          CASE ( 3 )
!           CALL mexWarnMsgTxt( ' 4' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( x, x_pr )

!  evaluate f(x) in Matlab

            status = mexCallMATLABWithTrap( 4, output_fgh, 1, input_x,         &
                                            eval_fgh )

!  check to see that the evaluation succeeded

            s_in = output_fgh( 4 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, data%eval_status )

!  recover the function value

            IF ( data%eval_status == 0 ) THEN
              f_in = output_fgh( 1 )
              f_pr = mxGetPr( f_in )
              CALL MATLAB_copy_from_ptr( f_pr, f )

!  recover the gradient value

              g_in = output_fgh( 2 )
              g_pr = mxGetPr( g_in )
              CALL MATLAB_copy_from_ptr( g_pr, g )
            END IF

!  obtain the objective function and its derivatives

          CASE ( 4 )
!           CALL mexWarnMsgTxt( ' 4' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( x, x_pr )

!  evaluate f(x) in Matlab

            status = mexCallMATLABWithTrap( 4, output_fgh, 1, input_x,         &
                                            eval_fgh )

!  check to see that the evaluation succeeded

             s_in = output_fgh( 4 )
             s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, data%eval_status )

!  recover the function value

            IF ( data%eval_status == 0 ) THEN
              f_in = output_fgh( 1 )
              f_pr = mxGetPr( f_in )
              CALL MATLAB_copy_from_ptr( f_pr, f )

!  recover the gradient value

              g_in = output_fgh( 2 )
              g_pr = mxGetPr( g_in )
              CALL MATLAB_copy_from_ptr( g_pr, g )

!  recover the 2nd derivative value

              h_in = output_fgh( 3 )
              h_pr = mxGetPr( h_in )
              CALL MATLAB_copy_from_ptr( h_pr, h )
            END IF

!  terminal exit from loop

          CASE DEFAULT
            EXIT
          END SELECT
        END DO

!  Print details to Matlab window

!       IF ( control%error > 0 ) CLOSE( control%error )
!       IF ( control%out > 0 .AND. control%error /= control%out )              &
!         CLOSE( control%out )

        IF ( control%out > 0 ) THEN
          REWIND( control%out, err = 500 )
          DO
            READ( control%out, "( A )", end = 500 ) str
            i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
          END DO
        END IF
   500  CONTINUE

!  Output solution and the function and derivative values

        plhs( x_arg ) = MATLAB_create_real( )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( x, x_pr )

        plhs( f_arg ) = MATLAB_create_real( )
        f_pr = mxGetPr( plhs( f_arg ) )
        CALL MATLAB_copy_to_ptr( f, f_pr )

        plhs( g_arg ) = MATLAB_create_real( )
        g_pr = mxGetPr( plhs( g_arg ) )
        CALL MATLAB_copy_to_ptr( g, g_pr )

        plhs( h_arg ) = MATLAB_create_real( )
        h_pr = mxGetPr( plhs( h_arg ) )
        CALL MATLAB_copy_to_ptr( h, h_pr )

!  Record output information

        CALL UGO_matlab_inform_get( inform, UGO_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to UGO_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        CALL UGO_terminate( data, control, inform )
      END IF

!  close any opened io units

      IF ( control%error > 0 ) THEN
        INQUIRE( control%error, OPENED = opened )
        IF ( opened ) CLOSE( control%error )
      END IF

      IF ( control%out > 0 ) THEN
        INQUIRE( control%out, OPENED = opened )
        IF ( opened ) CLOSE( control%out )
      END IF
      RETURN

      END SUBROUTINE mexFunction
