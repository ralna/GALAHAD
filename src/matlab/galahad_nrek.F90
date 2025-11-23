#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_NREK
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix H (and possibly S), an n-vector g,
!  a constant f, and scalars power and weight, find the solution of the
!  NORM-REGULARIZATION subproblem
!    minimize 0.5 * x' * H * x + c' * x + f + (weight/power) * ||x||_S^power
!  Here ||x||_S^2 = x' * S * x and S is diagonally dominant; if S is
!  not given, S=I and ||x||_S is thus taken to be the Euclidean (l_2-)norm
!  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H and S.
!
!  Simple usage -
!
!  to solve the norm-regularization subproblem in the Euclidean norm
!   [ x, inform ]
!     = galahad_nrek( H, c, power, weight, control, S )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_nrek( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, inform ]
!     = galahad_nrek( 'existing', H, c, power, weight, control, S )
!
!  to remove data structures after solution
!   galahad_nrek( 'final' )
!
!  Usual Input -
!          H: the symmetric n by n matrix H
!          c: the n-vector c
!     weight: the regularization weight (weight>0)
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type NREK_control_type as described in
!            the manual for the modern fortran package GALAHAD_NREK.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/nrek.pdf
!          S: the n by n symmetric, diagonally-dominant matrix S
!
!  Usual Output -
!          x: the global minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of the
!      derived type NREK_inform_type as described in the manual
!      for the fortran 90 package GALAHAD_NREK. The components
!      of inform.time, inform.NREK_inform and inform.SLS_inform 
!      are themselves structures, holding the
!      components of the derived types NREK_time_type, 
!      NREK_inform_type and SLS_inform_type, respectively.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/nrek.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.3.1. February 18th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_NREK_MATLAB_TYPES
      USE GALAHAD_NREK_double
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
      mwPointer :: mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, info
      INTEGER * 4 :: i4, n
      mwSize :: h_arg, c_arg, power_arg, weight_arg, con_arg, s_arg
      mwSize :: x_arg, i_arg, s_len

      mwPointer :: h_in, c_in, power_in, weight_in, g_in, s_in
      mwPointer :: x_pr, g_pr, power_pr, weight_pr

      INTEGER, PARAMETER :: history_max = 100
      CHARACTER ( len = 80 ) :: char_output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      INTEGER :: iores
      CHARACTER ( len = 8 ) :: mode
      TYPE ( NREK_pointer_type ) :: NREK_pointer
      mwPointer, ALLOCATABLE :: col_ptr( : )

!  arguments for NREK

      REAL ( KIND = wp ) :: power, weight
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, X
      TYPE ( SMT_type ) :: H, S
      TYPE ( NREK_data_type ), SAVE :: data
      TYPE ( NREK_control_type ), SAVE :: control
      TYPE ( NREK_inform_type ) :: inform

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_nrek requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_nrek' )
          h_arg = 2 ; c_arg = 3 ; power_arg = 4 ; weight_arg = 5
          con_arg = 6 ; s_arg = 7
          x_arg = 1 ; i_arg = 2
          IF ( nrhs > s_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_nrek' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_nrek' )
        h_arg = 1 ; c_arg = 2 ; power_arg = 3 ; weight_arg = 4
        con_arg = 5 ; s_arg = 6
        x_arg = 1 ; i_arg = 2
        IF ( nrhs > s_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_nrek' )
      END IF

      IF ( nlhs > 2 )                                                          &
        CALL mexErrMsgTxt( ' galahad_nrek provides at most 2 output arguments' )

!  Initialize the internal structures for nrek

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL NREK_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL NREK_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

!  Check that NREK_initialize has been called

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN
        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL NREK_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( char_output_unit, "( I0 )" ) control%error
          filename = "output_nrek." // TRIM( char_output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_nrek." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL NREK_matlab_inform_create( plhs( i_arg ), NREK_pointer )

!  Import the problem data

!  input H

        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, H, col_ptr, .TRUE. )
        n = H%n

!  input M

        IF ( nrhs >= s_arg ) THEN
          s_in = prhs( s_arg )
          IF ( mxIsNumeric( s_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a matrix S ' )
          CALL MATLAB_transfer_matrix( s_in, S, col_ptr, .TRUE. )
          IF ( S%n /= n )                                                      &
            CALL mexErrMsgTxt( ' Dimensions of H and S must agree' )
        END IF

!  Allocate space for input vector C

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )
        ALLOCATE( C( n ), STAT = info )

!  Input c

        g_in = prhs( c_arg )
        IF ( mxIsNumeric( g_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a vector c ' )
        g_pr = mxGetPr( g_in )
        CALL MATLAB_copy_from_ptr( g_pr, C, n )

!  Input power

        power_in = prhs( power_arg )
        IF ( mxIsNumeric( power_in ) == 0 )                                   &
          CALL mexErrMsgTxt( ' There must a scalar power ' )
        power_pr = mxGetPr( power_in )
        CALL MATLAB_copy_from_ptr( power_pr, power )

!  Input weight

        weight_in = prhs( weight_arg )
        IF ( mxIsNumeric( weight_in ) == 0 )                                   &
          CALL mexErrMsgTxt( ' There must a scalar weight ' )
        weight_pr = mxGetPr( weight_in )
        CALL MATLAB_copy_from_ptr( weight_pr, weight )

!  Allocate space for the solution

        ALLOCATE( X( n ), STAT = info )

!  Solve the problem

        IF ( nrhs >= s_arg ) THEN
          CALL NREK_solve( n, H, C, power, weight, X, data, control, inform,   &
                           S = S )
        ELSE
          CALL NREK_solve( n, H, C, power, weight, X, data, control, inform )
        END IF

!  Print details to Matlab window

       IF ( control%out > 0 ) THEN
         REWIND( control%out, err = 500 )
          DO
            READ( control%out, "( A )", end = 500 ) str
            i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
          END DO
        END IF
   500  CONTINUE

!  Output solution

         i4 = 1
         plhs( x_arg ) = MATLAB_create_real( n, i4 )
         x_pr = mxGetPr( plhs( x_arg ) )
         CALL MATLAB_copy_to_ptr( X, x_pr, n )

!  Record output information

         CALL NREK_matlab_inform_get( inform, NREK_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to NREK_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type, STAT = info )
        IF ( ALLOCATED( H%row ) ) DEALLOCATE( H%row, STAT = info )
        IF ( ALLOCATED( H%col ) ) DEALLOCATE( H%col, STAT = info )
        IF ( ALLOCATED( H%val ) ) DEALLOCATE( H%val, STAT = info )
        IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type, STAT = info )
        IF ( ALLOCATED( S%row ) ) DEALLOCATE( S%row, STAT = info )
        IF ( ALLOCATED( S%col ) ) DEALLOCATE( S%col, STAT = info )
        IF ( ALLOCATED( S%val ) ) DEALLOCATE( S%val, STAT = info )
        IF ( ALLOCATED( C ) ) DEALLOCATE( C, STAT = info )
        IF ( ALLOCATED( X ) ) DEALLOCATE( X, STAT = info )
        CALL NREK_terminate( data, control, inform )
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
