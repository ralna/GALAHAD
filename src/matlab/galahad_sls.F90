
#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.3 - 09/11/2020 AT 14:20 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_SLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix A and an n-vector b or an n by r
!  matrix B, solve the system A x = b or the system AX=B. The matrix
!  A need not be definite. Advantage is taken of sparse A. Options
!  are provided to factorize a matrix A without solving the system,
!  and to solve systems using previously-determined factors.
!
!  Simple usage -
!
!  to solve a system Ax=b or AX=B
!   [ x, inform ] = galahad_sls( A, b, control, solver )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to factorization
!   [ control ]
!     = galahad_sls( 'initial', solver )
!
!  to factorize A
!   [ inform ] = galahad_sls( 'factor', A, control )
!
!  to solve Ax=b or AX=B using existing factors
!   [ x, inform ] = galahad_sls( 'solve', b )
!
!  to remove data structures after solution
!   [ inform ] = galahad_sls( 'final' )
!
!  Usual Input -
!    A: the symmetric matrix A
!    b: a column vector b or matrix of right-hand sides B
!
!  Optional Input -
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type SLS_control_type as described in the
!      manual for the fortran 90 package GALAHAD_SLS.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sls.pdf
!    solver, the name of the desired linear solver. Possible values are:
!        'sils'
!        'ma27'
!        'ma57'
!        'ma77'
!        'ma86'
!        'ma87'
!        'ma97'
!        'ssids'
!        'pardiso'
!        'wsmp'
!        'potr'
!        'sytr'
!        'pbtr'
!      The default is 'sils'. Not all options will be available. For more
!      details, see: http://galahad.rl.ac.uk/galahad-www/doc/sls.pdf
!
!  Usual Output -
!   x: the vector of solutions to Ax=b or matrix of solutions to AX=B
!
!  Optional Output -
!    control: see above. Returned values are the defaults
!    inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type SLS_inform_type as described
!      in the manual for the fortran 90 package GALAHAD_SLS.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sls.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3 November 9th 2020

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_SPACE_double
      USE GALAHAD_SLS_double
      USE GALAHAD_SMT_double
      IMPLICIT NONE
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

! ------------------------- Do not change -------------------------------

!  keep the above subroutine, argument, and function declarations for use
!  in all your fortran mex files.
!
      INTEGER * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )

      INTEGER, PARAMETER :: slen = 30
      LOGICAL :: mxIsChar, mxIsStruct
      mwSize :: mxGetString, mxIsNumeric
      mwSize :: mxGetM, mxGetN
      mwPointer :: mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, info
      INTEGER * 4 :: n, nb, status, alloc_status
      mwSize :: s_len
      mwSize :: a_arg, c_arg, s_arg, b_arg, x_arg, i_arg

      mwPointer :: a_in, b_in, c_in, rhs_in, x_pr

      INTEGER, PARAMETER :: history_max = 100
      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE., factorized = .FALSE.
      INTEGER :: iores
      CHARACTER ( len = 7 ) :: mode, solver
      TYPE ( SLS_pointer_type ) :: SLS_pointer
      mwPointer, ALLOCATABLE :: col_ptr( : )
!     CHARACTER ( len = 80 ) :: message

!  arguments for SLS

      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: X2
      TYPE ( SMT_type ), SAVE :: A
      TYPE ( SLS_data_type ), SAVE :: data
      TYPE ( SLS_control_type ), SAVE :: control
      TYPE ( SLS_inform_type ), SAVE :: inform

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_sls requires at least 1 input argument' )

!  sophisticated entries

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        info = mxGetString( prhs( 1 ), mode, 7 )

!  initial entry

        IF ( TRIM( mode ) == 'initial' ) THEN
          s_arg = 2
          IF ( nrhs > s_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to sls' )

!  check for a non-default solver

          IF ( nrhs == s_arg ) THEN
            IF ( .NOT. mxIsChar( prhs( s_arg ) ) )                             &
              CALL mexErrMsgTxt( ' second argument must be a string' )

!  interpret the secondt argument

            i = mxGetString( prhs( s_arg ), solver, 7 )

!  otherwise provide a default

          ELSE
            solver = 'sils   '
          END IF

!  final entry

        ELSE IF ( TRIM( mode ) == 'final' ) THEN
          i_arg = 1
        ELSE
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to sls' )

!  factor entry

          IF ( TRIM( mode ) == 'factor' ) THEN
            a_arg = 2
            c_arg = 3
            i_arg = 1

!  solve entry

          ELSE IF ( TRIM( mode ) == 'solve' ) THEN
            b_arg = 2
            c_arg = 3
            x_arg = 1
            i_arg = 2

!  other entry

          ELSE
            a_arg = 2
            b_arg = 3
            c_arg = 4
            x_arg = 1
            i_arg = 2
          END IF
          IF ( nrhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to sls' )
        END IF

! simple entry

      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to sls' )
        a_arg = 1
        b_arg = 2
        c_arg = 3
        s_arg = 4
        x_arg = 1
        i_arg = 2
        IF ( nrhs > s_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to sls' )

!  check for a non-default solver

        IF ( nrhs == s_arg ) THEN
          IF ( .NOT. mxIsChar( prhs( 4 ) ) )                                   &
            CALL mexErrMsgTxt( ' fourth argument must be a string' )

!  interpret the secondt argument

          i = mxGetString( prhs( 4 ), solver, 7 )

!  otherwise provide a default

        ELSE
          solver = 'sils   '
        END IF
      END IF

!  initialize the internal structures for sls

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        CALL SLS_initialize( TRIM( solver ), data, control, inform )
        initial_set = .TRUE.

        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL SLS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

!  factorize entry

      IF ( TRIM( mode ) == 'factor' .OR. TRIM( mode ) == 'all' ) THEN

!  check that initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  if the third argument is present, extract the input control data

        s_len = slen
        IF ( nrhs == c_arg ) THEN
          c_in = prhs( c_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' last input argument must be a structure' )
          CALL SLS_matlab_control_set( c_in, control, s_len )
        END IF

!  open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_sls." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_sls." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

! `l input A

!       WRITE( message, "( ' input A' )" )
!       i4 = mexPrintf( TRIM( message ) // achar( 10 ) )
        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, A, col_ptr, .TRUE. )

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  form and factorize

        CALL SLS_analyse( A, data, control, inform )
        IF ( inform%status < 0 ) THEN
          CALL mexWarnMsgTxt( ' Call to SLS_analyse failed ' )
          WRITE( control%out, "( ' Error return from SLS_analyse, status = ',  &
         &   I0 )" ) inform%status
!         WRITE( control%out, "( ' ma97_info\%flag = ', I0 )" )                &
!           inform%ma97_info%flag
          GO TO 400
        END IF

        CALL SLS_factorize( A, data, control, inform )
        IF ( inform%status < 0 ) THEN
          CALL mexWarnMsgTxt( ' Call to SLS_factorize failed ' )
          WRITE( control%out, "( ' Error return from SLS_factorize, status = ',&
         &   I0 )" ) inform%status
          GO TO 400
        END IF
        factorized = .TRUE.
      END IF

!  solve entry

      IF ( TRIM( mode ) == 'solve' .OR. TRIM( mode ) == 'all' ) THEN

!  check that initialize has been called

        IF ( .NOT. factorized )                                                &
          CALL mexErrMsgTxt( ' "factorize" must be called first' )

        IF ( .NOT. ( ALLOCATED( A%row ) .AND. ALLOCATED( A%col ) .AND.         &
                     ALLOCATED( A%val ) ) )                                    &
          CALL mexErrMsgTxt( ' There must be existing factors ' )

!  check to ensure the input is a number

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a right-hand-side b ' )

 ! allocate space for the right-hand side and solution

        n = INT( mxGetM( b_in ), KIND = KIND( n ) )
        IF ( A%n /= n )                                                        &
          CALL mexErrMsgTxt( ' A and b/B must have compatible dimensions ' )

        nb = INT( mxGetN( b_in ), KIND = KIND( n ) )
        rhs_in = mxGetPr( b_in )

!  one right-hand side

        IF ( nb == 1 ) THEN
          CALL SPACE_resize_array( n, X, status, alloc_status )
          IF ( status /= 0 ) CALL mexErrMsgTxt( ' allocate error X' )
          CALL MATLAB_copy_from_ptr( rhs_in, X, n )
          CALL SLS_SOLVE( A, X, data, control, inform )

!  multiple right-hand sides

        ELSE
          CALL SPACE_resize_array( n, nb, X2, status, alloc_status )
          IF ( status /= 0 ) CALL mexErrMsgTxt( ' allocate error X2' )
          CALL MATLAB_copy_from_ptr( rhs_in, X2, n, nb )
          CALL SLS_SOLVE( A, X2, data, control, inform )
        END IF
        IF ( inform%status < 0 ) THEN
          CALL mexWarnMsgTxt( ' Call to SLS_solve failed ' )
          WRITE( control%out, "( ' Error return from SLS_solve, status = ',   &
         &   I0 )" ) inform%status
        END IF

!  output solution

        plhs( x_arg ) = MATLAB_create_real( n, nb )
        x_pr = mxGetPr( plhs( x_arg ) )
        IF ( nb == 1 ) THEN
          CALL MATLAB_copy_to_ptr( X, x_pr, n )
        ELSE
          CALL MATLAB_copy_to_ptr( X2, x_pr, n, nb )
        END IF
      END IF

!  final entry

     IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN

!  deallocate workspace

        CALL SPACE_dealloc_array( A%row, status, alloc_status )
        CALL SPACE_dealloc_array( A%col, status, alloc_status )
        CALL SPACE_dealloc_array( A%val, status, alloc_status )
        CALL SPACE_dealloc_array( X, status, alloc_status )
        CALL SPACE_dealloc_array( X2, status, alloc_status )
        CALL SLS_terminate( data, control, inform )
      END IF

!  print details to Matlab window

  400 CONTINUE
      IF ( control%out > 0 ) THEN
        REWIND( control%out, err = 500 )
        DO
          READ( control%out, "( A )", end = 500 ) str
          i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
        END DO
       END IF
  500 CONTINUE

!  create inform output structure

      CALL SLS_matlab_inform_create( plhs( i_arg ), SLS_pointer )

!  Record output information

      CALL SLS_matlab_inform_get( inform, SLS_pointer )

!  close any opened io units

     IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( control%error > 0 ) THEN
          INQUIRE( control%error, OPENED = opened )
          IF ( opened ) CLOSE( control%error )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( opened ) CLOSE( control%out )
        END IF
      END IF

!  Check for errors

!      WRITE( message, * ) 'here'
!      i = MEXPRINTF( TRIM( message ) // char( 13 ) )

      RETURN

      END SUBROUTINE mexFunction
