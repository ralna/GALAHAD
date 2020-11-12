#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_LLST
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an m by n matrix A, an m-vector b, a scalar radius, and possibly
!  a symmetric, diagonally dominant n by n matrix S, find the minimum
!  S-norm solution of the LEAST-SQUARES TRUST-REGION subproblem
!    minimize || A x - b ||_2
!    subject to ||x||_S <= radius
!  Here ||x||_S^2 = x' * S * x; if S is not given, S=I and ||x||_S is
!  thus taken to be the Euclidean (l_2-)norm sqrt(x' * x).
!  Advantage is taken of sparse A and S.
!
!  Simple usage -
!
!  to solve the least-squares trust-region subproblem in the Euclidean norm
!   [ x, obj, inform ]
!     = galahad_llst( A, b, radius, control, S )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_llst( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, obj, inform ]
!     = galahad_llst( 'existing', A, b, radius, control, S )
!
!  to remove data structures after solution
!   galahad_llst( 'final' )
!
!  Usual Input -
!          A: the m by n matrix A
!          b: the m-vector b
!     radius: the trust-region radius (radius>0)
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type LLST_CONTROL as described in the
!            manual for the fortran 90 package GALAHAD_LLST.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/llst.pdf
!          S: the n by n symmetric, diagonally-dominant matrix S
!
!  Usual Output -
!          x: the minimizer of least S-norm
!        obj: the optimal value of the objective function ||Ax-b||_2
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!           The components are of the form inform.value, where
!           value is the name of the corresponding component of the
!           derived type LLST_INFORM as described in the manual for
!           the fortran 90 package GALAHAD_LLST.
!           See: http://galahad.rl.ac.uk/galahad-www/doc/llst.pdf
!           Note that as the objective value is already available
!           the component r_norm from LLST_inform is omitted.
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.6. March 1st 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_LLST_MATLAB_TYPES
      USE GALAHAD_LLST_double
      USE GALAHAD_MOP_double
      USE GALAHAD_SMT_double
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
      INTEGER * 4 :: i4, m, n
      mwSize :: a_arg, b_arg, radius_arg, con_arg, s_arg
      mwSize :: x_arg, obj_arg, i_arg, s_len

      mwPointer :: a_in, b_in, c_in, radius_in, s_in
      mwPointer :: x_pr, obj_pr, b_pr, radius_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( LLST_pointer_type ) :: LLST_pointer

!  arguments for LLST

      REAL ( KIND = wp ) :: radius
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, X
      TYPE ( SMT_type ) :: A, S
      TYPE ( LLST_data_type ), SAVE :: data
      TYPE ( LLST_control_type ), SAVE :: control
      TYPE ( LLST_inform_type ) :: inform

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_llst requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_llst' )
          a_arg = 2 ; b_arg = 3 ; radius_arg = 4 ; con_arg = 5 ; s_arg = 6
          x_arg = 1 ; obj_arg = 2 ; i_arg = 3
          IF ( nrhs > s_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_llst' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_llst' )
        a_arg = 1 ; b_arg = 2 ; radius_arg = 3 ; con_arg = 4 ; s_arg = 5
        x_arg = 1 ; obj_arg = 2 ; i_arg = 3
        IF ( nrhs > s_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_llst' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_llst provides at most 3 output arguments' )

!  Initialize the internal structures for llst

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL LLST_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          con_arg = 1
          IF ( nlhs > con_arg )                                                &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL LLST_matlab_control_get( plhs( con_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that LLST_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL LLST_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_llst." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_llst." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL LLST_matlab_inform_create( plhs( i_arg ), LLST_pointer )

!  Import the problem data

!  Check to ensure the input for A is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, A, col_ptr, .FALSE. )
        m = A%m ; n = A%n

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vector B

        ALLOCATE( B( m ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector b ' )
        b_pr = mxGetPr( b_in )
        CALL MATLAB_copy_from_ptr( b_pr, B, m )

!  Input radius

        radius_in = prhs( radius_arg )
        IF ( mxIsNumeric( radius_in ) == 0 )                                   &
           CALL mexErrMsgTxt( ' There must a scalar radius ' )
        radius_pr = mxGetPr( radius_in )
        CALL MATLAB_copy_from_ptr( radius_pr, radius )

!  input S

        IF ( nrhs >= s_arg ) THEN
          s_in = prhs( s_arg )
          IF ( mxIsNumeric( s_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a matrix S ' )
          CALL MATLAB_transfer_matrix( s_in, S, col_ptr, .TRUE. )
          IF ( S%n /= n )                                                      &
            CALL mexErrMsgTxt( ' Column dimensions of A and S must agree' )
        END IF

!  Allocate space for the solution

        ALLOCATE( X( n ), STAT = info )

!  Solve the problem

        IF ( nrhs >= s_arg ) THEN
          CALL LLST_solve( m, n, radius, A, B, X, data, control, inform, S = S )
        ELSE
          CALL LLST_solve( m, n, radius, A, B, X, data, control, inform )
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

!  Output optimal objective

        plhs( obj_arg ) = MATLAB_create_real( i4 )
        obj_pr = mxGetPr( plhs( obj_arg ) )
        CALL MATLAB_copy_to_ptr( inform%r_norm, obj_pr )

!  Record output information

        CALL LLST_matlab_inform_get( inform, LLST_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to LLST_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( A%row ) ) DEALLOCATE( A%row, STAT = info )
        IF ( ALLOCATED( A%col ) ) DEALLOCATE( A%col, STAT = info )
        IF ( ALLOCATED( A%val ) ) DEALLOCATE( A%val, STAT = info )
        IF ( ALLOCATED( S%row ) ) DEALLOCATE( S%row, STAT = info )
        IF ( ALLOCATED( S%col ) ) DEALLOCATE( S%col, STAT = info )
        IF ( ALLOCATED( S%val ) ) DEALLOCATE( S%val, STAT = info )
        IF ( ALLOCATED( B ) ) DEALLOCATE( B, STAT = info )
        IF ( ALLOCATED( X ) ) DEALLOCATE( X, STAT = info )
        CALL LLST_terminate( data, control, inform )
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
