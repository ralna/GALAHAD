
#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_SBLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a BLOCK, REAL SYMMETRIC MATRIX
!
!         ( H  A^T ),
!         ( A  - C )
!
!  this package constructs a variety of PRECONDITIONERS of the form
!
!     K = ( G  A^T ).
!         ( A  - C )
!
!  Here, the leading-block matrix G is a suitably-chosen approximation
!  to H; it may either be prescribed EXPLICITLY, in which case a symmetric
!  indefinite factorization of K will be formed using the GALAHAD package
!  SLS, or IMPLICITLY by requiring certain sub-blocks of G be zero. In the
!  latter case, a factorization of K will be obtained implicitly (and more
!  efficiently) without recourse to SLS.
!
!  Once the preconditioner has been constructed, solutions to the
!  preconditioning system
!
!       ( G  A^T ) ( x ) = ( b )
!       ( A  - C ) ( y )   ( d )
!
!  may be obtained by the package. Full advantage is taken of any zero
!  coefficients in the matrices H, A and C.
!
!  Simple usage -
!
!  to form and factorize the matrix K
!   [ inform ]
!     = galahad_sbls( 'form_and_factorize', H, A, C, control )
!
!  to solve the preconditioning system after factorizing K
!
!   [ x, y, inform ]
!     = galahad_sbls( 'solve', b, d, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_sbls( 'initial' )
!
!  to remove data structures after solution
!  [ inform ]
!    = galahad_sbls( 'final' )
!
!  Usual Input (form-and-factorize) -
!          H: the real symmetric n by n matrix H
!          A: the real m by n matrix A
!          C: the real symmetric m by m matrix C
!  or (solve) -
!          b: the real m-vector b
!          d: the real n-vector d
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type SBLS_control_type as described in
!            the manual for the fortran 90 package GALAHAD_SBLS.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/sbls.pdf
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of the
!      derived type SBLS_inform_type as described in the manual
!      for the fortran 90 package GALAHAD_SBLS. The components
!      of inform.SLS_inform and inform.ULS_inform are themselves
!      structures, holding the components of the derived types
!      SLS_inform_type and ULS_inform_type, respectively.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sbls.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4 February 12th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_SBLS_double
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
      INTEGER * 4 :: m, n, i4
      mwSize :: s_len
      mwSize :: h_arg, a_arg, c_arg, b_arg, d_arg, con_arg
      mwSize :: x_arg, y_arg, i_arg

      mwPointer :: h_in, a_in, c_in, b_in, d_in
      mwPointer :: x_pr, y_pr, b_pr, d_pr

      INTEGER, PARAMETER :: history_max = 100
      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores
      CHARACTER ( len = 18 ) :: mode
      TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      mwPointer, ALLOCATABLE :: col_ptr( : )
!     CHARACTER ( len = 80 ) :: message

!  arguments for SBLS

      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
      TYPE ( SMT_type ) :: H
      TYPE ( SMT_type ), SAVE :: A, C
      TYPE ( SBLS_data_type ), SAVE :: data
      TYPE ( SBLS_control_type ), SAVE :: control
      TYPE ( SBLS_inform_type ), SAVE :: inform

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_sbls requires at least 1 input argument' )

      IF ( .NOT. mxIsChar( prhs( 1 ) ) )                                       &
        CALL mexErrMsgTxt( ' first argument must be a string' )

!  interpret the first argument

      i = mxGetString( prhs( 1 ), mode, 17 )

!  initial entry

      IF ( TRIM( mode ) == 'initial' ) THEN

        c_arg = 1
        IF ( nlhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' too many output arguments required' )

!  Initialize the internal structures for sbls

        CALL SBLS_initialize( data, control, inform )

!  If required, return the default control parameters

        IF ( nlhs > 0 )                                                        &
          CALL SBLS_matlab_control_get( plhs( c_arg ), control )
        RETURN

!  form_and_factorize entry

      ELSE IF ( TRIM( mode ) == 'form_and_factorize' ) THEN
        h_arg = 2 ; a_arg = 3 ; c_arg = 4 ; con_arg = 5
        IF ( nrhs > con_arg )                                                  &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_sbls' )
        i_arg = 1
        IF ( nlhs > i_arg )                                                    &
          CALL mexErrMsgTxt( ' too many output arguments required' )

!  Initialize the internal structures for sbls

        initial_set = .TRUE.
        CALL SBLS_initialize( data, control, inform )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL SBLS_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_sbls." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_sbls." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL SBLS_matlab_inform_create( plhs( i_arg ), SBLS_pointer )

!  Import the problem data

!  input H

!       WRITE( message, "( ' input H' )" )
!       i4 = mexPrintf( TRIM( message ) // achar( 10 ) )
        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, H, col_ptr, .TRUE. )

        n = H%n

!  input A

!       WRITE( message, "( ' input A' )" )
!       i4 = mexPrintf( TRIM( message ) // achar( 10 ) )

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, A, col_ptr, .FALSE. )
        IF ( A%n /= n )                                                        &
          CALL mexErrMsgTxt( ' Column dimensions of H and A must agree' )
        m = A%m

!  input C

!       WRITE( message, "( ' input C' )" )
!       i4 = mexPrintf( TRIM( message ) // achar( 10 ) )

        c_in = prhs( c_arg )
        IF ( mxIsNumeric( c_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix M ' )
        CALL MATLAB_transfer_matrix( c_in, C, col_ptr, .TRUE. )
        IF ( C%n /= m )                                                        &
          CALL mexErrMsgTxt( ' Dimensions of A and C must agree' )

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  form and factorize

        CALL SBLS_form_and_factorize( n, m, H, A, C, data, control, inform )

!  solve entry

      ELSE IF ( TRIM( mode ) == 'solve' ) THEN
        IF ( control%error > 0 ) REWIND( control%error )
        IF ( control%out > 0 ) REWIND( control%out )
         b_arg = 2 ; d_arg = 3 ; con_arg = 4
         IF ( nrhs > con_arg )                                                 &
           CALL mexErrMsgTxt( ' Too many input arguments to galahad_sbls' )
         x_arg = 1 ; y_arg = 2 ; i_arg = 3
         IF ( nlhs > i_arg )                                                   &
           CALL mexErrMsgTxt( ' too many output arguments required' )

!  Check that SBLS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "form_and_factorize" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL SBLS_matlab_control_set( c_in, control, s_len )
        END IF

!  Create inform output structure

        CALL SBLS_matlab_inform_create( plhs( i_arg ), SBLS_pointer )

!  Allocate space for input vector (b,d) in SOL

        m = A%m ; n = A%n
        ALLOCATE( SOL( n + m ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a vector b ' )
        b_pr = mxGetPr( b_in )
!       CALL mexWarnMsgTxt( ' here' )
        CALL MATLAB_copy_from_ptr( b_pr, SOL( 1 : n ), n )

!  Input d

        d_in = prhs( d_arg )
        IF ( mxIsNumeric( d_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a vector d ' )
        d_pr = mxGetPr( d_in )
        CALL MATLAB_copy_from_ptr( d_pr, SOL( n + 1 : n + m ), m )

!  Solve the system

        CALL SBLS_solve( n, m, A, C, data, control, inform, SOL )

!  Output solution

         i4 = 1
         plhs( x_arg ) = MATLAB_create_real( n, i4 )
         x_pr = mxGetPr( plhs( x_arg ) )
         CALL MATLAB_copy_to_ptr( SOL( : n ), x_pr, n )
         plhs( y_arg ) = MATLAB_create_real( m, i4 )
         y_pr = mxGetPr( plhs( y_arg ) )
         CALL MATLAB_copy_to_ptr( SOL( n + 1 : n + m ), y_pr, m )

!  final entry

      ELSE IF ( TRIM( mode ) == 'final' ) THEN

         i_arg = 1

!  Create inform output structure

        CALL SBLS_matlab_inform_create( plhs( i_arg ), SBLS_pointer )

        IF ( ALLOCATED( H%row ) ) DEALLOCATE( H%row, STAT = info )
        IF ( ALLOCATED( H%col ) ) DEALLOCATE( H%col, STAT = info )
        IF ( ALLOCATED( H%val ) ) DEALLOCATE( H%val, STAT = info )
        IF ( ALLOCATED( A%row ) ) DEALLOCATE( A%row, STAT = info )
        IF ( ALLOCATED( A%col ) ) DEALLOCATE( A%col, STAT = info )
        IF ( ALLOCATED( A%val ) ) DEALLOCATE( A%val, STAT = info )
        IF ( ALLOCATED( C%row ) ) DEALLOCATE( C%row, STAT = info )
        IF ( ALLOCATED( C%col ) ) DEALLOCATE( C%col, STAT = info )
        IF ( ALLOCATED( C%val ) ) DEALLOCATE( C%val, STAT = info )
        IF ( ALLOCATED( SOL ) ) DEALLOCATE( SOL, STAT = info )
        CALL SBLS_terminate( data, control, inform )

!  close any opened io units

        IF ( control%error > 0 ) THEN
          INQUIRE( control%error, OPENED = opened )
          IF ( opened ) CLOSE( control%error )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( opened ) CLOSE( control%out )
        END IF

!  unknown entry

      ELSE
        CALL mexErrMsgTxt( ' Unrecognised first input string ' )
      END IF

!  Print details to Matlab window

      IF ( control%out > 0 ) THEN
        REWIND( control%out, err = 500 )
        DO
          READ( control%out, "( A )", end = 500 ) str
          i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
        END DO
       END IF
  500 CONTINUE

!  Record output information

      CALL SBLS_matlab_inform_get( inform, SBLS_pointer )

!  Check for errors

      IF ( inform%status < 0 )                                                &
         CALL mexErrMsgTxt( ' Call to SBLS_solve failed ' )

!      WRITE( message, * ) 'here'
!      i = MEXPRINTF( TRIM( message ) // char( 13 ) )

      RETURN

      END SUBROUTINE mexFunction
