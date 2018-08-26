#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_GLTR
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix H (and possibly M), an n-vector g,
!  a constant f, and a scalar radius, find an approximate solution of
!  the TRUST-REGION subproblem
!    minimize 0.5 * x' * H * x + c' * x + f
!    subject to ||x||_M <= radius
!  using an iterative method.
!  Here ||x||_M^2 = x' * M * x and M is positive definite; if M is
!  not given, M=I and ||x||_M is thus taken to be the Euclidean (l_2-)norm
!  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H.
!
!  Simple usage -
!
!  to solve the trust-region subproblem in the M norm
!   [ x, obj, inform ]
!     = galahad_gltr( H, c, f, radius, control, M )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_gltr( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, obj, inform ]
!     = galahad_gltr( 'existing', H, c, f, radius, control, M )
!
!  to remove data structures after solution
!   galahad_gltr( 'final' )
!
!  Usual Input -
!          H: the symmetric n by n matrix H
!          c: the n-vector c
!          f: the scalar f
!     radius: the trust-region radius (radius>0)
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type GLTR_CONTROL as described in the
!            manual for the fortran 90 package GALAHAD_GLTR.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/gltr.pdf
!          M: the n by n symmetric, positive-definite matrix M
!
!  Usual Output -
!          x: the global minimizer
!        obj: the optimal value of the objective function
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type GLTR_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_GLTR.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/gltr.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.3.1. March 2nd 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_GLTR_MATLAB_TYPES
      USE GALAHAD_PSLS_double
      USE GALAHAD_GLTR_double
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
      mwSize :: mxGetString
      mwSize :: mxIsNumeric
      mwPointer :: mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, j, l, info
      INTEGER * 4 :: i4, n
      mwSize :: h_arg, c_arg, f_arg, radius_arg, con_arg, m_arg
      mwSize :: x_arg, obj_arg, i_arg
      mwSize :: s_len

      mwPointer :: h_in, c_in, f_in, radius_in, g_in, m_in
      mwPointer :: x_pr, obj_pr
      mwPointer :: g_pr, f_pr, radius_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores
      REAL ( KIND = wp ) :: val
      CHARACTER ( len = 8 ) :: mode
      TYPE ( GLTR_pointer_type ) :: GLTR_pointer

!  arguments for GLTR and (if needed) PSLS

      REAL ( KIND = wp ) :: radius, f
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, X
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R, VECTOR, H_VECTOR
      TYPE ( SMT_type ) :: H, M
      TYPE ( GLTR_data_type ), SAVE :: data
      TYPE ( GLTR_control_type ), SAVE :: control
      TYPE ( GLTR_info_type ) :: inform

      TYPE ( PSLS_data_type ), SAVE :: PSLS_data
      TYPE ( PSLS_control_type ), SAVE :: PSLS_control
      TYPE ( PSLS_inform_type ) :: PSLS_inform

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_gltr requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_gltr' )
          h_arg = 2 ; c_arg = 3 ; f_arg = 4 ; radius_arg = 5
          con_arg = 6 ; m_arg = 7
          x_arg = 1 ; obj_arg = 2 ; i_arg = 3
          IF ( nrhs > m_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_gltr' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_gltr' )
        h_arg = 1 ; c_arg = 2 ; f_arg = 3 ; radius_arg = 4
        con_arg = 5 ; m_arg = 6
        x_arg = 1 ; obj_arg = 2 ; i_arg = 3
        IF ( nrhs > m_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_gltr' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_gltr provides at most 3 output arguments' )

!  Initialize the internal structures for gltr

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL GLTR_initialize( data, control, inform )

        IF ( TRIM( mode ) == 'initial' )  THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL GLTR_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that GLTR_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL GLTR_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_gltr." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL GLTR_matlab_inform_create( plhs( i_arg ), GLTR_pointer )

!  Import the problem data

!  Check to ensure the input for H is a number

        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, H, col_ptr, .TRUE. )
        n = H%n

        IF ( n <= 0 ) THEN
           CALL mexErrMsgTxt( ' n <= 0' )
        END IF


!  Check to ensure the input for M is a number

        IF ( nrhs >= m_arg ) THEN
          m_in = prhs( m_arg )
          IF ( mxIsNumeric( m_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a matrix M ' )
          CALL MATLAB_transfer_matrix( m_in, M, col_ptr, .TRUE. )
          IF ( M%n /= n )                                                      &
            CALL mexErrMsgTxt( ' Column dimensions of H and M must agree' )
          control%unitm = .FALSE.
        END IF

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vector C and workspace vectors

        ALLOCATE( C( n ), R( n ), VECTOR( n ), H_VECTOR( n ), STAT = info )

!  Input C

        g_in = prhs( c_arg )
        IF ( mxIsNumeric( g_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a vector c ' )
        g_pr = mxGetPr( g_in )
        CALL MATLAB_copy_from_ptr( g_pr, C, n )

!  Input f

        f_in = prhs( f_arg )
        IF ( mxIsNumeric( f_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must a scalar f ' )
        f_pr = mxGetPr( f_in )
        CALL MATLAB_copy_from_ptr( f_pr, f )
        control%f_0 = f

!  Input radius

        radius_in = prhs( radius_arg )
        IF ( mxIsNumeric( radius_in ) == 0 )                                   &
           CALL mexErrMsgTxt( ' There must a scalar radius ' )
        radius_pr = mxGetPr( radius_in )
        CALL MATLAB_copy_from_ptr( radius_pr, radius )

!  Allocate space for the solution

        ALLOCATE( X( n ), STAT = info )

!  If needed, factorize the preconditioner

        IF ( .NOT. control%unitm ) THEN
          IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' )          &
            CALL PSLS_initialize( PSLS_data, PSLS_control, PSLS_inform )
          PSLS_control%definite_linear_solver = 'sils'
          PSLS_control%preconditioner = 5
          CALL PSLS_form_and_factorize( M, PSLS_data, PSLS_control,            &
                                        PSLS_inform )
          IF ( PSLS_inform%status /= 0 ) THEN
            inform%status = PSLS_inform%status
            CALL MATLAB_copy_to_ptr( inform%status,                            &
               mxGetPr( GLTR_pointer%status ) )
            CALL mexErrMsgTxt( ' Factorization of preconditioner M failed' )
          END IF
!         CALL mexWarnMsgTxt( ' M used ' )
        END IF

!  Solve the problem

        R( : n ) = C( : n )
        inform%status = 1
        DO                              !  Iteration to find the minimizer
          CALL GLTR_solve( n, radius, f, X, R, VECTOR, data, control, inform )
          SELECT CASE( inform%status )  ! Branch as a result of inform%status
          CASE( 2, 6 )                  ! Form the preconditioned gradient
            CALL PSLS_solve( VECTOR, PSLS_data, PSLS_control, PSLS_inform )
          CASE ( 3, 7 )                 ! Form the matrix-vector product
            H_VECTOR( : n ) = 0.0_wp
            DO l = 1, H%ne
              i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
              H_VECTOR( i ) = H_VECTOR( i ) + val * VECTOR( j )
              IF ( i /= j ) H_VECTOR( j ) = H_VECTOR( j ) + val * VECTOR( i )
            END DO
            VECTOR( : n ) = H_VECTOR( : n )
          CASE ( 5 )        !  Restart
            R( : n ) = C( : n )
          CASE DEFAULT      !  Successful and error returns
            EXIT
          END SELECT
        END DO

!  Print details to Matlab window

        IF ( control%out > 0 ) THEN
          REWIND( control%out, err = 500 )
          DO
            READ( control%out, "( A )", end = 500 ) str
            i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
          END DO
        END IF
    500 CONTINUE

!  Output solution

        i4 = 1
        plhs( x_arg ) = MATLAB_create_real( n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( X, x_pr, n )

!  Output optimal objective

        plhs( obj_arg ) = MATLAB_create_real( i4 )
        obj_pr = mxGetPr( plhs( obj_arg ) )
        CALL MATLAB_copy_to_ptr( f, obj_pr )

!  Record output information

        CALL GLTR_matlab_inform_get( inform, GLTR_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to gltr_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( H%row ) ) DEALLOCATE( H%row, STAT = info )
        IF ( ALLOCATED( H%col ) ) DEALLOCATE( H%col, STAT = info )
        IF ( ALLOCATED( H%val ) ) DEALLOCATE( H%val, STAT = info )
        IF ( ALLOCATED( M%row ) ) DEALLOCATE( M%row, STAT = info )
        IF ( ALLOCATED( M%col ) ) DEALLOCATE( M%col, STAT = info )
        IF ( ALLOCATED( M%val ) ) DEALLOCATE( M%val, STAT = info )
        IF ( ALLOCATED( C ) ) DEALLOCATE( C, STAT = info )
        IF ( ALLOCATED( X ) ) DEALLOCATE( X, STAT = info )
        IF ( ALLOCATED( R ) ) DEALLOCATE( R, STAT = info )
        IF ( ALLOCATED( VECTOR ) ) DEALLOCATE( VECTOR, STAT = info )
        IF ( ALLOCATED( H_VECTOR ) ) DEALLOCATE( H_VECTOR, STAT = info )
        CALL GLTR_terminate( data, control, inform )
        IF ( .NOT. control%unitm )                                &
          CALL PSLS_terminate( PSLS_data, PSLS_control, PSLS_inform )
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
