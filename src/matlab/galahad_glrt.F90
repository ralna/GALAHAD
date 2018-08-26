#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_GLRT
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix H (and possibly M), an n-vector g,
!  a constant f, and scalars p and sigma, find an approximate solution
!  of the REGULARISED quadratic subproblem
!    minimize  1/p sigma ||x||_M^p + 1/2 <x, H x> + <c, x> + f
!  using an iterative method.
!  Here ||x||_M^2 = x' * M * x and M is positive definite; if M is
!  not given, M=I and ||x||_M is thus taken to be the Euclidean (l_2-)norm
!  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H.
!
!  Simple usage -
!
!  to solve the regularized quadratic subproblem
!   [ x, obj, inform ]
!     = galahad_glrt( H, c, f, p, sigma, control, M )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_glrt( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, obj, inform ]
!     = galahad_glrt( 'existing', H, c, f, p, sigma, control, M )
!
!  to remove data structures after solution
!   galahad_glrt( 'final' )
!
!  Usual Input -
!          H: the symmetric n by n matrix H
!          c: the n-vector c
!          f: the scalar f
!          p: the regularisation order, p (p>=2)
!     sigma: the regularisation weight, sigma (sigma>=0)
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type GLRT_control as described in the
!            manual for the fortran 90 package GALAHAD_GLRT.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/glrt.pdf
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
!      the derived type GLRT_inform as described in the manual for
!      the fortran 90 package GALAHAD_GLRT.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/glrt.pdf
!      Note that as the objective value is already available
!      the component obj from GLRT_inform is omitted.
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.3.1. March 5th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_GLRT_MATLAB_TYPES
      USE GALAHAD_PSLS_double
      USE GALAHAD_GLRT_double
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
      mwSize :: h_arg, c_arg, f_arg, p_arg, sigma_arg, con_arg, m_arg
      mwSize :: x_arg, obj_arg, i_arg
      mwSize :: s_len

      mwPointer :: h_in, c_in, f_in, p_in, sigma_in, g_in, m_in
      mwPointer :: x_pr, obj_pr
      mwPointer :: g_pr, f_pr, p_pr, sigma_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores
      REAL ( KIND = wp ) :: val
      CHARACTER ( len = 8 ) :: mode
      TYPE ( GLRT_pointer_type ) :: GLRT_pointer

!  arguments for GLRT and (if needed) PSLS

      REAL ( KIND = wp ) :: p, sigma, f
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, X
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R, VECTOR, H_VECTOR
      TYPE ( SMT_type ) :: H, M
      TYPE ( GLRT_data_type ), SAVE :: data
      TYPE ( GLRT_control_type ), SAVE :: control
      TYPE ( GLRT_inform_type ) :: inform

      TYPE ( PSLS_data_type ), SAVE :: PSLS_data
      TYPE ( PSLS_control_type ), SAVE :: PSLS_control
      TYPE ( PSLS_inform_type ) :: PSLS_inform

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_glrt requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_glrt' )
          h_arg = 2 ; c_arg = 3 ; f_arg = 4 ; p_arg = 5 ; sigma_arg = 6
          con_arg = 7 ; m_arg = 8
          x_arg = 1 ; obj_arg = 2 ; i_arg = 3
          IF ( nrhs > m_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_glrt' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_glrt' )
        h_arg = 1 ; c_arg = 2 ; f_arg = 3 ; p_arg = 4 ; sigma_arg = 5
        con_arg = 6 ; m_arg = 7
        x_arg = 1 ; obj_arg = 2 ; i_arg = 3
        IF ( nrhs > m_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_glrt' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_glrt provides at most 3 output arguments' )

!  Initialize the internal structures for glrt

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL GLRT_initialize( data, control, inform )

        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL GLRT_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that GLRT_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL GLRT_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_glrt." // TRIM( output_unit )
           OPEN( control%error, FILE = filename, FORM = 'FORMATTED',           &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_glrt." // TRIM( output_unit )
             OPEN( control%out, FILE = filename, FORM = 'FORMATTED',           &
                   STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL GLRT_matlab_inform_create( plhs( i_arg ), GLRT_pointer )

!  Import the problem data

!  Check to ensure the input for H is a number

        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, H, col_ptr, .TRUE. )
        n = H%n

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

!  Input p

        p_in = prhs( p_arg )
        IF ( mxIsNumeric( p_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must a scalar p ' )
        p_pr = mxGetPr( p_in )
        CALL MATLAB_copy_from_ptr( p_pr, p )

!  Input sigma

        sigma_in = prhs( sigma_arg )
        IF ( mxIsNumeric( sigma_in ) == 0 )                                    &
           CALL mexErrMsgTxt( ' There must a scalar sigma ' )
        sigma_pr = mxGetPr( sigma_in )
        CALL MATLAB_copy_from_ptr( sigma_pr, sigma )

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

!  check for error returns

          IF ( PSLS_inform%status /= 0 ) THEN
            inform%status = PSLS_inform%status
            CALL MATLAB_copy_to_ptr( inform%status,                            &
               mxGetPr(  GLRT_pointer%status ) )
            CALL mexErrMsgTxt( ' Factorization of preconditioner M failed' )
          END IF
!         CALL mexWarnMsgTxt( ' M used ' )
        END IF

!  Solve the problem

        R( : n ) = C( : n )
        inform%status = 1
        DO                              !  Iteration to find the minimizer
          CALL GLRT_solve( n, p, sigma, X, R, VECTOR, data, control, inform )
          SELECT CASE( inform%status )  ! Branch as a result of inform%status
          CASE( 2 )                     ! Form the preconditioned gradient
            CALL PSLS_solve( VECTOR, PSLS_data, PSLS_control, PSLS_inform )
          CASE ( 3 )                    ! Form the matrix-vector product
            H_VECTOR( : n ) = 0.0_wp
            DO l = 1, H%ne
              i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
              H_VECTOR( i ) = H_vECTOR( i ) + val * VECTOR( j )
              IF ( i /= j ) H_VECTOR( j ) = H_VECTOR( j ) + val * VECTOR( i )
            END DO
            VECTOR( : n ) = H_VECTOR( : n )
          CASE ( 4 )        !  Restart
            R( : n ) = C( : n )
          CASE ( 5 )                  ! Form the product with the preconditioner
            H_VECTOR( : n ) = 0.0_wp
            DO l = 1, M%ne
              i = M%row( l ) ; j = M%col( l ) ; val = M%val( l )
              H_VECTOR( i ) = H_vECTOR( i ) + val * VECTOR( j )
              IF ( i /= j ) H_VECTOR( j ) = H_VECTOR( j ) + val * VECTOR( i )
            END DO
            VECTOR( : n ) = H_VECTOR( : n )
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
        CALL MATLAB_copy_to_ptr( inform%obj_regularized, obj_pr )

!  Record output information

        CALL GLRT_matlab_inform_get( inform, GLRT_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to glrt_solve failed ' )
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
        CALL GLRT_terminate( data, control, inform )
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
