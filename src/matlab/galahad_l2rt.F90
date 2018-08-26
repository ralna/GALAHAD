#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_L2RT
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an m by n matrix A, an m-vector b, and scalars p, sigma and mu, find
!  an approximate solution of the REGULARISED LEAST-L_2-NORM subproblem
!    minimize sqrt{|| A x - b ||_2^2 + mu ||x||_2^2} + 1/p sigma ||x||^p_2
!  using an iterative method. Here ||.||_2 is the Euclidean (l_2-)norm.
!  Advantage is taken of sparse A.
!
!  Simple usage -
!
!  to solve the regularised least-l_2-norm subproblem
!   [ x, obj, inform ]
!     = galahad_l2rt( A, b, p, sigma, mu, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_l2rt( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, obj, inform ]
!     = galahad_l2rt( 'existing', A, b, p, sigma, mu, control )
!
!  to remove data structures after solution
!   galahad_l2rt( 'final' )
!
!  Usual Input -
!          A: the m by n matrix A
!          b: the m-vector b
!          p: the regularisation order, p (p>=2)
!      sigma: the regularisation weight, sigma (sigma>=0)
!         mu: the shift weight, mu (sigma>=0)
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type L2RT_CONTROL as described in the
!            manual for the fortran 90 package GALAHAD_L2RT.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/l2rt.pdf
!
!  Usual Output -
!          x: the global minimizer
!        obj: the optimal value of the objective function
!
!  Optional Output -
!    control: see above. Returned values are the defaults
!     inform: a structure containing information parameters
!            The components are of the form inform.value, where
!            value is the name of the corresponding component of the
!            derived type L2RT_INFORM as described in the manual for
!            the fortran 90 package GALAHAD_L2RT.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/l2rt.pdf
!            Note that as the objective value is already available
!            the component obj from L2RT_inform is omitted.
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
      USE GALAHAD_L2RT_MATLAB_TYPES
      USE GALAHAD_L2RT_double
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
      mwSize :: a_arg, b_arg, p_arg, sigma_arg, mu_arg, con_arg
      mwSize :: x_arg, obj_arg, i_arg, s_len

      mwPointer :: a_in, b_in, c_in, p_in, sigma_in, mu_in
      mwPointer :: x_pr, obj_pr, b_pr, p_pr, sigma_pr, mu_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      LOGICAL * 4 :: true = .TRUE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( L2RT_pointer_type ) :: L2RT_pointer

!  arguments for L2RT

      REAL ( KIND = wp ) :: p, sigma, mu
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, X, U, V
      TYPE ( SMT_type ) :: A
      TYPE ( L2RT_data_type ), SAVE :: data
      TYPE ( L2RT_control_type ), SAVE :: control
      TYPE ( L2RT_inform_type ) :: inform

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_l2rt requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_l2rt' )
          a_arg = 2 ; b_arg = 3 ; p_arg = 4 ; sigma_arg = 5 ; mu_arg = 6
          con_arg = 7
          x_arg = 1 ; obj_arg = 2 ; i_arg = 3
          IF ( nrhs > con_arg )                                                &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_l2rt' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_l2rt' )
        a_arg = 1 ; b_arg = 2 ; p_arg = 3 ; sigma_arg = 4 ; mu_arg = 5
        con_arg = 6
        x_arg = 1 ; obj_arg = 2 ; i_arg = 3
        IF ( nrhs > con_arg )                                                  &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_l2rt' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_l2rt provides at most 3 output arguments' )

!  Initialize the internal structures for l2rt

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL L2RT_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          con_arg = 1
          IF ( nlhs > con_arg )                                                &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL L2RT_matlab_control_get( plhs( con_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that L2RT_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL L2RT_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_l2rt." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_l2rt." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL L2RT_matlab_inform_create( plhs( i_arg ), L2RT_pointer )

!  Import the problem data

!  Check to ensure the input for A is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, A, col_ptr, .FALSE. )
        m = A%m ; n = A%n

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vector B and workspace vectors

        ALLOCATE( B( m ), U( m ), V( n ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector b ' )
        b_pr = mxGetPr( b_in )
        CALL MATLAB_copy_from_ptr( b_pr, B, m )

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

!  Input mu

        mu_in = prhs( mu_arg )
        IF ( mxIsNumeric( mu_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must a scalar mu ' )
        mu_pr = mxGetPr( mu_in )
        CALL MATLAB_copy_from_ptr( mu_pr, mu )

!  Allocate space for the solution

        ALLOCATE( X( n ), STAT = info )

!  Solve the problem

        U( : m ) = B( : m )
        inform%status = 1

        DO                              ! Iteration to find the minimizer
          CALL L2RT_solve( m, n, p, sigma, mu, X, U, V, data, control, inform )
          SELECT CASE( inform%status )  !  Branch as a result of inform%status
          CASE( 2 )                     !  Form u <- u + A * v
            i4 = 0
            CALL MOP_Ax( 1.0_wp, A, V, 1.0_wp, U, i4, i4, i4 )
          CASE( 3 )                     !  Form v <- v + A^T * u
            i4 = 0
            CALL MOP_Ax( 1.0_wp, A, U, 1.0_wp, V, i4, i4, i4,                  &
                         transpose = true )
          CASE ( 4 )                    !  Restart
             U = B                      !  re-initialize u to b
          CASE DEFAULT                  !  Successful and error returns
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
   500   CONTINUE

!  Output solution

        i4 = 1
        plhs( x_arg ) = MATLAB_create_real( n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( X, x_pr, n )

!  Output optimal objective

        plhs( obj_arg ) = MATLAB_create_real( i4 )
        obj_pr = mxGetPr( plhs( obj_arg ) )
        CALL MATLAB_copy_to_ptr( inform%obj, obj_pr )

!  Record output information

        CALL L2RT_matlab_inform_get( inform, L2RT_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to L2RT_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( A%row ) ) DEALLOCATE( A%row, STAT = info )
        IF ( ALLOCATED( A%col ) ) DEALLOCATE( A%col, STAT = info )
        IF ( ALLOCATED( A%val ) ) DEALLOCATE( A%val, STAT = info )
        IF ( ALLOCATED( B ) ) DEALLOCATE( B, STAT = info )
        IF ( ALLOCATED( X ) ) DEALLOCATE( X, STAT = info )
        IF ( ALLOCATED( U ) ) DEALLOCATE( U, STAT = info )
        IF ( ALLOCATED( V ) ) DEALLOCATE( V, STAT = info )
        CALL L2RT_terminate( data, control, inform )
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
