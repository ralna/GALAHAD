#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_LSTR
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an m by n matrix A, an m-vector b, and a scalar radius, find
!  an approximate solution of the LEAST-SQUARES TRUST-REGION subproblem
!    minimize || A x - b ||_2
!    subject to ||x||_2 <= radius
!  using an iterative method. Here ||.||_2 is the Euclidean (l_2-)norm.
!  Advantage is taken of sparse A.
!
!  Simple usage -
!
!  to solve the least-squares trust-region subproblem in the Euclidean norm
!   [ x, obj, inform ]
!     = galahad_lstr( A, b, radius, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_lstr( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, obj, inform ]
!     = galahad_lstr( 'existing', A, b, radius, control )
!
!  to remove data structures after solution
!   galahad_lstr( 'final' )
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
!            the derived type LSTR_CONTROL as described in the
!            manual for the fortran 90 package GALAHAD_LSTR.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/lstr.pdf
!
!  Usual Output -
!          x: the global minimizer
!        obj: the optimal value of the objective function ||Ax-b||_2
!
!  Optional Output -
!    control: see above. Returned values are the defaults
!     inform: a structure containing information parameters
!            The components are of the form inform.value, where
!            value is the name of the corresponding component of the
!            derived type LSTR_INFORM as described in the manual for
!            the fortran 90 package GALAHAD_LSTR.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/lstr.pdf
!            Note that as the objective value is already available
!            the component r_norm from LSTR_inform is omitted.
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
      USE GALAHAD_LSTR_MATLAB_TYPES
      USE GALAHAD_LSTR_double
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
      mwSize :: a_arg, b_arg, radius_arg, con_arg
      mwSize :: x_arg, obj_arg, i_arg, s_len

      mwPointer :: a_in, b_in, c_in, radius_in
      mwPointer :: x_pr, obj_pr, b_pr, radius_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      LOGICAL * 4 :: true = .TRUE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( LSTR_pointer_type ) :: LSTR_pointer

!  arguments for LSTR

      REAL ( KIND = wp ) :: radius
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, X, U, V
      TYPE ( SMT_type ) :: A
      TYPE ( LSTR_data_type ), SAVE :: data
      TYPE ( LSTR_control_type ), SAVE :: control
      TYPE ( LSTR_inform_type ) :: inform

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_lstr requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_lstr' )
          a_arg = 2 ; b_arg = 3 ; radius_arg = 4 ; con_arg = 5
          x_arg = 1 ; obj_arg = 2 ; i_arg = 3
          IF ( nrhs > con_arg )                                                &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_lstr' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_lstr' )
        a_arg = 1 ; b_arg = 2 ; radius_arg = 3 ; con_arg = 4
        x_arg = 1 ; obj_arg = 2 ; i_arg = 3
        IF ( nrhs > con_arg )                                                  &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_lstr' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_lstr provides at most 3 output arguments' )

!  Initialize the internal structures for lstr

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL LSTR_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          con_arg = 1
          IF ( nlhs > con_arg )                                                &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL LSTR_matlab_control_get( plhs( con_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that LSTR_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL LSTR_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_lstr." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_lstr." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL LSTR_matlab_inform_create( plhs( i_arg ), LSTR_pointer )

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

!  Input radius

        radius_in = prhs( radius_arg )
        IF ( mxIsNumeric( radius_in ) == 0 )                                   &
           CALL mexErrMsgTxt( ' There must a scalar radius ' )
        radius_pr = mxGetPr( radius_in )
        CALL MATLAB_copy_from_ptr( radius_pr, radius )

!  Allocate space for the solution

        ALLOCATE( X( n ), STAT = info )

!  Solve the problem

        U( : m ) = B( : m )
        inform%status = 1

        DO                              ! Iteration to find the minimizer
          CALL LSTR_solve( m, n, radius, X, U, V, data, control, inform )
          SELECT CASE( inform%status )  !  Branch as a result of inform%status
          CASE( 2 )                     !  Form u <- u + A * v
            i4 = 0
            CALL MOP_Ax( 1.0_wp, A, V, 1.0_wp, U, i4, i4, i4 )
          CASE( 3 )                     !  Form v <- v + A^T * u
            i4 = 0
            CALL MOP_Ax( 1.0_wp, A, U, 1.0_wp, V, i4, i4, i4,                  &
                         transpose = true )
          CASE ( 4, 5 )                 !  Restart
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
        CALL MATLAB_copy_to_ptr( inform%r_norm, obj_pr )

!  Record output information

        CALL LSTR_matlab_inform_get( inform, LSTR_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to LSTR_solve failed ' )
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
        CALL LSTR_terminate( data, control, inform )
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
