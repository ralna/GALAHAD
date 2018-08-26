#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_EQP
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector
!  g, a constant f and an m-vector c, find the minimizer of the
!  EQUALITY-CONSTRAINED QUADRATIC PROGRAMMING problem
!    minimize 0.5 * x' * H * x + g' * x + f
!    subject to  A * x + c = 0.
!  An additional trust-region constraint may be imposed to prevent unbounded
!  solutions. H need not be definite. Advantage is taken of sparse A and H.
!
!  Simple usage -
!
!  to solve the quadratic program
!   [ x, inform, aux ]
!     = galahad_eqp( H, g, f, A, c, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_eqp( 'initial' )
!
!  to solve the quadratic program using existing data structures
!   [ x, inform, aux ]
!     = galahad_eqp( 'existing', H, g, f, A, c, control )
!
!  to remove data structures after solution
!   galahad_eqp( 'final' )
!
!  Usual Input -
!    H: the symmetric n by n matrix H
!    g: the n-vector g
!    f: the scalar f
!    A: the m by n matrix A
!    c: the m-vector c
!
!  Optional Input -
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type EQP_CONTROL as described in the
!      manual for the fortran 90 package GALAHAD_EQP.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/eqp.pdf
!
!  Usual Output -
!   x: a local minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type EQP_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_EQP.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/eqp.pdf
!  aux: a structure containing Lagrange multipliers and constraint status
!   aux.y: Lagrange multipliers corresponding to the constraints A x + c = 0
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 18th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_EQP_MATLAB_TYPES
      USE GALAHAD_EQP_double
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
      mwPointer :: mxCreateStructMatrix, mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, info
      INTEGER * 4 :: i4
      mwSize :: h_arg, g_arg, f_arg, a_arg, c_arg
      mwSize :: con_arg, x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: g_pr, f_pr, c_pr
      mwPointer :: h_in, g_in, f_in, a_in, c_in, con_in
      mwPointer :: x_pr, y_pr, aux_y_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 20 ) :: message
      CHARACTER ( len = 8 ) :: mode
      TYPE ( EQP_pointer_type ) :: EQP_pointer

      INTEGER * 4, PARAMETER :: naux = 1
      CHARACTER ( LEN = 6 ), PARAMETER :: faux( naux ) = (/ 'y     ' /)

!  arguments for EQP

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( EQP_control_type ), SAVE :: control
      TYPE ( EQP_inform_type ), SAVE :: inform
      TYPE ( EQP_data_type ), SAVE :: data

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_eqp requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_eqp' )
          h_arg = 2 ; g_arg = 3 ; f_arg = 4 ; a_arg = 5
          c_arg = 6 ; con_arg = 7
          IF ( nrhs > con_arg )                                                &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_eqp' )
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_eqp' )
        h_arg = 1 ; g_arg = 2 ; f_arg = 3 ; a_arg = 4 ; c_arg = 5 ; con_arg = 6
        IF ( nrhs > con_arg )                                                  &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_eqp' )
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_eqp provides at most 3 output arguments' )

!  Initialize the internal structures for eqp

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL EQP_initialize( data, control, inform )

        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL EQP_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that EQP_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the third argument is present, extract the input control data

        s_len = slen
        IF ( nrhs == con_arg ) THEN
          con_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( con_in ) )                                    &
            CALL mexErrMsgTxt( ' last input argument must be a structure' )
          CALL EQP_matlab_control_set( con_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_eqp." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_eqp." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL EQP_matlab_inform_create( plhs( i_arg ), EQP_pointer )

!  Import the problem data

         p%new_problem_structure = .TRUE.

!  Check to ensure the input for H is a number

        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, p%H, col_ptr, .TRUE. )
        p%n = p%H%n

!  Check to ensure the input for A is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, p%A, col_ptr, .FALSE. )
        IF ( p%A%n /= p%n )                                                    &
          CALL mexErrMsgTxt( ' Column dimensions of H and A must agree' )
        p%m = p%A%m

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

        ALLOCATE( p%G( p%n ), p%C( p%m ), STAT = info )

!  Input g

        g_in = prhs( g_arg )
        IF ( mxIsNumeric( g_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector g ' )
        g_pr = mxGetPr( g_in )
        CALL MATLAB_copy_from_ptr( g_pr, p%G, p%n )

!  Input f

        f_in = prhs( f_arg )
        IF ( mxIsNumeric( f_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must a scalar f ' )
        f_pr = mxGetPr( f_in )
        CALL MATLAB_copy_from_ptr( f_pr, p%f )

!  Input c

        c_in = prhs( c_arg )
        IF ( mxIsNumeric( c_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a vector c ' )
        c_pr = mxGetPr( c_in )
        CALL MATLAB_copy_from_ptr( c_pr, p%C, p%m )

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%Y( p%m ), STAT = info )

        p%X = 0.0_wp
        p%Y = 0.0_wp

!  Solve the QP

        CALL EQP_solve( p, data, control, inform )

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
        plhs( x_arg ) = MATLAB_create_real( p%n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( p%X, x_pr, p%n )

!  Record output information

        CALL EQP_matlab_inform_get( inform, EQP_pointer )

!  if required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1_mws_, 1_mws_, naux, faux )

          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
                                             'y', p%m, aux_y_pr )

!  copy the values

          y_pr = mxGetPr( aux_y_pr )
          CALL MATLAB_copy_to_ptr( p%Y, y_pr, p%m )
        END IF
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%H%row ) ) DEALLOCATE( p%H%row, STAT = info )
        IF ( info /=0 ) CALL mexErrMsgTxt( ' deallocate failed ' )
        IF ( ALLOCATED( p%H%col ) ) DEALLOCATE( p%H%col, STAT = info )
        IF ( info /=0 ) CALL mexErrMsgTxt( ' deallocate failed ' )
        IF ( ALLOCATED( p%H%val ) ) DEALLOCATE( p%H%val, STAT = info )
        IF ( info /=0 ) CALL mexErrMsgTxt( ' deallocate failed ' )
        IF ( ALLOCATED( p%G ) ) DEALLOCATE( p%G, STAT = info )
        IF ( ALLOCATED( p%A%row ) ) DEALLOCATE( p%A%row, STAT = info )
        IF ( ALLOCATED( p%A%col ) ) DEALLOCATE( p%A%col, STAT = info )
        IF ( ALLOCATED( p%A%val ) ) DEALLOCATE( p%A%val, STAT = info )
        IF ( ALLOCATED( p%C ) ) DEALLOCATE( p%C, STAT = info )
        IF ( ALLOCATED( p%X ) ) DEALLOCATE( p%X, STAT = info )
        IF ( ALLOCATED( p%Y ) ) DEALLOCATE( p%Y, STAT = info )
        CALL EQP_terminate( data, control, inform )
      END IF

!  Check for errors

      IF ( TRIM( mode ) == 'all' ) THEN
        IF ( inform%status < 0 ) THEN
          WRITE( message, * ) ' inform_status ', inform%status
          i = mexPrintf( TRIM( message ) // char( 13 ) )
!         WRITE( message,*) ' sls_status ', inform%SBLS_inform%SLS_inform%status
!         i = mexPrintf( TRIM( message ) // char( 13 ) )
          CALL mexErrMsgTxt( ' Call to EQP_solve failed ' )
        END IF
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
