#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.3 - 12/12/2020 AT 17:05 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_BLLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector
!  g, a constant f, and n-vectors x_l <= x_u, find a local mimimizer
!  of the BOUND_CONSTRAINED LINER LEAST-SQUARES problem
!    minimize 0.5 || A x - b ||^2 + 0.5 sigma ||x||^2
!    subject to x_l <= x <= x_u
!  using a projection method.
!  Advantage is taken of sparse A.
!
!  Simple usage -
!
!  to solve the bound-constrained quadratic program
!   [ x, inform, aux ]
!     = galahad_blls( A, b,  x_l, x_u, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_blls( 'initial' )
!
!  to solve the bound-constrained QP using existing data structures
!   [ x, inform, aux ]
!     = galahad_blls( 'existing', A, b, x_l, x_u, control )
!
!  to remove data structures after solution
!   galahad_blls( 'final' )
!
!  Usual Input -
!    A: the m by n matrix A
!    b: the m-vector b
!    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
!    x_u: the n-vector x_u. The value inf should be used for infinite bounds
!
!  Optional Input -
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type BLLS_CONTROL as described in the
!      manual for the fortran 90 package GALAHAD_BLLS.
!      In particular if the weight sigma is nonzero, it 
!      should be passed via control.weight.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/blls.pdf
!
!  Usual Output -
!   x: a global minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type BLLS_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_BLLS.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/blls.pdf
!   aux: a structure containing Lagrange multipliers and constraint status
!    aux.z: dual variables corresponding to the bound constraints
!         x_l <= x <= x_u
!    aux.x_stat: vector indicating the status of the bound constraints
!            x_stat(i) < 0 if (x_l)_i = (x)_i
!            x_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
!            x_stat(i) > 0 if (x_u)_i = (x)_i
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. December 12th 2020

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_BLLS_MATLAB_TYPES
      USE GALAHAD_BLLS_double
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
      mwSize :: mxGetString
      mwSize :: mxIsNumeric
      mwPointer :: mxCreateStructMatrix, mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, info
      INTEGER * 4 :: i4
      mwSize :: a_arg, b_arg
      mwSize :: xl_arg, xu_arg, c_arg, x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: b_pr, xl_pr, xu_pr
      mwPointer :: a_in, b_in, xl_in, xu_in, c_in
      mwPointer :: x_pr, z_pr, x_stat_pr
      mwPointer :: aux_z_pr, aux_x_stat_pr

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( BLLS_pointer_type ) :: BLLS_pointer
      INTEGER * 4, ALLOCATABLE, DIMENSION( : ) :: X_stat

      mwSize, PARAMETER :: naux = 2
      CHARACTER ( LEN = 6 ), PARAMETER :: faux( naux ) = (/                    &
           'z     ', 'x_stat' /)

!  arguments for BLLS

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( BLLS_control_type ), SAVE :: control
      TYPE ( BLLS_inform_type ), SAVE :: inform
      TYPE ( BLLS_data_type ), SAVE :: data
      TYPE ( GALAHAD_userdata_type ) :: userdata

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_blls requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_blls' )
          a_arg = 2 ; b_arg = 3
          xl_arg = 4 ; xu_arg = 5 ; c_arg = 6
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_blls' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_blls' )
        a_arg = 1 ; b_arg = 2 ; 
        xl_arg = 3 ; xu_arg = 4 ; c_arg = 5
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_blls' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_blls provides at most 3 output arguments' )

!  Initialize the internal structures for blls

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL BLLS_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL BLLS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that BLLS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the third argument is present, extract the input control data

        s_len = slen
        IF ( nrhs == c_arg ) THEN
          c_in = prhs( c_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' last input argument must be a structure' )
          CALL BLLS_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_blls." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_blls." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL BLLS_matlab_inform_create( plhs( i_arg ), BLLS_pointer )

!  Import the problem data

         p%new_problem_structure = .TRUE.

!  Check to ensure the input for H is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, p%A, col_ptr, .FALSE. )
        p%n = p%A%n ; p%m = p%A%m

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

        ALLOCATE( p%B( p%m ), p%X_l( p%n ), p%X_u( p%n ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector g ' )
        b_pr = mxGetPr( b_in )
        CALL MATLAB_copy_from_ptr( b_pr, p%B, p%m )

!  Input x_l

        xl_in = prhs( xl_arg )
        IF ( mxIsNumeric( xl_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a vector x_l ' )
        xl_pr = mxGetPr( xl_in )
        CALL MATLAB_copy_from_ptr( xl_pr, p%X_l, p%n )
        p%X_l = MAX( p%X_l, - 10.0_wp * CONTROL%infinity )

!  Input x_u

        xu_in = prhs( xu_arg )
        IF ( mxIsNumeric( xu_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a vector x_u ' )
        xu_pr = mxGetPr( xu_in )
        CALL MATLAB_copy_from_ptr( xu_pr, p%X_u, p%n )
        p%X_u = MIN( p%X_u, 10.0_wp * CONTROL%infinity )

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%Z( p%n ), STAT = info )

        p%X = 0.0_wp
        p%Z = 0.0_wp

        ALLOCATE( X_stat( p%n ), STAT = info )

!  Solve the bound-constrained least-squares problem

        CALL BLLS_solve( p, X_stat, data, control, inform, userdata )

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

        CALL BLLS_matlab_inform_get( inform, BLLS_pointer )

!  if required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1, 1, naux, faux )

          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'z', p%n, aux_z_pr )
          CALL MATLAB_create_integer_component( plhs( aux_arg ),               &
            'x_stat', p%n, aux_x_stat_pr )

!  copy the values

          z_pr = mxGetPr( aux_z_pr )
          CALL MATLAB_copy_to_ptr( p%Z, z_pr, p%n )
          x_stat_pr = mxGetPr( aux_x_stat_pr )
          CALL MATLAB_copy_to_ptr( X_stat, x_stat_pr, p%n )

        END IF

!  Check for errors

        IF ( inform%status < 0 )                                              &
          CALL mexErrMsgTxt( ' Call to BLLS_solve failed ' )
      END IF

!      WRITE( message, * ) inform%status
!      CALL MEXPRINTF( TRIM( message ) // char( 13 ) )

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%A%row ) ) DEALLOCATE( p%A%row, STAT = info )
        IF ( ALLOCATED( p%A%col ) ) DEALLOCATE( p%A%col, STAT = info )
        IF ( ALLOCATED( p%A%val ) ) DEALLOCATE( p%A%val, STAT = info )
        IF ( ALLOCATED( p%B ) ) DEALLOCATE( p%B, STAT = info )
        IF ( ALLOCATED( p%G ) ) DEALLOCATE( p%G, STAT = info )
        IF ( ALLOCATED( p%C ) ) DEALLOCATE( p%C, STAT = info )
        IF ( ALLOCATED( p%X_l ) ) DEALLOCATE( p%X_l, STAT = info )
        IF ( ALLOCATED( p%X_u ) ) DEALLOCATE( p%X_u, STAT = info )
        IF ( ALLOCATED( p%X ) ) DEALLOCATE( p%X, STAT = info )
        IF ( ALLOCATED( p%Z ) ) DEALLOCATE( p%Z, STAT = info )
        IF ( ALLOCATED( X_stat ) ) DEALLOCATE( X_stat, STAT = info )
        CALL BLLS_terminate( data, control, inform )
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
