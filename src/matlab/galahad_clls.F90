#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.5 - 2026-03-24 AT 13:40 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_CLLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an o by n matrix A_o, an m by n matrix A, an o by o
!  diagonal scaling matrix W, an o-vector b, a constant sigma >= 0,
!  n-vectors x_l <= x_u and m-vectors c_l <= c_u,
!  find a local minimizer of the (REGULARIZED) LINEARLY-CONSTRAINED
!  LINEAR LEAST-SQUARES problem
!    minimize 0.5 * || A x - b||_W^2 + 0.5 * sigma ||x||^2
!    subject to c_l <= A * x <= c_u and x_l <= x <= x_u
!  and where ||v||^2 = v' v and ||v||_W^2 = v' W v.
!  Advantage is taken of sparse A_o and A.
!
!  Simple usage -
!
!  to solve the constrained linear least-squares problem
!   [ x, inform, aux ]
!    = galahad_clls( A_o, b, sigma, A, c_l, c_u, x_l, x_u, w, x_s, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!    = galahad_clls( 'initial' )
!
!  to solve the same problem using existing data structures
!   [ x, inform, aux ]
!    = galahad_clls( 'existing', A_o, b, sigma, A, c_l, c_u, x_l, x_u, ...
!                     w, x_s, control )
!
!  to remove data structures after solution
!   galahad_clls( 'final' )
!
!  Usual Input -
!    A_o: the o by n matrix A_o
!    b: the o-vector b
!    sigma: the regularization parameter sigma >= 0
!    A: the m by n matrix A
!    c_l: the m-vector c_l. The value -inf should be used for infinite bounds
!    c_u: the m-vector c_u. The value inf should be used for infinite bounds
!    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
!    x_u: the n-vector x_u. The value inf should be used for infinite bounds
!
!  Optional Input (either or both may be given, with w before control)
!    w: the o-vector of weights w for which W=diag(w) (= 1 if w is 
!       not specified)
!    x_s: the n-vector of shifts x_s (= 0 if x_s is not specified)
!       ** N.B. If x_s is required and n=o, w and x_s should 
!          both be provided in that order, even if defaults are used. 
!    control: a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type CLLS_CONTROL as described in the
!      manual for the fortran 90 package GALAHAD_CLLS.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/clls.pdf
!
!  Usual Output -
!   x: a local minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type CLLS_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_CLLS.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/clls.pdf
!  aux: a structure containing Lagrange multipliers and constraint status
!   aux.r: values of the residuals A_o * x - b
!   aux.c: values of the constraints A * x
!   aux.y: Lagrange multipliers corresponding to the general constraints
!        c_l <= A * x <= c_u
!   aux.z: dual variables corresponding to the bound constraints
!        x_l <= x <= x_u
!   aux.c_status: vector indicating the status of the general constraints
!           c_status(i) < 0 if (c_l)_i = (A * x)_i
!           c_status(i) = 0 if (c_i)_i < (A * x)_i < (c_u)_i
!           c_status(i) > 0 if (c_u)_i = (A * x)_i
!   aux.x_status: vector indicating the status of the bound constraints
!           x_status(i) < 0 if (x_l)_i = (x)_i
!           x_status(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
!           x_status(i) > 0 if (x_u)_i = (x)_i
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.2. December 18th, 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_CLLS_MATLAB_TYPES
      USE GALAHAD_CLLS_double
      IMPLICIT NONE
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

! ------------------------- Do not change -------------------------------

!  Keep the above subroutine, argument, and function declarations for use
!  in all your fortran mex files.
!
      INTEGER * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )

      INTEGER, PARAMETER :: slen = 30
      LOGICAL :: mxIsChar, mxIsStruct, mxIsClass
      mwSize :: mxGetString
      mwSize :: mxIsNumeric
      mwSize :: mxGetN
      mwPointer :: mxCreateStructMatrix, mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, n_v, info
      INTEGER * 4 :: i4
      mwSize :: ao_arg, b_arg, a_arg, cl_arg, cu_arg, xl_arg, xu_arg
      mwSize :: sigma_arg, w_arg, xs_arg, c_arg, optional_arg
      mwSize :: x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: b_pr, sigma_pr, cl_pr, cu_pr, xl_pr, xu_pr, w_pr, xs_pr
      mwPointer :: ao_in, b_in, a_in, cl_in, cu_in, xl_in, xu_in
      mwPointer :: sigma_in, w_in, xs_in, c_in, i_in
      mwPointer :: x_pr, y_pr, z_pr, c_status_pr, x_status_pr, c_pr, r_pr
      mwPointer :: aux_r_pr, aux_c_pr, aux_y_pr, aux_z_pr
      mwPointer :: aux_c_status_pr, aux_x_status_pr

      CHARACTER ( LEN = 1 ) :: array
      CHARACTER ( len = 80 ) :: char_output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( CLLS_pointer_type ) :: CLLS_pointer
      REAL ( KIND = wp ) :: sigma
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W

      mwSize, PARAMETER :: naux = 6
      CHARACTER ( LEN = 8 ), PARAMETER :: faux( naux ) = (/                    &
           'r       ', 'c       ', 'y       ', 'z       ',                     &
           'c_status', 'x_status' /)

!  arguments for CLLS

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( CLLS_control_type ), SAVE :: control
      TYPE ( CLLS_inform_type ), SAVE :: inform
      TYPE ( CLLS_data_type ), SAVE :: data

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_clls requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_clls' )
          ao_arg = 2 ; b_arg = 3 ; sigma_arg = 4 ; a_arg = 5
          cl_arg = 6 ; cu_arg = 7 ; xl_arg = 8 ; xu_arg = 9 ; optional_arg = 10
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > 12 )                                                     &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_clls' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_clls' )
        ao_arg = 1 ; b_arg = 2 ; sigma_arg = 3 ; a_arg = 4
        cl_arg = 5 ; cu_arg = 6 ; xl_arg = 7 ; xu_arg = 8 ; optional_arg = 9
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > 11 )                                                       &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_clls' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_clls provides at most 3 output arguments' )

!  Initialize the internal structures for clls

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL CLLS_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL CLLS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that CLLS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  Import the problem data

        p%new_problem_structure = .TRUE.

!  Check to ensure the input for Ao is a number

        ao_in = prhs( ao_arg )
        IF ( mxIsNumeric( ao_in ) == 0 )                                       &
          CALL mexErrMsgTxt( ' There must be a matrix A_o ' )
        CALL MATLAB_transfer_matrix( ao_in, p%Ao, col_ptr, .FALSE. )
        p%n = p%Ao%n ; p%o = p%Ao%m

!  Check to ensure the input for A is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, p%A, col_ptr, .FALSE. )
        IF ( p%A%n /= p%n )                                                    &
          CALL mexErrMsgTxt( ' Column dimensions of A_o and A must agree' )
        p%m = p%A%m

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

        ALLOCATE( p%B( p%o ), p%X_l( p%n ), p%X_u( p%n ), p%R( p%o ),          &
                  p%C( p%m ), p%C_l( p%m ), p%C_u( p%m ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector b ' )
        b_pr = mxGetPr( b_in )
        CALL MATLAB_copy_from_ptr( b_pr, p%B, p%n )

!  Input sigma

        sigma_in = prhs( sigma_arg )
        IF ( mxIsNumeric( sigma_in ) == 0 )                                    &
           CALL mexErrMsgTxt( ' There must a scalar sigma ' )
        sigma_pr = mxGetPr( sigma_in )
        CALL MATLAB_copy_from_ptr( sigma_pr, sigma )

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

!  Input c_l

        cl_in = prhs( cl_arg )
        IF ( mxIsNumeric( cl_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a vector c_l ' )
        cl_pr = mxGetPr( cl_in )
        CALL MATLAB_copy_from_ptr( cl_pr, p%C_l, p%m )
        p%C_l = MAX( p%C_l, - 10.0_wp * CONTROL%infinity )

!  Input c_u

        cu_in = prhs( cu_arg )
        IF ( mxIsNumeric( cu_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a vector c_u ' )
        cu_pr = mxGetPr( cu_in )
        CALL MATLAB_copy_from_ptr( cu_pr, p%C_u, p%m )
        p%C_u = MIN( p%C_u, 10.0_wp * CONTROL%infinity )

!  Sort through optional arguments

        w_arg = - 1 ; xs_arg = - 1 ; c_arg = - 1
        DO i = optional_arg, nrhs
          i_in = prhs( i )

!  Input optional control data

          IF ( mxIsStruct( i_in ) ) THEN ! input control
            c_arg = i
            c_in = i_in
            s_len = slen
            CALL CLLS_matlab_control_set( c_in, control, s_len )

!  Input optional array argument

          ELSE IF ( mxIsClass( i_in, 'double') ) THEN
            IF ( w_arg > 0 .AND. xs_arg > 0 ) CALL mexErrMsgTxt(               &
              ' array input arguments cohort, w and x_s already provided' )
            n_v = INT( mxGetN( i_in ) )
            IF ( p%n /= p%o ) THEN
              IF ( n_v == p%o ) THEN  ! argument is w
                array = 'w'
              ELSE IF ( n_v == p%n ) THEN ! argument is x_s
                array = 'x'
              ELSE ! argument not recognised
                CALL mexErrMsgTxt( ' array input argument not recognised' )
              END IF              
            ELSE
              IF ( n_v == p%n ) THEN  ! argument is x_s or w
                IF ( w_arg < 0 ) THEN
                  array = 'w'
                ELSE IF ( xs_arg < 0 ) THEN
                  array = 'x'
                ELSE
                  CALL mexErrMsgTxt( ' too may optional array input arguments' )
                END IF
              ELSE ! argument not recognised
                CALL mexErrMsgTxt( ' arrays input argument incorrect size' )
              END IF              
            END IF

!  Input w

            IF ( array == 'w' ) THEN
              w_arg = i
              w_in = i_in
              w_pr = mxGetPr( w_in )
              ALLOCATE( p%W( p%o ), STAT = info )
              CALL MATLAB_copy_from_ptr( w_pr, p%W, p%o )

!  Input x_s

            ELSE
              xs_arg = i
              xs_in = i_in
              xs_pr = mxGetPr( xs_in )
              ALLOCATE( p%X_s( p%n ), STAT = info )
              CALL MATLAB_copy_from_ptr( xs_pr, p%X_s, p%n )
            END IF
          ELSE
            CALL mexErrMsgTxt( ' Unknown optional argument' )
          END IF
        END DO

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( char_output_unit, "( I0 )" ) control%error
          filename = "output_clls." // TRIM( char_output_unit )
           OPEN( control%error, FILE = filename, FORM = 'FORMATTED',           &
                 STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_clls." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                     STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL CLLS_matlab_inform_create( plhs( i_arg ), CLLS_pointer )

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%Z( p%n ), p%Y( p%m ), STAT = info )
        p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp

!  Solve the linearly-constrained least-squares problem

!       WRITE( str, "( ' print_level = ', I0 )" ) control%print_level
!       i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )

        CALL CLLS_solve( p, data, control, inform )

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
        plhs( x_arg ) = MATLAB_create_real( p%n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( p%X, x_pr, p%n )

!  Record output information

        CALL CLLS_matlab_inform_get( inform, CLLS_pointer )

!  if required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1_mws_, 1_mws_, naux, faux )

          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'r', p%o, aux_r_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'c', p%m, aux_c_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'y', p%m, aux_y_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'z', p%n, aux_z_pr )
          CALL MATLAB_create_integer_component( plhs( aux_arg ),               &
            'c_status', p%m, aux_c_status_pr )
          CALL MATLAB_create_integer_component( plhs( aux_arg ),               &
            'x_status', p%n, aux_x_status_pr )

!  copy the values

          r_pr = mxGetPr( aux_r_pr )
          CALL MATLAB_copy_to_ptr( p%R, r_pr, p%o )
          c_pr = mxGetPr( aux_c_pr )
          CALL MATLAB_copy_to_ptr( p%C, c_pr, p%m )
          y_pr = mxGetPr( aux_y_pr )
          CALL MATLAB_copy_to_ptr( p%Y, y_pr, p%m )
          z_pr = mxGetPr( aux_z_pr )
          CALL MATLAB_copy_to_ptr( p%Z, z_pr, p%n )
          c_status_pr = mxGetPr( aux_c_status_pr )
          CALL MATLAB_copy_to_ptr( p%C_status, c_status_pr, p%m )
          x_status_pr = mxGetPr( aux_x_status_pr )
          CALL MATLAB_copy_to_ptr( p%X_status, x_status_pr, p%n )

        END IF

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexWarnMsgTxt( ' Call to CLLS_solve failed ' )
      END IF

!      WRITE( message, * ) inform%status
!      CALL MEXPRINTF( TRIM( message ) // char( 13 ) )

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%Ao%row ) ) DEALLOCATE( p%Ao%row, STAT = info )
        IF ( ALLOCATED( p%Ao%col ) ) DEALLOCATE( p%Ao%col, STAT = info )
        IF ( ALLOCATED( p%Ao%val ) ) DEALLOCATE( p%Ao%val, STAT = info )
        IF ( ALLOCATED( p%b ) ) DEALLOCATE( p%B, STAT = info )
        IF ( ALLOCATED( p%A%row ) ) DEALLOCATE( p%A%row, STAT = info )
        IF ( ALLOCATED( p%A%col ) ) DEALLOCATE( p%A%col, STAT = info )
        IF ( ALLOCATED( p%A%val ) ) DEALLOCATE( p%A%val, STAT = info )
        IF ( ALLOCATED( p%C_l ) ) DEALLOCATE( p%C_l, STAT = info )
        IF ( ALLOCATED( p%C_u ) ) DEALLOCATE( p%C_u, STAT = info )
        IF ( ALLOCATED( p%X_l ) ) DEALLOCATE( p%X_l, STAT = info )
        IF ( ALLOCATED( p%X_u ) ) DEALLOCATE( p%X_u, STAT = info )
        IF ( ALLOCATED( p%X ) ) DEALLOCATE( p%X, STAT = info )
        IF ( ALLOCATED( p%Y ) ) DEALLOCATE( p%Y, STAT = info )
        IF ( ALLOCATED( p%Z ) ) DEALLOCATE( p%Z, STAT = info )
        IF ( ALLOCATED( p%R ) ) DEALLOCATE( p%R, STAT = info )
        IF ( ALLOCATED( p%C ) ) DEALLOCATE( p%C, STAT = info )
        IF ( ALLOCATED( p%C_status ) ) DEALLOCATE( p%C_status, STAT = info )
        IF ( ALLOCATED( p%X_status ) ) DEALLOCATE( p%X_status, STAT = info )
        IF ( ALLOCATED( W ) ) DEALLOCATE( W, STAT = info )
        IF ( ALLOCATED( p%X_s ) ) DEALLOCATE( p%X_S, STAT = info )
        CALL CLLS_terminate( data, control, inform )
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
