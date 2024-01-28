#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.3 - 2023-12-27 AT 13:00 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_BLLSB
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an o by n matrix A_o, an o by o diagonal scaling matrix W,
!  an o-vector b, a constant sigma >= 0, and n-vectors x_l <= x_u,
!  find a local minimizer of the (REGULARIZED) BOUND-CONSTRAINED
!  LINEAR LEAST-SQUARES problem
!    minimize 0.5 * || A x - b||_W^2 + 0.5 * sigma ||x||^2
!    subject to x_l <= x <= x_u
!  and where ||v||^2 = v' v and ||v||_W^2 = v' W v.
!  Advantage is taken of sparse A_o.
!
!  Simple usage -
!
!  to solve the constrained linear least-squares problem
!   [ x, inform, aux ]
!    = galahad_bllsb( A_o, b, sigma, x_l, x_u, w, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!    = galahad_bllsb( 'initial' )
!
!  to solve the same problem using existing data structures
!   [ x, inform, aux ]
!    = galahad_bllsb( 'existing', A_o, b, sigma, x_l, x_u, w, control )
!
!  to remove data structures after solution
!   galahad_bllsb( 'final' )
!
!  Usual Input -
!    A_o: the o by n matrix A_o
!    b: the o-vector b
!    sigma: the regularization parameter sigma >= 0
!    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
!    x_u: the n-vector x_u. The value inf should be used for infinite bounds
!
!  Optional Input (either or both may be given, with w before control)
!    w: the (diagonal) components of the diagonal scaling matrix W
!    control: a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type BLLSB_CONTROL as described in the
!      manual for the fortran 90 package GALAHARS_BLLSB.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/bllsb.pdf
!
!  Usual Output -
!   x: a local minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type BLLSB_INFORM as described in the manual for
!      the fortran 90 package GALAHARS_BLLSB.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/bllsb.pdf
!  aux: a structure containing Lagrange multipliers and constraint status
!   aux.r: values of the residuals A_o * x - b
!   aux.z: dual variables corresponding to the bound constraints
!        x_l <= x <= x_u
!   aux.x_stat: vector indicating the status of the bound constraints
!           x_stat(i) < 0 if (x_l)_i = (x)_i
!           x_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
!           x_stat(i) > 0 if (x_u)_i = (x)_i
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
      USE GALAHAD_BLLSB_MATLAB_TYPES
      USE GALAHAD_BLLSB_double
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
      mwSize :: ao_arg, b_arg, xl_arg, xu_arg
      mwSize :: sigma_arg, w_arg, c_arg, x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: b_pr, sigma_pr, xl_pr, xu_pr, w_pr
      mwPointer :: ao_in, b_in, xl_in, xu_in
      mwPointer :: sigma_in, w_in, c_in
      mwPointer :: x_pr, z_pr, x_stat_pr, r_pr
      mwPointer :: aux_r_pr, aux_z_pr, aux_x_stat_pr

      CHARACTER ( len = 80 ) :: char_output_unit, filename
      LOGICAL :: opened, w_present, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( BLLSB_pointer_type ) :: BLLSB_pointer
      REAL ( KIND = wp ) :: sigma
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W

      mwSize, PARAMETER :: naux = 3
      CHARACTER ( LEN = 6 ), PARAMETER :: faux( naux ) = (/                    &
           'r     ', 'z     ', 'x_stat' /)

!  arguments for BLLSB

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( BLLSB_control_type ), SAVE :: control
      TYPE ( BLLSB_inform_type ), SAVE :: inform
      TYPE ( BLLSB_data_type ), SAVE :: data

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_bllsb requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_bllsb' )
          ao_arg = 2 ; b_arg = 3 ; sigma_arg = 4 ; xl_arg = 5 ; xu_arg = 6
          w_arg = 7 ; c_arg = 8
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_bllsb' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_bllsb' )
        ao_arg = 1 ; b_arg = 2 ; sigma_arg = 3 ; xl_arg = 4 ; xu_arg = 5
        w_arg = 6 ; c_arg = 7
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_bllsb' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_bllsb provides at most 3 output arguments')

!  Initialize the internal structures for bllsb

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL BLLSB_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL BLLSB_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that BLLSB_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the third argument is present, extract the input control data

        s_len = slen
!       IF ( nrhs == c_arg ) THEN
!         c_in = prhs( c_arg )
!         IF ( .NOT. mxIsStruct( c_in ) )                                      &
!           CALL mexErrMsgTxt( ' last input argument must be a structure' )
!         CALL BLLSB_matlab_control_set( c_in, control, s_len )
!       END IF
        IF ( nrhs > sigma_arg ) THEN
          c_in = prhs( nrhs )
          IF ( mxIsStruct( c_in ) )                                            &
            CALL BLLSB_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( char_output_unit, "( I0 )" ) control%error
          filename = "output_bllsb." // TRIM( char_output_unit )
           OPEN( control%error, FILE = filename, FORM = 'FORMATTED',           &
                 STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_bllsb." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                     STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL BLLSB_matlab_inform_create( plhs( i_arg ), BLLSB_pointer )

!  Import the problem data

         p%new_problem_structure = .TRUE.

!  Check to ensure the input for Ao is a number

        ao_in = prhs( ao_arg )
        IF ( mxIsNumeric( ao_in ) == 0 )                                       &
          CALL mexErrMsgTxt( ' There must be a matrix A_o ' )
        CALL MATLAB_transfer_matrix( ao_in, p%Ao, col_ptr, .FALSE. )
        p%n = p%Ao%n ; p%o = p%Ao%m

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

        ALLOCATE( p%B( p%o ), p%X_l( p%n ), p%X_u( p%n ), p%R( p%o ),          &
                  STAT = info )

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

!  Input w

        w_present = .FALSE.
        IF ( nrhs > sigma_arg ) THEN
          w_in = prhs( w_arg )
          IF ( mxIsNumeric( w_in ) /= 0 ) THEN
            w_present = .TRUE.
            ALLOCATE( W( p%o ), STAT = info )
            w_pr = mxGetPr( w_in )
            CALL MATLAB_copy_from_ptr( w_pr, W, p%o )
          END IF
        END IF

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%Z( p%n ), STAT = info )

        p%X = 0.0_wp
        p%Z = 0.0_wp

!  Solve the QP

!       WRITE( str, "( ' print_level = ', I0 )" ) control%print_level
!       i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )

        IF ( w_present ) THEN
          CALL BLLSB_solve( p, data, control, inform,                          &
                            regularization_weight = sigma, W = W )
        ELSE
          CALL BLLSB_solve( p, data, control, inform,                          &
                            regularization_weight = sigma )
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
        plhs( x_arg ) = MATLAB_create_real( p%n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( p%X, x_pr, p%n )

!  Record output information

        CALL BLLSB_matlab_inform_get( inform, BLLSB_pointer )

!  if required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1_mws_, 1_mws_, naux, faux )

          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'r', p%o, aux_r_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'z', p%n, aux_z_pr )
          CALL MATLAB_create_integer_component( plhs( aux_arg ),               &
            'x_stat', p%n, aux_x_stat_pr )

!  copy the values

          r_pr = mxGetPr( aux_r_pr )
          CALL MATLAB_copy_to_ptr( p%R, r_pr, p%o )
          z_pr = mxGetPr( aux_z_pr )
          CALL MATLAB_copy_to_ptr( p%Z, z_pr, p%n )
          x_stat_pr = mxGetPr( aux_x_stat_pr )
          CALL MATLAB_copy_to_ptr( p%X_status, x_stat_pr, p%n )

        END IF

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexWarnMsgTxt( ' Call to BLLSB_solve failed ' )
      END IF

!      WRITE( message, * ) inform%status
!      CALL MEXPRINTF( TRIM( message ) // char( 13 ) )

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%Ao%row ) ) DEALLOCATE( p%Ao%row, STAT = info )
        IF ( ALLOCATED( p%Ao%col ) ) DEALLOCATE( p%Ao%col, STAT = info )
        IF ( ALLOCATED( p%Ao%val ) ) DEALLOCATE( p%Ao%val, STAT = info )
        IF ( ALLOCATED( p%b ) ) DEALLOCATE( p%B, STAT = info )
        IF ( ALLOCATED( p%X_l ) ) DEALLOCATE( p%X_l, STAT = info )
        IF ( ALLOCATED( p%X_u ) ) DEALLOCATE( p%X_u, STAT = info )
        IF ( ALLOCATED( p%X ) ) DEALLOCATE( p%X, STAT = info )
        IF ( ALLOCATED( p%Z ) ) DEALLOCATE( p%Z, STAT = info )
        IF ( ALLOCATED( p%R ) ) DEALLOCATE( p%R, STAT = info )
        IF ( ALLOCATED( p%X_status ) ) DEALLOCATE( p%X_status, STAT = info )
        IF ( ALLOCATED( W ) ) DEALLOCATE( W, STAT = info )
        CALL BLLSB_terminate( data, control, inform )
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
