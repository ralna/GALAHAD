#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_WCP
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an m by n matrix A, n-vectors g, x_l <= x_u and m-vectors c_l <= c_u,
!  find a well-centered interior point within the polytope
!          c_l <= A x <= c_u and x_l <=  x <= x_u,
!  for which the dual feasibility conditions
!           g = A' y + z
!  for Lagrange multipliers y and z are satisfied, using an infeasible-point
!  primal-dual method. Advantage is taken of sparse A.
!
!  Simple usage -
!
!  to find a well-centered feasible point
!   [ x, inform, aux ]
!     = galahad_wcp( A, c_l, c_u, x_l, x_u, g )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_wcp( 'initial' )
!
!  to find a well-centered feasible point using existing structures
!   [ x, inform, aux ]
!     = galahad_wcp( 'existing, g, A, c_l, c_u, x_l, x_u, g, control )
!
!  to remove data structures after solution
!   galahad_wcp( 'final' )
!
!  Usual Input -
!    A: the m by n matrix A
!    c_l: the m-vector c_l. The value -inf should be used for infinite bounds
!    c_u: the m-vector c_u. The value inf should be used for infinite bounds
!    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
!    x_u: the n-vector x_u. The value inf should be used for infinite bounds
!
!  Optional Input -
!    g: the n-vector g. If absent, g = 0 is presumed
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type WCP_CONTROL as described in the
!      manual for the fortran 90 package GALAHAD_WCP.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/wcp.pdf
!
!  Usual Output -
!   x: a local minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type WCP_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_WCP.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/wcp.pdf
!  aux: a structure containing Lagrange multipliers and constraint status
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
!   originally released with GALAHAD Version 2.3. November 6th 2008

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_WCP_MATLAB_TYPES
      USE GALAHAD_WCP_double
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
      mwPointer :: mxCreateStructMatrix, mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, info
      INTEGER * 4 :: i4
      mwSize :: s_len
      mwSize :: a_arg, cl_arg, cu_arg
      mwSize :: xl_arg, xu_arg, c_arg, g_arg, x_arg, i_arg, aux_arg

      mwPointer :: a_in, cl_in, cu_in, xl_in, xu_in, g_in
      mwPointer :: c_in, x_pr, y_pr, z_pr, x_status_pr
      mwPointer :: cl_pr, cu_pr, xl_pr, xu_pr, g_pr
      mwPointer :: c_status_pr, c_pr

      mwPointer :: aux_c_pr, aux_y_pr, aux_z_pr
      mwPointer :: aux_c_status_pr, aux_x_status_pr

!     CHARACTER ( len = 80 ) :: message
      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( WCP_pointer_type ) :: WCP_pointer

      INTEGER * 4, PARAMETER :: naux = 5
      CHARACTER ( LEN = 8 ), PARAMETER :: faux( naux ) = (/                    &
           'c       ', 'y       ', 'z       ', 'c_status', 'x_status' /)

!  arguments for WCP

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( WCP_control_type ), SAVE :: control
      TYPE ( WCP_inform_type ) :: inform
      TYPE ( WCP_data_type ), SAVE :: data

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_wcp requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_wcp' )
          a_arg = 2 ; cl_arg = 3 ; cu_arg = 4
          xl_arg = 5 ; xu_arg = 6 ; g_arg = 7 ; c_arg = 8
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_wcp' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_wcp' )
        a_arg = 1 ; cl_arg = 2 ; cu_arg = 3
        xl_arg = 4 ; xu_arg = 5 ; g_arg = 6 ; c_arg = 7
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_wcp' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_wcp provides at most 3 output arguments' )

!  Initialize the internal structures for wcp

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL WCP_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL WCP_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that WCP_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the third argument is present, extract the input control data

        s_len = slen
        IF ( nrhs == c_arg ) THEN
          c_in = prhs( c_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' last input argument must be a structure' )
          CALL WCP_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_wcp." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_wcp." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL WCP_matlab_inform_create( plhs( i_arg ), WCP_pointer )

!  Import the problem data

         p%new_problem_structure = .TRUE.

!  Check to ensure the input for A is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, p%A, col_ptr, .FALSE. )
        p%m = p%A%m ; p%n = p%A%n

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

        ALLOCATE( p%G( p%n ), p%X_l( p%n ), p%X_u( p%n ), p%C( p%m ),          &
                  p%C_l( p%m ), p%C_u( p%m ), p%Z_l( p%n ), p%Z_u( p%n ),      &
                  p%Y_l( p%m ), p%Y_u( p%m ), STAT = info )

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

!  Input g

        IF ( nrhs == g_arg ) THEN
          g_in = prhs( g_arg )
          IF ( mxIsNumeric( g_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a vector g ' )
          g_pr = mxGetPr( g_in )
          CALL MATLAB_copy_from_ptr( g_pr, p%G, p%n )
          p%gradient_kind = 2
        ELSE
          p%gradient_kind = 0
        END IF

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%Z( p%n ), p%Y( p%m ), STAT = info )

        p%X = 0.0_wp
        p%Y = 0.0_wp
        p%Z = 0.0_wp

        p%Y_l = 0.0_wp
        p%Y_u = 0.0_wp

        p%Z_l = 0.0_wp
        p%Z_u = 0.0_wp

!  Find the well-centered feasible point

        inform%status = - 1
        CALL WCP_solve( p, data, control, inform )

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

        CALL WCP_matlab_inform_get( inform, WCP_pointer )

!  if required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1_mws_, 1_mws_, naux, faux )

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

          c_pr = mxGetPr( aux_c_pr )
          CALL MATLAB_copy_to_ptr( p%C, c_pr, p%m )
          y_pr = mxGetPr( aux_y_pr )
          CALL MATLAB_copy_to_ptr( p%Y, y_pr, p%m )
          z_pr = mxGetPr( aux_z_pr )
          CALL MATLAB_copy_to_ptr( p%Z, z_pr, p%n )
          c_status_pr = mxGetPr( aux_c_status_pr )
          CALL MATLAB_copy_to_ptr( inform%c_status, c_status_pr, p%m )
          x_status_pr = mxGetPr( aux_x_status_pr )
          CALL MATLAB_copy_to_ptr( inform%x_status, x_status_pr, p%n )

        END IF

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to WCP_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%G ) ) DEALLOCATE( p%G, STAT = info )
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
        IF ( ALLOCATED( p%C ) ) DEALLOCATE( p%C, STAT = info )
        IF ( ALLOCATED( inform%c_status ) )                                    &
          DEALLOCATE( inform%c_status, STAT = info )
        IF ( ALLOCATED( inform%x_status ) )                                    &
          DEALLOCATE( inform%x_status, STAT = info )
        CALL WCP_terminate( data, control, inform )
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
