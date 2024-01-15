#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.3 - 2023-12-30 AT 15:30 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_SLLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an o by n matrix Ao, an o-vector b, and a constant sigma >= 0, find
!  a local mimimizer of the SIMPLEX_CONSTRAINED LINER LEAST-SQUARES problem
!    minimize 0.5 || Ao x - b ||^2 + 0.5 sigma ||x||^2
!    subject to sum x_i = 1, x_i >= 0
!  using a projection method.
!  Advantage is taken of sparse Ao.
!
!  Simple usage -
!
!  to solve the simplex_constrained liner least-squares problem
!   [ x, inform, aux ]
!     = galahad_slls( Ao, b, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_slls( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, inform, aux ]
!     = galahad_slls( 'existing', Ao, b, control )
!
!  to remove data structures after solution
!   galahad_slls( 'final' )
!
!  Usual Input -
!    A: the o by n matrix Ao
!    b: the o-vector b
!
!  Optional Input -
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type SLLS_CONTROL as described in the
!      manual for the fortran 90 package GALAHAD_SLLS.
!      In particular if the weight sigma is nonzero, it
!      should be passed via control.weight.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/slls.pdf
!
!  Usual Output -
!   x: a global minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type SLLS_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_SLLS.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/slls.pdf
!   aux: a structure containing Lagrange multipliers and constraint status
!    aux.z: dual variables corresponding to the non-negativity constraints
!         x_i >= 0
!    aux.x_stat: vector indicating the status of the bound constraints
!            x_stat(i) < (x)_i = 0
!            x_stat(i) = (x)_i > 0
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
      USE GALAHAD_SLLS_MATLAB_TYPES
      USE GALAHAD_SLLS_double
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
      mwSize :: c_arg, x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: b_pr
      mwPointer :: a_in, b_in, c_in
      mwPointer :: x_pr, z_pr, x_stat_pr
      mwPointer :: aux_z_pr, aux_x_stat_pr

      CHARACTER ( len = 80 ) :: char_output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( SLLS_pointer_type ) :: SLLS_pointer
      INTEGER * 4, ALLOCATABLE, DIMENSION( : ) :: X_stat

      mwSize, PARAMETER :: naux = 2
      CHARACTER ( LEN = 6 ), PARAMETER :: faux( naux ) = (/                    &
           'z     ', 'x_stat' /)

!  arguments for SLLS

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( SLLS_control_type ), SAVE :: control
      TYPE ( SLLS_inform_type ), SAVE :: inform
      TYPE ( SLLS_data_type ), SAVE :: data
      TYPE ( GALAHAD_userdata_type ) :: userdata

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_slls requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_slls' )
          a_arg = 2 ; b_arg = 3 ; c_arg = 4
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_slls' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_slls' )
        a_arg = 1 ; b_arg = 2 ; c_arg = 3
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_slls' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_slls provides at most 3 output arguments' )

!  Initialize the internal structures for slls

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL SLLS_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL SLLS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that SLLS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the third argument is present, extract the input control data

        s_len = slen
        IF ( nrhs == c_arg ) THEN
          c_in = prhs( c_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' last input argument must be a structure' )
          CALL SLLS_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( char_output_unit, "( I0 )" ) control%error
          filename = "output_slls." // TRIM( char_output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_slls." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL SLLS_matlab_inform_create( plhs( i_arg ), SLLS_pointer )

!  Import the problem data

         p%new_problem_structure = .TRUE.

!  Check to ensure the input for H is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, p%Ao, col_ptr, .FALSE. )
        p%n = p%Ao%n ; p%o = p%Ao%m

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

        ALLOCATE( p%B( p%o ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector b ' )
        b_pr = mxGetPr( b_in )
        CALL MATLAB_copy_from_ptr( b_pr, p%B, p%o )

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%Z( p%n ), STAT = info )

        p%X = 0.0_wp
        p%Z = 0.0_wp

        ALLOCATE( X_stat( p%n ), STAT = info )

!  Solve the bound-constrained least-squares problem

        CALL SLLS_solve( p, X_stat, data, control, inform, userdata )

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

        CALL SLLS_matlab_inform_get( inform, SLLS_pointer )

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
          CALL mexErrMsgTxt( ' Call to SLLS_solve failed ' )
      END IF

!      WRITE( message, * ) inform%status
!      CALL MEXPRINTF( TRIM( message ) // char( 13 ) )

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%Ao%row ) ) DEALLOCATE( p%Ao%row, STAT = info )
        IF ( ALLOCATED( p%Ao%col ) ) DEALLOCATE( p%Ao%col, STAT = info )
        IF ( ALLOCATED( p%Ao%val ) ) DEALLOCATE( p%Ao%val, STAT = info )
        IF ( ALLOCATED( p%B ) ) DEALLOCATE( p%B, STAT = info )
        IF ( ALLOCATED( p%G ) ) DEALLOCATE( p%G, STAT = info )
        IF ( ALLOCATED( p%R ) ) DEALLOCATE( p%R, STAT = info )
        IF ( ALLOCATED( p%X ) ) DEALLOCATE( p%X, STAT = info )
        IF ( ALLOCATED( p%Z ) ) DEALLOCATE( p%Z, STAT = info )
        IF ( ALLOCATED( X_stat ) ) DEALLOCATE( X_stat, STAT = info )
        CALL SLLS_terminate( data, control, inform )
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
