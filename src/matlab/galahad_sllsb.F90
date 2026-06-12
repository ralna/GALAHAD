#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.5 - 2026-03-25 AT 13:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_SLLSB
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given an o by n matrix Ao, an o-vector b, and a constant sigma >= 0, find
!  a local mimimizer of the SIMPLEX_CONSTRAINED LINER LEAST-SQUARES problem
!    minimize 0.5 || Ao x - b ||_w^2 + 0.5 sigma || x - x_s ||^2
!    subject to sum_{C_j} x_i = 1, x_{C_j} >= 0 for j = 1,...,m
!  where ||v||^2 = v' v and ||v||_W^2 = v' W v, using an interior-point method
!  Advantage is taken of sparse Ao.
!
!  Simple usage -
!
!  to solve the simplex_constrained liner least-squares problem
!   [ x, inform, aux ]
!     = galahad_sllsb( Ao, b, sigma, cohort, w, x_s, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_sllsb( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, inform, aux ]
!     = galahad_sllsb( 'existing', Ao, b, sigma, cohort, w, x_s, control )
!
!  to remove data structures after solution
!   galahad_sllsb( 'final' )
!
!  Usual Input -
!    A: the o by n matrix Ao
!    b: the o-vector b
!    sigma: the regulaisation weight (sigma>0)
!
!  Optional Input -
!    cohort: the cohorts, so that variable x_i is in cohort C_j if 
!       cohort[i] = j, and x_i is not constrained if cohort[i] = 0 
!    w: the o-vector of weights w for which W=diag(w) (= 1 if w is 
!       not specified)
!       ** N.B. If n=o and both w and cohort are provided, cohort 
!          must proceed w in the calling sequence
!    x_s: the n-vector of shifts x_s (= 0 if x_s is not specified)
!       ** N.B. If x_s is required and n=o, cohort, w and x_s should 
!          all be provided in that order, even if defaults are used. 
!          Otherwise, if cohort and x_s are both required, cohort 
!          must proceed x_s in the calling sequence
!    cohort: the cohorts, so that variable x_i is in cohort C_j if 
!       cohort[i] = j, and x_i is not constrained if cohort[i] = 0 
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type SLLSB_CONTROL as described in the
!      manual for the fortran 90 package GALAHAD_SLLSB.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sllsb.pdf
!
!  Usual Output -
!   x: a global minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type SLLSB_INFORM as described in the manual for
!      the fortran 90 package GALAHAD_SLLSB.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sllsb.pdf
!   aux: a structure containing Lagrange multipliers and constraint status
!    aux.y: Largrange multipliers y corresponding to the simplex constraints
!    aux.z: dual variables z corresponding to the non-negativity constraints
!         x_i >= 0
!    aux.r: values of the residuals r(x) = Ao x - b
!    aux.x_status: vector indicating the status of the bound constraints
!            x_status(i) < (x)_i = 0
!            x_status(i) = (x)_i > 0
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
      USE GALAHAD_SLLSB_MATLAB_TYPES
      USE GALAHAD_SLLSB_double
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
!     LOGICAL :: mxIsChar, mxIsClass
!     INTEGER * 4 :: mxIsStruct
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
      mwSize :: a_arg, b_arg, sigma_arg
      mwSize :: w_arg, xs_arg, co_arg, c_arg, optional_arg
      mwSize :: x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: a_in, b_in, sigma_in, w_in, xs_in, co_in, c_in, i_in
      mwPointer :: b_pr, sigma_pr, w_pr, xs_pr, co_pr
      mwPointer :: x_pr, y_pr, z_pr, r_pr, x_status_pr
      mwPointer :: aux_y_pr, aux_z_pr, aux_r_pr, aux_x_status_pr

      CHARACTER ( LEN = 1 ) :: array
      CHARACTER ( LEN = 80 ) :: char_output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 8 ) :: mode
      TYPE ( SLLSB_pointer_type ) :: SLLSB_pointer

      mwSize, PARAMETER :: naux = 4
      CHARACTER ( LEN = 8 ), PARAMETER :: faux( naux ) = (/                    &
           'y       ', 'z       ', 'r       ', 'x_status' /)

!  arguments for SLLSB

      TYPE ( QPT_problem_type ), SAVE :: p
      TYPE ( SLLSB_control_type ), SAVE :: control
      TYPE ( SLLSB_inform_type ), SAVE :: inform
      TYPE ( SLLSB_data_type ), SAVE :: data

      mwPointer, ALLOCATABLE :: col_ptr( : )
      REAL ( KIND = wp ), ALLOCATABLE :: real_cohort( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_sllsb requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 4 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_sllsb' )
          a_arg = 2 ; b_arg = 3 ; sigma_arg = 4 ; optional_arg = 5
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > 8 )                                                      &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_sllsb' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 3 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_sllsb' )
        a_arg = 1 ; b_arg = 2 ; sigma_arg = 3 ; optional_arg = 4
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > 7 )                                                        &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_sllsb' )
      END IF

      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_sllsb provides at most 3 output arguments' )

!  Initialize the internal structures for sllsb

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL SLLSB_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL SLLSB_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that SLLSB_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  Import the problem data

        p%new_problem_structure = .TRUE.

!  Check to ensure the input for A is a number

        a_in = prhs( a_arg )
        IF ( mxIsNumeric( a_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix A ' )
        CALL MATLAB_transfer_matrix( a_in, p%Ao, col_ptr, .FALSE. )
        p%n = p%Ao%n ; p%o = p%Ao%m ; p%m = 1

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Allocate space for input vectors

!       ALLOCATE( p%B( p%o ), p%X( p%n ), p%X_status( p%n ), STAT = info )
        ALLOCATE( p%B( p%o ), STAT = info )

!  Input b

        b_in = prhs( b_arg )
        IF ( mxIsNumeric( b_in ) == 0 )                                        &
           CALL mexErrMsgTxt( ' There must be a vector b ' )
        b_pr = mxGetPr( b_in )
        CALL MATLAB_copy_from_ptr( b_pr, p%B, p%o )

!  Input sigma

        sigma_in = prhs( sigma_arg )
        IF ( mxIsNumeric( sigma_in ) == 0 )                                    &
          CALL mexErrMsgTxt( ' There must a scalar sigma ' )
        sigma_pr = mxGetPr( sigma_in )
        CALL MATLAB_copy_from_ptr( sigma_pr, p%regularization_weight )

!  Sort through optional arguments

        co_arg = - 1  ; w_arg = - 1 ; xs_arg = - 1 ; c_arg = - 1
        DO i = optional_arg, nrhs
          i_in = prhs( i )

!  Input optional control data

          IF ( mxIsStruct( i_in ) ) THEN ! input control
            c_arg = i
            c_in = i_in
            s_len = slen
            CALL SLLSB_matlab_control_set( c_in, control, s_len )

!  Input optional array argument

          ELSE IF ( mxIsClass( i_in, 'double') ) THEN
            IF ( co_arg > 0 .AND. w_arg > 0 .AND. xs_arg > 0 )                 &
              CALL mexErrMsgTxt(                                               &
              ' array input arguments cohort, w and x_s already provided' )
            n_v = INT( mxGetN( i_in ) )
            IF ( p%n /= p%o ) THEN
              IF ( n_v == p%o ) THEN  ! argument is w
                array = 'w'
              ELSE IF ( n_v == p%n ) THEN ! argument is x_s
                IF ( co_arg < 0 ) THEN
                  array = 'c'
                ELSE
                  array = 'x'
                END IF
              ELSE ! argument not recognised
                CALL mexErrMsgTxt( ' array input argument not recognised' )
              END IF              
            ELSE
              IF ( n_v == p%n ) THEN  ! argument is x_s or w
                IF ( co_arg < 0 ) THEN
                  array = 'c'
                ELSE IF ( w_arg < 0 ) THEN
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

!  Input cohort

            IF ( array == 'c' ) THEN
              co_arg = i
              co_in = i_in
              co_pr = mxGetPr( co_in )

!  kludge to avoid Mex incorrect type error for supposed integers

              ALLOCATE( real_cohort( p%n ), STAT = info )
              CALL MATLAB_copy_from_ptr( co_pr, real_cohort, p%n )
              ALLOCATE( p%COHORT( p%n ), STAT = info )
              p%COHORT( : p%n ) = INT( real_cohort, KIND = KIND( p%n ) )
              DEALLOCATE( real_cohort, STAT = info )
              p%m = MAXVAL( p%COHORT( : p%n ) )

!  Input w

            ELSE IF ( array == 'w' ) THEN
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
          filename = "output_sllsb." // TRIM( char_output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_sllsb." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL SLLSB_matlab_inform_create( plhs( i_arg ), SLLSB_pointer )

!  Allocate space for the solution

        ALLOCATE( p%X( p%n ), p%X_status( p%n ), STAT = info )
        p%X = 0.0_wp 

!  Solve the simplex-constrained least-squares problem

        CALL SLLSB_solve( p, data, control, inform )

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

        CALL SLLSB_matlab_inform_get( inform, SLLSB_pointer )

!  if required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1, 1, naux, faux )

          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'y', p%m, aux_y_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'z', p%n, aux_z_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'r', p%o, aux_r_pr )
          CALL MATLAB_create_integer_component( plhs( aux_arg ),               &
            'x_status', p%n, aux_x_status_pr )

!  copy the values

          y_pr = mxGetPr( aux_y_pr )
          CALL MATLAB_copy_to_ptr( p%Y, y_pr, p%m )
          z_pr = mxGetPr( aux_z_pr )
          CALL MATLAB_copy_to_ptr( p%Z, z_pr, p%n )
          r_pr = mxGetPr( aux_r_pr )
          CALL MATLAB_copy_to_ptr( p%R, r_pr, p%o )
          x_status_pr = mxGetPr( aux_x_status_pr )
          CALL MATLAB_copy_to_ptr( p%X_status, x_status_pr, p%n )

        END IF

!  Check for errors

        IF ( inform%status < 0 )                                              &
          CALL mexErrMsgTxt( ' Call to SLLSB_solve failed ' )
      END IF

!      WRITE( message, * ) inform%status
!      CALL MEXPRINTF( TRIM( message ) // char( 13 ) )

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( p%Ao%row ) ) DEALLOCATE( p%Ao%row, STAT = info )
        IF ( ALLOCATED( p%Ao%col ) ) DEALLOCATE( p%Ao%col, STAT = info )
        IF ( ALLOCATED( p%Ao%val ) ) DEALLOCATE( p%Ao%val, STAT = info )
        IF ( ALLOCATED( p%B ) ) DEALLOCATE( p%B, STAT = info )
        IF ( ALLOCATED( p%X ) ) DEALLOCATE( p%X, STAT = info )
        IF ( ALLOCATED( p%Y ) ) DEALLOCATE( p%Y, STAT = info )
        IF ( ALLOCATED( p%Z ) ) DEALLOCATE( p%Z, STAT = info )
        IF ( ALLOCATED( p%R ) ) DEALLOCATE( p%R, STAT = info )
        IF ( ALLOCATED( p%G ) ) DEALLOCATE( p%G, STAT = info )
        IF ( ALLOCATED( p%X_status ) ) DEALLOCATE( p%X_status, STAT = info )
        IF ( ALLOCATED( p%COHORT ) ) DEALLOCATE( p%COHORT, STAT = info )
        IF ( ALLOCATED( p%W ) ) DEALLOCATE( p%W, STAT = info )
        IF ( ALLOCATED( p%X_s ) ) DEALLOCATE( p%X_S, STAT = info )
        CALL SLLSB_terminate( data, control, inform )
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
