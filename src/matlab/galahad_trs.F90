#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_TRS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix H (and possibly M), optionally an
!  m by n matrix A, an n-vector g, a constant f, and a scalar radius,
!  find the solution of the TRUST-REGION subproblem
!    minimize 0.5 * x' * H * x + c' * x + f
!    subject to ||x||_M <= radius (and perhaps Ax=0).
!  Here ||x||_M^2 = x' * M * x and M is diagonally dominant; if M is
!  not given, M=I and ||x||_M is thus taken to be the Euclidean (l_2-)norm
!  sqrt(x' * x). H need not be definite. Advantage is taken of sparse A and H.
!
!  Simple usage -
!
!  to solve the trust-region subproblem in the Euclidean norm
!   [ x, inform ]
!     = galahad_trs( H, c, f, radius, control, M, A )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!     = galahad_trs( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, inform ]
!     = galahad_trs( 'existing', H, c, f, radius, control, M, A )
!
!  to remove data structures after solution
!   galahad_trs( 'final' )
!
!  Usual Input -
!          H: the symmetric n by n matrix H
!          c: the n-vector c
!          f: the scalar f
!     radius: the trust-region radius (radius>0)
!
!  Optional Input -
!    control: a structure containing control parameters.
!            The components are of the form control.value, where
!            value is the name of the corresponding component of
!            the derived type TRS_control_type as described in
!            the manual for the fortran 90 package GALAHAD_TRS.
!            See: http://galahad.rl.ac.uk/galahad-www/doc/trs.pdf
!          M: the n by n symmetric, diagonally-dominant matrix M
!          A: the m by n matrix A
!
!  Usual Output -
!          x: the global minimizer
!
!  Optional Output -
!   control: see above. Returned values are the defaults
!   inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of the
!      derived type TRS_inform_type as described in the manual
!      for the fortran 90 package GALAHAD_TRS. The components
!      of inform.time, inform.history, inform.IR_inform and
!      inform.SLS_inform are themselves structures, holding the
!      components of the derived types TRS_time_type, TRS_history_type,
!      IR_inform_type and SLS_inform_type, respectively.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/trs.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.3.1. February 18th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_TRS_MATLAB_TYPES
      USE GALAHAD_TRS_double
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
      INTEGER * 4 :: i4, n
      mwSize :: h_arg, c_arg, f_arg, radius_arg, con_arg, a_arg, m_arg
      mwSize :: x_arg, i_arg, s_len

      mwPointer :: h_in, c_in, f_in, radius_in, g_in, m_in, a_in
      mwPointer :: x_pr, g_pr, f_pr, radius_pr

      INTEGER, PARAMETER :: history_max = 100
      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: opened, initial_set = .FALSE.
      INTEGER :: iores
      CHARACTER ( len = 8 ) :: mode
      TYPE ( TRS_pointer_type ) :: TRS_pointer
      mwPointer, ALLOCATABLE :: col_ptr( : )

!  arguments for TRS

      REAL ( KIND = wp ) :: radius, f
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, X
      TYPE ( SMT_type ) :: H, M, A
      TYPE ( TRS_data_type ), SAVE :: data
      TYPE ( TRS_control_type ), SAVE :: control
      TYPE ( TRS_inform_type ) :: inform

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_trs requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_trs' )
          h_arg = 2 ; c_arg = 3 ; f_arg = 4 ; radius_arg = 5
          con_arg = 6 ; m_arg = 7 ; a_arg = 8
          x_arg = 1 ; i_arg = 2
          IF ( nrhs > a_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_trs' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_trs' )
        h_arg = 1 ; c_arg = 2 ; f_arg = 3 ; radius_arg = 4
        con_arg = 5 ; m_arg = 6 ; a_arg = 7
        x_arg = 1 ; i_arg = 2
        IF ( nrhs > a_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_trs' )
      END IF

      IF ( nlhs > 2 )                                                          &
        CALL mexErrMsgTxt( ' galahad_trs provides at most 2 output arguments' )

!  Initialize the internal structures for trs

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL TRS_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL TRS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that TRS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the control argument is present, extract the input control data

        s_len = slen
        IF ( nrhs >= con_arg ) THEN
          c_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL TRS_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_trs." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_trs." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        CALL TRS_matlab_inform_create( plhs( i_arg ), TRS_pointer )

!  Import the problem data

!  input H

        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, H, col_ptr, .TRUE. )
        n = H%n

!  input M

        IF ( nrhs >= m_arg ) THEN
          m_in = prhs( m_arg )
          IF ( mxIsNumeric( m_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a matrix M ' )
          CALL MATLAB_transfer_matrix( m_in, M, col_ptr, .TRUE. )
          IF ( M%n /= n )                                                      &
            CALL mexErrMsgTxt( ' Dimensions of H and M must agree' )
        END IF

!  input A

        IF ( nrhs >= a_arg ) THEN
          a_in = prhs( a_arg )
          IF ( mxIsNumeric( a_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a matrix A ' )
          CALL MATLAB_transfer_matrix( a_in, A, col_ptr, .FALSE. )
          IF ( A%n /= n )                                                      &
            CALL mexErrMsgTxt( ' Column dimensions of H and A must agree' )
        END IF

!  Allocate space for input vector C

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )
        ALLOCATE( C( n ), STAT = info )

!  Input c

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

!  Input radius

        radius_in = prhs( radius_arg )
        IF ( mxIsNumeric( radius_in ) == 0 )                                   &
          CALL mexErrMsgTxt( ' There must a scalar radius ' )
        radius_pr = mxGetPr( radius_in )
        CALL MATLAB_copy_from_ptr( radius_pr, radius )

!  Allocate space for the solution

        ALLOCATE( X( n ), STAT = info )

!  Solve the problem

        IF ( nrhs >= a_arg ) THEN
          CALL TRS_solve( n, radius, f, C, H, X, data, control, inform,        &
                          M = M, A = A )
        ELSE IF ( nrhs >= m_arg ) THEN
          CALL TRS_solve( n, radius, f, C, H, X, data, control, inform,        &
                          M = M )
        ELSE
          CALL TRS_solve( n, radius, f, C, H, X, data, control, inform )
        END IF

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

!  Record output information

         CALL TRS_matlab_inform_get( inform, TRS_pointer )

!  Check for errors

        IF ( inform%status < 0 )                                               &
          CALL mexErrMsgTxt( ' Call to TRS_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( H%row ) ) DEALLOCATE( H%row, STAT = info )
        IF ( ALLOCATED( H%col ) ) DEALLOCATE( H%col, STAT = info )
        IF ( ALLOCATED( H%val ) ) DEALLOCATE( H%val, STAT = info )
        IF ( ALLOCATED( M%row ) ) DEALLOCATE( M%row, STAT = info )
        IF ( ALLOCATED( M%col ) ) DEALLOCATE( M%col, STAT = info )
        IF ( ALLOCATED( M%val ) ) DEALLOCATE( M%val, STAT = info )
        IF ( ALLOCATED( A%row ) ) DEALLOCATE( A%row, STAT = info )
        IF ( ALLOCATED( A%col ) ) DEALLOCATE( A%col, STAT = info )
        IF ( ALLOCATED( A%val ) ) DEALLOCATE( A%val, STAT = info )
        IF ( ALLOCATED( C ) ) DEALLOCATE( C, STAT = info )
        IF ( ALLOCATED( X ) ) DEALLOCATE( X, STAT = info )
        CALL TRS_terminate( data, control, inform )
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
