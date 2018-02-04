#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 09/03/2010 AT 08:05 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO HSL_MA57
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  Given a symmetric n by n matrix A and an n-vector b or an n by r 
!  matrix B, solve the system A x = b or the system AX=B. The matrix 
!  A need not be definite. Advantage is taken of sparse A. Options
!  are provided to factorize a matrix A without solving the system, 
!  and to solve systems using previously-determined factors.
!
!  Simple usage -
!
!  to solve a system Ax=b or AX=B
!   [ x, inform ] = galahad_sils( A, b, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to factorization
!   [ control ] 
!     = galahad_sils( 'initial' )
!
!  to factorize A
!   [ inform ] = galahad_sils( 'factor', A, control )
!
!  to solve Ax=b or AX=B using existing factors
!   [ x, inform ] = galahad_sils( 'solve', b )
!
!  to remove data structures after solution
!   galahad_sils( 'final' )
!
!  Usual Input -
!    A: the symmetric matrix A
!    b a column vector b or matrix of right-hand sides B
!
!  Optional Input -
!    control, a structure containing control parameters.
!      The components are of the form control.value, where
!      value is the name of the corresponding component of
!      the derived type MA57_CONTROL as described in the 
!      manual for the fortran 90 package HSL_MA57.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sils.pdf
!
!  Usual Output -
!   x: the vector of solutions to Ax=b or matrix of solutions to AX=B
!
!  Optional Output -
!    control: see above. Returned values are the defaults
!    inform: a structure containing information parameters
!      The components are of the form inform.value, where
!      value is the name of the corresponding component of
!      the derived type MA57_AINFO/FINFO/SINFO as described 
!      in the manual for the fortran 90 package HSL_MA57.
!      See: http://galahad.rl.ac.uk/galahad-www/doc/sils.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.1. July 4th 2007

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
      USE HSL_MA57_MATLAB_TYPES
      USE HSL_ZD11_double
      USE HSL_MA57_DOUBLE
      IMPLICIT NONE
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

! ------------------------- Do not change -------------------------------

!  Keep the above subroutine, argument, and function declarations for use
!  in all your fortran mex files.
!
      INTEGER * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )

      INTEGER, PARAMETER :: slen = 30
      CHARACTER ( len = 16 ) :: fi, mxGetFieldNameByNumber
      LOGICAL :: mxIsSparse, mxIsStruct, mxIsChar
      mwSize :: mxGetString
      mwSize :: mxGetM, mxGetN, mxGetIr, mxGetJc
      mwSize :: mxIsNumeric, mxGetNzmax, mxGetNumberOfFields
      mwPointer :: mxCreateStructMatrix, mxGetPr

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      mwSize :: i, j, k, l, n, nb, nfields, info
      mwSize :: a_arg, b_arg, c_arg, x_arg, i_arg
      mwPointer :: a_in, b_in, c_in, val_in, rhs_in, cpr_in, row_in, cn_in, x_pr
      mwSize :: s_len

      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      INTEGER :: iores

      CHARACTER ( len = 7 ) :: mode
      TYPE ( MA57_pointer_type ) :: MA57_pointer
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, X
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: B2, X2

!  arguments for MA57

      TYPE ( ZD11_type ), SAVE :: A
      TYPE ( MA57_control ), SAVE :: CONTROL
      TYPE ( MA57_ainfo ), SAVE :: AINFO
      TYPE ( MA57_finfo ), SAVE :: FINFO
      TYPE ( MA57_sinfo ), SAVE :: SINFO
      TYPE ( MA57_factors ), SAVE :: FACTORS

      mwPointer, ALLOCATABLE :: col_ptr( : )

!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' sils requires at least 1 input argument' )

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 7 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 2 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to sils' )
          IF ( TRIM( mode ) == 'factor' ) THEN
            a_arg = 2
            c_arg = 3
            i_arg = 1
          ELSE IF ( TRIM( mode ) == 'solve' ) THEN
            b_arg = 2
            c_arg = 3
            x_arg = 1
            i_arg = 2
!         ELSE IF ( TRIM( mode ) == 'all' ) THEN
          ELSE
            a_arg = 2
            b_arg = 3
            c_arg = 4
            x_arg = 1
            i_arg = 2
          END IF
          IF ( nrhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' Too many input arguments to sils' )
        END IF
      ELSE
        mode = 'all'
        IF ( nrhs < 2 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to sils' )
        a_arg = 1
        b_arg = 2
        c_arg = 3
        x_arg = 1
        i_arg = 2
        IF ( nrhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to sils' )
      END IF

      IF ( nlhs > 2 )                                                          &
        CALL mexErrMsgTxt( ' sils provides at most 2 output arguments' )

!  Initialize the internal structures for sils

      IF ( TRIM( mode ) == 'initial' .OR.    &
           TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL MA57_INITIALIZE( FACTORS, CONTROL )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  If required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL MA57_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  Check that MA57_initialize has been called 

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  If the third argument is present, extract the input control data

        s_len = slen
        IF ( nrhs == c_arg ) THEN
          c_in = prhs( c_arg )
          IF ( .NOT. mxIsStruct( c_in ) )                                      &
            CALL mexErrMsgTxt( ' last input argument must be a structure' )
          CALL MA57_matlab_control_set( c_in, control, s_len )
        END IF

!  Open i/o units

        IF ( CONTROL%lp > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) CONTROL%lp
          filename = "output_sils." // TRIM( output_unit ) 
          INQUIRE( FILE = filename, EXIST = filexx )
          IF ( filexx ) THEN
             OPEN( CONTROL%lp, FILE = filename, FORM = 'FORMATTED',            &
                    STATUS = 'OLD', IOSTAT = iores )
          ELSE
             OPEN( CONTROL%lp, FILE = filename, FORM = 'FORMATTED',            &
                     STATUS = 'NEW', IOSTAT = iores )
          END IF
        END IF

        IF ( CONTROL%wp > 0 ) THEN
          INQUIRE( CONTROL%wp, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) CONTROL%wp
            filename = "output_sils." // TRIM( output_unit ) 
            INQUIRE( FILE = filename, EXIST = filexx )
            IF ( filexx ) THEN
               OPEN( CONTROL%wp, FILE = filename, FORM = 'FORMATTED',          &
                      STATUS = 'OLD', IOSTAT = iores )
            ELSE
               OPEN( CONTROL%wp, FILE = filename, FORM = 'FORMATTED',          &
                       STATUS = 'NEW', IOSTAT = iores )
            END IF
          END IF
        END IF

        IF ( CONTROL%mp > 0 ) THEN
          INQUIRE( CONTROL%mp, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) CONTROL%mp
            filename = "output_sils." // TRIM( output_unit ) 
            INQUIRE( FILE = filename, EXIST = filexx )
            IF ( filexx ) THEN
               OPEN( CONTROL%mp, FILE = filename, FORM = 'FORMATTED',          &
                      STATUS = 'OLD', IOSTAT = iores )
            ELSE
               OPEN( CONTROL%mp, FILE = filename, FORM = 'FORMATTED',          &
                       STATUS = 'NEW', IOSTAT = iores )
            END IF
          END IF
        END IF

        IF ( CONTROL%sp > 0 ) THEN
          INQUIRE( CONTROL%sp, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) CONTROL%sp
            filename = "output_sils." // TRIM( output_unit ) 
            INQUIRE( FILE = filename, EXIST = filexx )
            IF ( filexx ) THEN
               OPEN( CONTROL%sp, FILE = filename, FORM = 'FORMATTED',          &
                      STATUS = 'OLD', IOSTAT = iores )
            ELSE
               OPEN( CONTROL%sp, FILE = filename, FORM = 'FORMATTED',          &
                       STATUS = 'NEW', IOSTAT = iores )
            END IF
          END IF
        END IF

!  Create inform output structure

        CALL MA57_matlab_inform_create( plhs( i_arg ), MA57_pointer )

!  Factorization phase

        IF ( TRIM( mode ) == 'factor' .OR. TRIM( mode ) == 'all' ) THEN

!  Check to ensure the input is a number

          a_in = prhs( a_arg )
          IF ( mxIsNumeric( a_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a matrix A ' )
          CALL MATLAB_transfer_matrix( a_in, A, col_ptr, .FALSE. )

        IF ( ALLOCATED( col_ptr ) ) DEALLOCATE( col_ptr, STAT = info )

!  Analyse

          CALL MA57_ANALYSE( A, FACTORS, CONTROL, AINFO )

!  Record output information

          CALL MA57_matlab_inform_get_ainfo( AINFO, MA57_pointer )

!  Check for errors

          IF ( AINFO%FLAG < 0 )                                                &
            CALL mexErrMsgTxt( ' Call to MA57_analyse failed ' )

!  Factorize

          CALL MA57_FACTORIZE( A, FACTORS, CONTROL, FINFO )

!  Record output information

          CALL MA57_matlab_inform_get_finfo( FINFO, MA57_pointer )

!  Check for errors

          IF ( FINFO%FLAG < 0 )                                                &
            CALL mexErrMsgTxt( ' Call to MA57_factorize failed ' )
        END IF

!  Solve phase

        IF ( TRIM( mode ) == 'solve' .OR. TRIM( mode ) == 'all' ) THEN

          IF ( .NOT. ( ALLOCATED( A%row ) .AND.                                &
                       ALLOCATED( A%col ) .AND.                                &
                       ALLOCATED( A%val ) ) )                                  &
            CALL mexErrMsgTxt( ' There must be existing factors ' )

!  Check to ensure the input is a number

          b_in = prhs( b_arg )
          IF ( mxIsNumeric( b_in ) == 0 )                                      &
            CALL mexErrMsgTxt( ' There must be a right-hand-side b ' )

 !  Allocate space for the right-hand side and solution

          n = mxGetM( b_in )
          IF ( A%n /= n )                                                      &
            CALL mexErrMsgTxt( ' A and b/B must have compatible dimensions ' )

          nb = mxGetN( b_in )
          rhs_in = mxGetPr( b_in )

!  one right-hand side

          IF ( nb == 1 ) THEN
            ALLOCATE( B( n ), X( n ) )
            CALL MATLAB_copy_from_ptr( rhs_in, B, n )

!  Solve without refinement

            X = B
            CALL MA57_SOLVE( A, FACTORS, X, CONTROL, SINFO )

!  Perform one refinement

            CALL MA57_SOLVE( A, FACTORS, X, CONTROL, SINFO, B )

!  multiple right-hand sides

          ELSE
            ALLOCATE( B2( n, nb ), X2( n, nb ) )
            CALL MATLAB_copy_from_ptr( rhs_in, B2, n, nb )

!  Solve without refinement

            X2 = B2
            CALL MA57_SOLVE( A, FACTORS, X2, CONTROL, SINFO )

!  Perform one refinement

!           CALL MA57_SOLVE( A, FACTORS, X2, CONTROL, SINFO, B2 )

          END IF

!  Print details to Matlab window

         IF ( control%mp > 0 ) THEN
           REWIND( control%mp, err = 500 )
            DO
              READ( control%mp, "( A )", end = 500 ) str
              i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
            END DO
          END IF
   500   CONTINUE

!  Output solution

          plhs( x_arg ) = MATLAB_create_real( n, nb )
          x_pr = mxGetPr( plhs( x_arg ) )
          IF ( nb == 1 ) THEN
            CALL MATLAB_copy_to_ptr( X, x_pr, n )     
          ELSE
            CALL MATLAB_copy_to_ptr( X2, x_pr, n, nb )     
          END IF

!  Record output information

          CALL MA57_matlab_inform_get_sinfo( SINFO, MA57_pointer )

        END IF
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
         IF ( ALLOCATED( A%row ) ) DEALLOCATE( A%row, STAT = info )
         IF ( ALLOCATED( A%col ) ) DEALLOCATE( A%col, STAT = info )
         IF ( ALLOCATED( A%val ) ) DEALLOCATE( A%val, STAT = info )
         IF ( ALLOCATED( B ) ) DEALLOCATE( B, STAT = info )
         IF ( ALLOCATED( B2 ) ) DEALLOCATE( B2, STAT = info )
         IF ( ALLOCATED( X ) ) DEALLOCATE( X, STAT = info )
         IF ( ALLOCATED( X2 ) ) DEALLOCATE( X2, STAT = info )
         CALL MA57_finalize( FACTORS, CONTROL, info )
      END IF

!  close any opened io units

      IF ( CONTROL%lp > 0 ) THEN
         INQUIRE( CONTROL%lp, OPENED = opened )
         IF ( opened ) CLOSE( control%lp )
      END IF

      IF ( CONTROL%wp > 0 ) THEN
         INQUIRE( CONTROL%wp, OPENED = opened )
         IF ( opened ) CLOSE( control%wp )
      END IF

      IF ( CONTROL%mp > 0 ) THEN
         INQUIRE( CONTROL%mp, OPENED = opened )
         IF ( opened ) CLOSE( control%mp )
      END IF

      IF ( CONTROL%sp > 0 ) THEN
         INQUIRE( CONTROL%sp, OPENED = opened )
         IF ( opened ) CLOSE( control%sp )
      END IF

      RETURN
      END SUBROUTINE mexFunction

