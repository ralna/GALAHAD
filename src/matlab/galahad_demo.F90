#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 26/02/2010 AT 14:00 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO TEST GALAHAD_MATLAB
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!   [ control ] 
!     = galahad_demo( 'initial' )
!  or
!   [ control ] 
!     = galahad_demo( H )
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4 February 12th 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
      USE GALAHAD_TRANSFER_MATLAB
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

      INTEGER :: i, m, n, info
      mwSize :: h_arg, c_arg, i_arg, s_len, k

      mwPointer :: h_in

      INTEGER, PARAMETER :: history_max = 100
      CHARACTER ( len = 80 ) :: output_unit, filename
      LOGICAL :: filexx, opened, initial_set = .FALSE.
      CHARACTER ( len = 18 ) :: mode
      mwPointer, ALLOCATABLE :: col_ptr( : )
!     CHARACTER ( len = 80 ) :: message
      TYPE ( SMT_type ) :: H

!  Test input/output arguments

      i = mexPrintf( "hello\n" )

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_demos requires at least 1 input argument' )

      IF ( .NOT. mxIsChar( prhs( 1 ) ) )                                       &
        CALL mexErrMsgTxt( ' first argument must be a string' )
i = mexPrintf( "hello\n" )


!  interpret the first argument


      k = 17
      i = mxGetString( prhs( 1 ), mode, k )
!i = mexPrintf( "hello %-18s\n", mode )
!RETURN

!  initial entry

      IF ( TRIM( mode ) == 'initial' ) THEN

        c_arg = 1
        IF ( nlhs > c_arg )                                                    &
          CALL mexErrMsgTxt( ' too many output arguments required' )

!  form_and_factorize entry

      ELSE
        h_arg = 2
        IF ( nrhs > h_arg )                                                    &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_demo' )
        i_arg = 1
        IF ( nlhs > i_arg )                                                    &
          CALL mexErrMsgTxt( ' too many output arguments required' )

!  input H

        h_in = prhs( h_arg )
        IF ( mxIsNumeric( h_in ) == 0 )                                        &
          CALL mexErrMsgTxt( ' There must be a matrix H ' )
        CALL MATLAB_transfer_matrix( h_in, H, col_ptr, .TRUE. )
        n = H%n

!      ELSE
!        CALL mexErrMsgTxt( ' Unrecognised first input string ' )
      END IF

!  Check for errors

!     IF ( inform%status < 0 )                                                &
!        CALL mexErrMsgTxt( ' Call to demo failed ' )

!      WRITE( message, * ) 'here'
!      i = MEXPRINTF( TRIM( message ) // char( 13 ) )

      RETURN

      END SUBROUTINE mexFunction
