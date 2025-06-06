#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 26/08/2018 AT 14:20 GMT.

!-*-*-*- G A L A H A D _ T R A N S F E R  _ M A T L A B  M O D U L E -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 12th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_TRANSFER_MATLAB

      USE GALAHAD_SMT_double
      USE GALAHAD_SPACE_double
      USE GALAHAD_MATLAB

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: MATLAB_transfer_matrix

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
      INTEGER * 4, PARAMETER :: i4_1 = 1
      INTEGER, PARAMETER :: int4 = KIND( i4_1 )

    CONTAINS

!  -*-*-*-  M A T L A B _ t r a n s f e r _ m a t r i x  -*-*-*-*-

      SUBROUTINE MATLAB_transfer_matrix( a_in, A, col_ptr, symmetric )

!  -----------------------------------------------

!  Transfer the matrix A from Matlab to SMT format

!  Arguments

!  a_in - pointer to matrix structure holding A
!  A - SMT structure for Fortran A
!  col_ptr - allocatable workspace
!  symmetric - true if A is considered symmetric

!  -----------------------------------------------

      mwPointer :: a_in
      TYPE ( SMT_type ) :: A
      mwPointer, ALLOCATABLE :: col_ptr( : )
      mwPointer, ALLOCATABLE :: row_ptr( : )
      LOGICAL :: symmetric

!  local variables

      INTEGER :: i, j, k, l, info
      INTEGER * 4 :: stat, np1, status, alloc_status
      mwPointer :: a_cpr_pr, a_row_pr, val_pr
      mwPointer :: mxGetPr
      mwSize :: mxGetM, mxGetN, mxGetNzmax
!     mwSize :: mxGetIr, mxGetJc
      mwPointer :: mxGetIr, mxGetJc
      LOGICAL :: mxIsSparse

     INTEGER :: iores
     LOGICAL :: filexx
     CHARACTER ( len = 80 ) :: filename
!    INTEGER ::  mexPrintf
!    CHARACTER ( LEN = 200 ) :: str

!  Get the row and column dimensions

      A%m = INT( mxGetM( a_in ), KIND = int4 )
      A%n = INT( mxGetN( a_in ), KIND = int4 )

      IF ( symmetric .AND. A%m /= A%n )                                        &
        CALL mexErrMsgTxt( ' The matrix must be square ' )

!  Set up the structure to hold A in co-ordinate form

      IF ( mxIsSparse( a_in ) ) THEN
        A%ne = INT( mxGetNzmax( a_in ), KIND = int4 )
      ELSE
        A%ne = A%m * A%n
      END IF
      CALL SMT_put( A%type, 'COORDINATE', stat )

!  Allocate space for the input matrix A

      IF ( ALLOCATED( row_ptr ) ) DEALLOCATE( row_ptr )
      ALLOCATE( row_ptr(  A%ne ) )
!     CALL SPACE_resize_array( A%ne, row_ptr, status, alloc_status )
!     IF ( status /= 0 ) CALL mexErrMsgTxt( ' allocate error row_ptr' )

      CALL SPACE_resize_array( A%ne, A%row, status, alloc_status )
      IF ( status /= 0 ) CALL mexErrMsgTxt( ' allocate error A%row' )
      CALL SPACE_resize_array( A%ne, A%col, status, alloc_status )
      IF ( status /= 0 ) CALL mexErrMsgTxt( ' allocate error A%col' )
      CALL SPACE_resize_array( A%ne, A%val, status, alloc_status )
      IF ( status /= 0 ) CALL mexErrMsgTxt( ' allocate error A%val' )

      IF ( mxIsSparse( a_in ) ) THEN

!  allocate temporary workspace

        np1 = A%n + i4_1
        IF ( ALLOCATED( col_ptr ) ) THEN
          IF ( SIZE( col_ptr ) < np1 ) THEN
            DEALLOCATE( col_ptr, STAT = info )
            IF ( info /= 0 ) CALL mexErrMsgTxt( ' deallocate error col_ptr' )
            ALLOCATE( col_ptr( np1 ), STAT = info )
            IF ( info /= 0 ) CALL mexErrMsgTxt( ' allocate error col_ptr' )
          END IF
        ELSE
          ALLOCATE( col_ptr( np1 ), STAT = info )
          IF ( info /= 0 ) CALL mexErrMsgTxt( ' allocate error col_ptr' )
        END IF

!  Copy the integer components of A into co-ordinate form.
!  ** N.B. indices start at 0 in C **

        a_row_pr = mxGetIr( a_in )
        a_cpr_pr = mxGetJc( a_in )

!--------------open print from fortran--------
filename = "output_galahad.89"
INQUIRE( FILE = filename, EXIST = filexx )
IF ( filexx ) THEN
   OPEN( 89, FILE = filename, FORM = 'FORMATTED', &
          STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
ELSE
   OPEN( 89, FILE = filename, FORM = 'FORMATTED', &
           STATUS = 'NEW', IOSTAT = iores )
END IF
!--------------open print---------------------

        CALL galmxCopyPtrToInteger44( a_row_pr, A%row, A%ne )
        CALL galmxCopyPtrToInteger84( a_row_pr, row_ptr, A%ne )
        CALL galmxCopyPtrToInteger84( a_cpr_pr, col_ptr, np1 )

        col_ptr = col_ptr + 1
        A%row = A%row + i4_1

WRITE(89, "(' n ', I0 )" ) A%n
WRITE(89, "(' a_row ', /, 6( 1X,  I11 ) )" ) A%row( :  A%ne )
WRITE(89, "(' row_ptr ', /, 6( 1X,  I11 ) )" ) row_ptr( :  A%ne )
WRITE(89, "(' col_ptr ', /, 6( 1X,  I11 ) )" ) col_ptr( :  A%n + 1 )

!--------------print---------------------
!REWIND( 89, err = 500 )
!DO
!  READ( 89, "( A )", end = 500 ) str
!  i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
!END DO
!500 CONTINUE
!--------------print---------------------

        DO i = 1, A%n
          DO j = col_ptr( i ), col_ptr( i + 1 ) - 1
            A%col( j ) = INT( i, KIND = int4 )
          END DO
        END DO
        A%ne = INT( col_ptr( A%n + 1 ) - 1, KIND = int4 )

!  Set the row and column indices if the matrix is dense

      ELSE
        l = 0
        DO j = 1, A%n
          DO i = 1, A%m
            l = l + 1
            A%row( l ) = INT( i, KIND = int4 )
            A%col( l ) = INT( j, KIND = int4 )
          END DO
        END DO
        A%ne = INT( l, KIND = int4 )
      END IF

!  copy the real components of A

      val_pr = mxGetPr( a_in )
      CALL MATLAB_copy_from_ptr( val_pr, A%val, A%ne )

!  remove the lower triangle if the matrix is symmetric

      IF ( symmetric ) THEN
        l = 0
        DO k = 1, A%ne
          i = A%row( k )
          j = A%col( k )
          IF ( i <= j ) THEN
            l = l + 1
            A%row( l ) = INT( i, KIND = int4 )
            A%col( l ) = INT( j, KIND = int4 )
            A%val( l ) = A%val( k )
          END IF
        END DO
        A%ne = INT( l, KIND = int4 )
      END IF

!  remove zeros

      l = 0
      DO k = 1, A%ne
        IF ( A%val( k ) /= 0.0_wp ) THEN
          l = l + 1
          A%row( l ) = A%row( k )
          A%col( l ) = A%col( k )
          A%val( l ) = A%val( k )
        END IF
      END DO
      A%ne = INT( l, KIND = int4 )

!-------------- more print---------------
!BACKSPACE(89)
WRITE(89, "(' coordinate' )" )
WRITE(89, "(' n, ne ', I0, 1X, I0 )" ) A%n, A%ne
WRITE(89, "(' a_row, col, val ', /, 2( 1X,  2I11, ES12.4 ) )" ) &
( A%row(i),a%col(i),a%val(i), i = 1, A%ne )
CLOSE(89)
!--------------print---------------------

      RETURN

!  End of SUBROUTINE MATLAB_transfer_matrix

      END SUBROUTINE MATLAB_transfer_matrix

!-*- E N D  o f  G A L A H A D _ T R A N S F E R  _ M A T L A B  M O D U L E -*-

    END MODULE GALAHAD_TRANSFER_MATLAB
