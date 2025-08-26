! THIS VERSION: GALAHAD 5.3 - 2025-08-25 AT 09:30 GMT.

#include "ssids_procedures.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ M U   M O D U L E  *-*-*-*-*-*-*-*-*-*-*-

!      ----------------------------------------------------------------
!     | Matrix utility package originally spral_matrix_util from SPRAL |
!      ----------------------------------------------------------------

!  COPYRIGHT (c) 2010, 2013 Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  authors: Jonathan Hogg, John Reid, Jennifer Scott and Sue Thorne
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

      MODULE GALAHAD_MU_precision
        USE GALAHAD_KINDS_precision

        IMPLICIT NONE

        PRIVATE
        PUBLIC :: MU_half_to_full, MU_convert_coord_to_cscl, MU_print_matrix,  &
                  MU_clean_cscl_oop, MU_apply_conversion_map, MU_cscl_verify,  &
                  SSIDS_MATRIX_UNSPECIFIED, SSIDS_MATRIX_REAL_RECT,            &
                  SSIDS_MATRIX_CPLX_RECT, SSIDS_MATRIX_REAL_UNSYM,             &
                  SSIDS_MATRIX_CPLX_UNSYM, SSIDS_MATRIX_REAL_SYM_PSDEF,        &
                  SSIDS_MATRIX_CPLX_HERM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF,  &
                  SSIDS_MATRIX_CPLX_HERM_INDEF, SSIDS_MATRIX_CPLX_SYM,        &
                  SSIDS_MATRIX_REAL_SKEW, SSIDS_MATRIX_CPLX_SKEW

!----------------------
!   P a r a m e t e r s
!----------------------

        REAL (KIND = rp_ ), PARAMETER :: zero = 0.0_rp_

!  the following are all types of matrix we may need to deal with -

!  matrix types : real

!  undefined/known

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_UNSPECIFIED = 0

!  real rectangular

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_REAL_RECT = 1

!  real unsymmetric

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_REAL_UNSYM = 2

!  real sym positive definite

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_REAL_SYM_PSDEF = 3

!  real sym indefinite

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_REAL_SYM_INDEF = 4

!  real skew symmetric

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_REAL_SKEW = 6

!  matrix types : complex

!  complex rectangular

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_CPLX_RECT = - 1

!  complex unsymmetric

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_CPLX_UNSYM = - 2

!  Hermitian positive definite

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_CPLX_HERM_PSDEF = - 3

! Hermitian indefinite

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_CPLX_HERM_INDEF = - 4

!  complex symmetric

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_CPLX_SYM = - 5

!  complex skew symmetric

        INTEGER ( KIND = ip_ ), PARAMETER :: SSIDS_MATRIX_CPLX_SKEW = - 6

!  error flags

        INTEGER ( KIND = ip_ ), PARAMETER :: SUCCESS = 0
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ALLOCATION = - 1
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_MATRIX_TYPE = - 2
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_N_OOR = - 3
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_M_NE_N = - 4
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_PTR_1 = - 5
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_PTR_MONO = - 6
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ROW_BAD_ORDER = - 7
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ROW_OOR = - 8
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ROW_DUP = - 9
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ALL_OOR = - 10
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_MISSING_DIAGONAL = - 11
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_IMAG_DIAGONAL = - 12
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_MISMATCH_LWRUPR = - 13
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_UPR_ENTRY = - 14
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_VAL_MISS = - 15
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_LMAP_MISS = - 16

!  warning flags

        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_IDX_OOR = 1
        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_DUP_IDX = 2
        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_DUP_AND_OOR = 3
        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_MISSING_DIAGONAL = 4
        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_MISS_DIAG_OORDUP = 5

! Possible error returns:

!  ERROR_ALLOCATION         Allocation error
!  ERROR_MATRIX_TYPE        Problem with matrix_type
!  ERROR_N_OOR              n < 0 or m < 0
!  ERROR_PTR_1              ptr( 1 ) < 1
!  ERROR_PTR_MONO           Error in ptr ( not monotonic )
!  ERROR_VAL_MISS           Only one of val_in and val_out is present
!  ERROR_ALL_OOR            All the variables in a column are out-of-range
!  ERROR_IMAG_DIAGONAL      Hermitian case and diagonal not real
!  ERROR_ROW_BAD_ORDER      Entries within a column are not sorted by
!                                increasing row index
!  ERROR_MISMATCH_LWRUPR    Symmetric, skew symmetric or Hermitian:
!                                entries in upper and lower
!                                triangles do not match
!  ERROR_MISSING_DIAGONAL   Pos def and diagonal entries missing
!  ERROR_ROW_OOR            Row contains out-of-range entries
!  ERROR_ROW_DUP            Row contains duplicate entries
!  ERROR_M_NE_N             Square matrix and m .ne. n

!  Possible warnings:

!  WARNING_IDX_OOR          Out-of-range variable indices
!  WARNING_DUP_IDX          Duplicated variable indices
!  WARNING_DUP_AND_OOR      out of range and duplicated indices
!  WARNING_MISSING_DIAGONAL Indefinite case and diagonal entries missing
!  WARNING_MISS_DIAG_OORDUP As WARNING_MISSING_DIAGONAL, and
!                                out-of-range and/or duplicates

!----------------------
!   I n t e r f a c e s
!----------------------

!  applies a map from previous conversion call

        INTERFACE MU_apply_conversion_map
          MODULE PROCEDURE apply_conversion_map_ptr32_precision,               &
                           apply_conversion_map_ptr64_precision
        END INTERFACE MU_apply_conversion_map

!  cleans a CSC-lower matrix out-of-place

        INTERFACE MU_clean_cscl_oop
          MODULE PROCEDURE clean_cscl_oop_ptr32_precision,                     &
                           clean_cscl_oop_ptr64_precision
        END INTERFACE MU_clean_cscl_oop

!  converts a coord matrix to CSC-lower format

        INTERFACE MU_convert_coord_to_cscl
          MODULE PROCEDURE convert_coord_to_cscl_ptr32_precision,              &
                           convert_coord_to_cscl_ptr64_precision
        END INTERFACE MU_convert_coord_to_cscl

!  expands a matrix from CSC-lower to CSC-full

        INTERFACE MU_half_to_full
          MODULE PROCEDURE half_to_full_int32, half_to_full_int64
        END INTERFACE MU_half_to_full

!  verifies if matrix in CSC-lower form

        INTERFACE MU_cscl_verify
          MODULE PROCEDURE cscl_verify_precision
        END INTERFACE MU_cscl_verify

!  pretty-print a CSC-lower matrix ( summary )

        INTERFACE MU_print_matrix
          MODULE PROCEDURE print_matrix_int_precision,                         &
                           print_matrix_long_precision
        END INTERFACE MU_print_matrix

!  internal clean-up dupliactes routine

        INTERFACE cleanup_dup
          MODULE PROCEDURE cleanup_dup32, cleanup_dup64
        END INTERFACE cleanup_dup

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

        TYPE dup_list
          INTEGER ( KIND = i4_ ) :: src
          INTEGER ( KIND = i4_ ) :: dest
          TYPE ( dup_list ), POINTER :: next => null(  )
        END TYPE dup_list

        TYPE dup_list64
          INTEGER ( KIND = i8_ ) :: src
          INTEGER ( KIND = i8_ ) :: dest
          TYPE ( dup_list64 ), POINTER :: next => null(  )
        END TYPE dup_list64

      CONTAINS

!-*-*-  G A L A H A D - c s c l _ v e r i f y   S U B R O U T I N E  -*-*-

        SUBROUTINE cscl_verify_precision( lp, matrix_type, m, n, ptr, row,     &
                                          flag, more, val )

!  to verify that a matrix is in standard format, or identify why it is not

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: lp !  output unit

!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column starts

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: ptr

!  row indices. Entries within each column must be sorted in order of
!  increasing row index. no duplicates and/or out of range entries allowed.

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  futher error information ( or set to 0 )

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: more

!  matrix values,if any

        REAL ( KIND = rp_ ), DIMENSION ( * ), OPTIONAL, INTENT ( IN ) :: val

        INTEGER ( KIND = ip_ ) :: col !  current column
        CHARACTER ( 50 ) :: context !  Procedure name ( used when printing ).
        LOGICAL :: diag !  flag for detection of diagonal
        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = ip_ ) :: j
        INTEGER ( KIND = ip_ ) :: k
        INTEGER ( KIND = ip_ ) :: last !  last row index
        LOGICAL :: lwronly
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: ptr2
        INTEGER ( KIND = ip_ ) :: st

        context = 'cscl_verify'
        flag = SUCCESS

        more = 0 !  ensure more is not undefined.

!  check matrix_type

        SELECT CASE ( matrix_type )
        CASE ( 0 ) !  Undefined matrix. OK, do nothing
        CASE ( 1 : 4, 6 ) !  Real matrix. OK, do nothing
        CASE ( - 6 : - 1 ) !  Complex matrix. Issue error
          flag = ERROR_MATRIX_TYPE
          CALL print_matrix_flag( context, lp, flag )
          RETURN
        CASE DEFAULT !  Out of range value. Issue error
          flag = ERROR_MATRIX_TYPE
          CALL print_matrix_flag( context, lp, flag )
          RETURN
        END SELECT

!  check m and n are valid; skip remaining tests if n=0

        IF ( n < 0 .OR. m < 0 ) THEN
          flag = ERROR_N_OOR
          CALL print_matrix_flag( context, lp, flag )
          RETURN
        END IF
        IF ( ABS( matrix_type ) /= SSIDS_MATRIX_REAL_RECT .AND. m /= n ) THEN
          flag = ERROR_M_NE_N
          CALL print_matrix_flag( context, lp, flag )
          RETURN
        END IF
        IF ( n == 0 .OR. m == 0 ) RETURN

!  check ptr(1) is valid

        IF ( ptr( 1 ) < 1 ) THEN
          more = ptr( 1 )
          flag = ERROR_PTR_1
          CALL print_matrix_flag( context, lp, flag )
          RETURN
        END IF

!  check ptr is monotonically increasing

        DO i = 1, n

 !  ptr is monotonically increasing, good

          IF ( ptr( i + 1 ) >= ptr( i ) ) CYCLE
          flag = ERROR_PTR_MONO
          more = i + 1
          CALL print_matrix_flag( context, lp, flag )
          RETURN
        END DO

!  count number of entries in each row. Also:
!  * Check ordering of entries
!  * Check for out-of-range entries
!  * Check for duplicate entries
!  * Lack of diagonal entries in real pos. def. case.

!  ptr2(k+2) is set to hold the number of entries in row k

        ALLOCATE ( ptr2( m + 2 ), STAT = st )
        IF ( st /= 0 ) GO TO 100
        ptr2( : ) = 0

        lwronly = ABS( matrix_type ) /= SSIDS_MATRIX_UNSPECIFIED .AND.         &
                  ABS( matrix_type ) /= SSIDS_MATRIX_REAL_RECT .AND.           &
                  ABS( matrix_type ) /= SSIDS_MATRIX_REAL_UNSYM
        DO col = 1, n
          last = - 1
          diag = .FALSE.
          DO j = ptr( col ), ptr( col + 1 ) - 1
            k = row( j )
            IF ( k < 1 .OR. k > m ) THEN ! check out-of-range
              flag = ERROR_ROW_OOR
              more = j
              CALL print_matrix_flag( context, lp, flag )
              RETURN
            END IF
            IF ( lwronly .AND. k < col ) THEN
              flag = ERROR_UPR_ENTRY
              more = j
              CALL print_matrix_flag( context, lp, flag )
              RETURN
            END IF
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW .AND.            &
                 k == col ) THEN
              flag = ERROR_UPR_ENTRY
              more = j
              CALL print_matrix_flag( context, lp, flag )
              RETURN
            END IF
            IF ( k == last ) THEN !  check for duplicates
              flag = ERROR_ROW_DUP
              more = j - 1
              CALL print_matrix_flag( context, lp, flag )
              RETURN
            END IF

!  check order

            IF ( k < last ) THEN
              flag = ERROR_ROW_BAD_ORDER
              more = j
              CALL print_matrix_flag( context, lp, flag )
              RETURN
            END IF

!  check for diagonal

            diag = diag .OR. ( k == col )

!  increase count for row k

            ptr2( k + 2 ) = ptr2( k + 2 ) + 1

!  store value for next loop
            last = k
          END DO

!  If marked as positive definite, check if diagonal was present

          IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF .AND.         &
               .NOT. diag ) THEN
            flag = ERROR_MISSING_DIAGONAL
            more = col
            CALL print_matrix_flag( context, lp, flag )
            RETURN
          END IF
        END DO

        IF ( PRESENT( val ) ) THEN
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF )

!  check for positive diagonal entries

            DO j = 1, n
              k = ptr( j )

!  note: column cannot be empty as previously checked pattern ok

              IF ( REAL( val( k ) ) <= zero ) THEN
                flag = ERROR_MISSING_DIAGONAL
                more = j
                CALL print_matrix_flag( context, lp, flag )
                RETURN
              END IF
            END DO
          END SELECT
        END IF
        RETURN

100     CONTINUE !  allocation error
        flag = ERROR_ALLOCATION
        more = st
        CALL print_matrix_flag( context, lp, flag )
        RETURN

        END SUBROUTINE cscl_verify_precision

!-*  G A L A H A D -  p r i n t _ m a t r i x _ l o n g   S U B R O U T I N E  -

        SUBROUTINE print_matrix_long_precision( lp, lines, matrix_type, m, n,  &
                                                ptr, row, val, cbase )

!  pretty prints a matrix as best it can

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: lp !  unit to print on

!  max number of lines to use ( ignored -ive )

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: lines
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type !  type of matrix
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows in matrix
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols in matrix

!  column pointers

        INTEGER ( KIND = i8_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr

!  row indices

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row

!  matrix values

        REAL ( KIND = rp_ ), DIMENSION ( * ), OPTIONAL, INTENT ( IN ) :: val

!  if true, input uses C indexing

        LOGICAL, OPTIONAL, INTENT ( IN ) :: cbase

        INTEGER ( KIND = i4_ ), DIMENSION ( : ), ALLOCATABLE :: ptr32

        ALLOCATE( ptr32( n + 1 ) )

!  assume we are not printing anything huge

        ptr32( 1 : n + 1 ) = INT( ptr( 1 : n + 1 ) ) 

        CALL print_matrix_int_precision( lp, lines, matrix_type, m, n, ptr32,  &
                                         row, val, cbase = cbase )
        RETURN

        END SUBROUTINE print_matrix_long_precision

!-*-  G A L A H A D - p r i n t _ m a t r i x _ i n t   S U B R O U T I N E  -*-

        SUBROUTINE print_matrix_int_precision( lp, lines, matrix_type, m, n,   &
                                               ptr, row, val, cbase )

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: lp !  unit to print on

!  max number of lines to use ( ignored -ive )

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: lines
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type !  type of matrix
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows in matrix
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of cols in matrix

!  column pointers

        INTEGER ( KIND = i4_ ), DIMENSION ( n + 1 ), INTENT ( IN ) :: ptr

!  row indices

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row

!  matrix vals

        REAL ( KIND = rp_ ), DIMENSION ( * ), OPTIONAL, INTENT ( IN ) :: val

!  if true, input uses C indexing

        LOGICAL, OPTIONAL, INTENT ( IN ) :: cbase

        INTEGER ( KIND = ip_ ) :: col, j, k
        INTEGER ( KIND = ip_ ) :: llines
        INTEGER ( KIND = ip_ ), DIMENSION ( :, : ), ALLOCATABLE :: dmat
        CHARACTER ( LEN=5 ) :: mfrmt, nfrmt, nefrmt
        CHARACTER ( LEN=12 ) :: negfrmt, valfrmt, emptyfrmt
        INTEGER ( KIND = ip_ ) :: rebase

        IF ( lp < 0 ) RETURN !  invalid unit

!  check if input is based for C

        rebase = 0
        IF ( PRESENT( cbase ) ) THEN
          IF ( cbase ) rebase = 1
        END IF

!  calculate number of lines to play with

        llines = huge( llines )
        IF ( lines>0 ) llines = lines

!  print a summary statement about the matrix

        mfrmt = digit_format( m )
        nfrmt = digit_format( n )
        nefrmt = digit_format( ptr( n + 1 ) - 1_ip_ )

        SELECT CASE ( matrix_type )
        CASE ( SSIDS_MATRIX_UNSPECIFIED )
          WRITE ( lp, '( A )', ADVANCE = 'no' )                                &
            'Matrix of undefined type, dimension '
        CASE ( SSIDS_MATRIX_REAL_RECT )
          WRITE ( lp, '( A )', ADVANCE = 'no' )                                &
            'Real rectangular matrix, dimension '
        CASE ( SSIDS_MATRIX_REAL_UNSYM )
          WRITE ( lp, '( A )', ADVANCE = 'no' )                                &
            'Real unsymmetric matrix, dimension '
        CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF )
          WRITE ( lp, '( A )', ADVANCE = 'no' )                                &
            'Real symmetric positive definite matrix, dimension '
        CASE ( SSIDS_MATRIX_REAL_SYM_INDEF )
          WRITE ( lp, '( A )', ADVANCE = 'no' )                                &
            'Real symmetric indefinite matrix, dimension '
        CASE ( SSIDS_MATRIX_REAL_SKEW )
          WRITE ( lp, '( A )', ADVANCE = 'no' )                                &
            'Real skew symmetric matrix, dimension '
        CASE DEFAULT
          WRITE ( lp, '( A, I5 )' ) 'Unrecognised matrix_type = ', matrix_type
          RETURN
        END SELECT
        WRITE ( lp, mfrmt, ADVANCE = 'no' ) m
        WRITE ( lp, '( a )', ADVANCE = 'no' ) 'x'
        WRITE ( lp, nfrmt, ADVANCE = 'no' ) n
        WRITE ( lp, '( a )', ADVANCE = 'no' ) ' with '
        WRITE ( lp, nefrmt, ADVANCE = 'no' ) ptr( n + 1 ) - 1 + rebase
        WRITE ( lp, '( a )' ) ' entries.'

!  immediate return if m = 0 or n = 0

        IF ( m == 0 .OR. n == 0 ) RETURN

        IF ( ( ( PRESENT( val ) .AND. n < 10 ) .OR.                            &
               ( .NOT. PRESENT( val ) .AND. n < 24 ) ) .AND.                   &
              m + 1 <= llines ) THEN

!  print the matrix as if dense

          ALLOCATE( dmat( m, n ) )
          dmat( : , : ) = 0
          DO col = 1, n
            DO j = ptr( col ) + rebase, ptr( col + 1 ) + rebase - 1
              k = row( j ) + rebase
              IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                dmat( col, k ) = -j
              END IF
              dmat( k, col ) = j
            END DO
          END DO

          SELECT CASE ( n )
          CASE ( : 6 )
            valfrmt = '(1X,ES12.4)'
            negfrmt = valfrmt
            emptyfrmt = '( 1X,A12 )'
          CASE ( 7 )
            valfrmt = '(1X,ES10.2)'
            negfrmt = valfrmt
            emptyfrmt = '( 1X,A10 )'
          CASE ( 8 : )
            valfrmt = '(1X,ES8.2)'
            negfrmt = '(1X,ES8.1)'
            emptyfrmt = '( 1X,A8 )'
          END SELECT

          DO k = 1, m
            WRITE ( lp, mfrmt, ADVANCE = 'no' ) k - rebase
            WRITE ( lp, '( '':'' )', ADVANCE = 'no' )
            IF ( PRESENT( val ) ) THEN
              DO j = 1, n
                IF ( dmat( k,j ) == 0 ) THEN !  nothing here
                  WRITE ( lp, emptyfrmt, ADVANCE = 'no' )                      &
                    '                         '
                ELSE IF ( dmat( k, j ) > 0 ) THEN
                  IF ( val( dmat( k, j ) ) > zero ) THEN
                    WRITE ( lp, valfrmt, ADVANCE = 'no' ) val( dmat( k, j ) )
                  ELSE
                    WRITE ( lp, negfrmt, ADVANCE = 'no' ) val( dmat( k, j ) )
                  END IF
                ELSE !  in upper triangle
                  SELECT CASE ( matrix_type )
                  CASE ( SSIDS_MATRIX_REAL_SYM_INDEF,                          &
                         SSIDS_MATRIX_REAL_SYM_PSDEF )
                    IF ( val( - dmat( k,j ) ) > zero ) THEN
                      WRITE ( lp, valfrmt, ADVANCE = 'no' )                    &
                        val( - dmat( k, j ) )
                    ELSE
                      WRITE ( lp, negfrmt, ADVANCE = 'no' )                    &
                        val( - dmat( k, j ) )
                    END IF
                  CASE ( SSIDS_MATRIX_REAL_SKEW )
                    IF ( - val( - dmat( k,j ) ) > zero ) THEN
                      WRITE ( lp, valfrmt, ADVANCE = 'no' )                    &
                        - val( - dmat( k, j ) )
                    ELSE
                      WRITE ( lp, negfrmt, ADVANCE = 'no' )                   &
                        - val( - dmat( k, j ) )
                    END IF
                  END SELECT
                END IF
              END DO
            ELSE !  pattern only
              DO j = 1, n
                IF ( dmat( k,j ) == 0 ) THEN !  nothing here
                  WRITE ( lp, '( ''  '' )', ADVANCE = 'no' )
                ELSE
                  WRITE ( lp, '( 1X, ''x'' )', ADVANCE = 'no' )
                END IF
              END DO
            END IF
            WRITE ( lp, '(  )' )
          END DO
        ELSE

!  output first x entries from each column

          llines = llines - 1 !  Discount first info line
          IF ( llines <= 2 ) RETURN
          WRITE ( lp, '( a )' ) 'First 4 entries in columns:'
          llines = llines - 1
          DO col = 1, min( n, llines )
            WRITE ( lp, '( A )', ADVANCE = 'no' ) 'Col '
            WRITE ( lp, nfrmt, ADVANCE = 'no' ) col - rebase
            WRITE ( lp, '( '':'' )', ADVANCE = 'no' )
            DO j = ptr( col ) + rebase,                                        &
                   MIN( ptr( col + 1 ) + rebase - 1, ptr( col ) + rebase + 3 )
              WRITE ( lp, '( ''  '' )', ADVANCE = 'no' )
              WRITE ( lp, mfrmt, ADVANCE = 'no' ) row( j )
              IF ( PRESENT( val ) ) THEN
                WRITE ( lp, '( '' ( '', ES12.4,'' )'' )', ADVANCE = 'no' )     &
                  val( j )
              END IF
            END DO
            WRITE ( lp, '(  )' )
          END DO
        END IF
        END SUBROUTINE print_matrix_int_precision

!-*-*-*-*-  G A L A H A D - d i g i t _ f o r m a t   F U N C T I O N  -*-*-*-*-

        CHARACTER ( LEN = 5 ) FUNCTION digit_format( x )
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: x

        INTEGER ( KIND = ip_ ) :: ndigit

        ndigit = INT( LOG10( real( x ) ) ) + 1
        IF ( ndigit < 10 ) THEN
          WRITE ( digit_format, '( ''( i'', i1,'' )'' )' ) ndigit
        ELSE
          WRITE ( digit_format, '( ''( i'', i2,'' )'' )' ) ndigit
        END IF
        RETURN

        END FUNCTION digit_format

!-*-*-  G A L A H A D - clean_cscl_oop_ptr32   S U B R O U T I N E  -*-*-

        SUBROUTINE clean_cscl_oop_ptr32_precision( matrix_type, m, n, ptr_in,  &
                                                   row_in, ptr_out, row_out,   &
                                                   flag, val_in, val_out,      &
                                                   lmap, map, lp, noor, ndup )

!  clean CSC matrix out of place. Lower entries only for symmetric,
!  skew-symmetric and Hermitian matrices to standard format

!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column pointers on input

        INTEGER ( KIND = i4_ ), DIMENSION ( * ), INTENT ( IN ) :: ptr_in

!  row indices on input. These may be unordered within each column and may 
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row_in

!  col ptr output

        INTEGER ( KIND = i4_ ), DIMENSION ( * ), INTENT ( OUT ) :: ptr_out
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ),                  &
                                             INTENT ( OUT ) :: row_out

!  row indices out. Duplicates and out-of-range entries are dealt with and
!  the entries within each column are ordered by increasing row index

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  values input

        REAL ( KIND = rp_ ), OPTIONAL, DIMENSION ( * ), INTENT ( IN ) :: val_in

!  values on output

        REAL ( KIND = rp_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: val_out
        INTEGER ( KIND = i4_ ), OPTIONAL, INTENT ( OUT ) :: lmap

!  map(1:size(val_out)) gives src: map(i) = j means val_out(i)=val_in(j).
!  map(size(val_out)+1:) gives pairs: map(i:i+1)=(j,k) means
!  val_out(j) = val_out(j)+val_in(k)

        INTEGER ( KIND = i4_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: map

!  unit for printing output if wanted

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( IN ) :: lp

!  number of out-of-range entries

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: noor

!  number of duplicates summed

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: ndup

!  Local variables

        CHARACTER ( 50 ) :: context !  Procedure name (used when printing)

!  output unit ( set to - 1 if nout not present )

        INTEGER ( KIND = ip_ ) :: nout

        context = 'clean_cscl_oop'

        nout = - 1
        IF ( PRESENT( lp ) ) nout = lp

!  note: have to change this test for complex code

        IF ( matrix_type < 0 .OR. matrix_type == 5 .OR. matrix_type>6 ) THEN
          flag = ERROR_MATRIX_TYPE
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        CALL clean_cscl_oop_main_ptr32( context, 1_ip_, matrix_type, m, n,     &
                                        ptr_in, row_in, ptr_out, row_out,      &
                                        flag, val_in, val_out, lmap, map,      &
                                        lp, noor, ndup )
        RETURN

        END SUBROUTINE clean_cscl_oop_ptr32_precision

!-*-*-  G A L A H A D - clean_cscl_oop_ptr64   S U B R O U T I N E  -*-*-

        SUBROUTINE clean_cscl_oop_ptr64_precision( matrix_type, m, n, ptr_in,  &
                                                   row_in, ptr_out, row_out,   &
                                                   flag, val_in, val_out,      &
                                                   lmap, map, lp, noor, ndup )

!  clean CSC matrix out of place. Lower entries only for symmetric,
!  skew-symmetric and Hermitian matrices to standard format

!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column pointers on input

        INTEGER ( KIND = i8_ ), DIMENSION ( * ), INTENT ( IN ) :: ptr_in

!  row indices on input. These may be unordered within each column and may
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row_in

!  col ptr output

        INTEGER ( KIND = i8_ ), DIMENSION ( * ), INTENT ( OUT ) :: ptr_out

!  row indices out. Duplicates and out-of-range entries are dealt with and
!  the entries within each column are ordered by increasing row index.

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ),                  &
                                             INTENT ( OUT ) :: row_out
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  values input

        REAL ( KIND = rp_ ), OPTIONAL, DIMENSION ( * ), INTENT ( IN ) :: val_in

!  values on output

        REAL ( KIND = rp_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: val_out
        INTEGER ( KIND = i8_ ), OPTIONAL, INTENT ( OUT ) :: lmap

!  map(1:size(val_out)) gives src: map(i) = j means val_out(i)=val_in(j).
!  map(size(val_out)+1:) gives pairs: map(i:i+1) = (j,k) means
!  val_out(j) = val_out(j) + val_in(k)

        INTEGER ( KIND = i8_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: map

!  unit for printing output if wanted

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( IN ) :: lp

!  number of out-of-range entries

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: noor

!  number of duplicates summed

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: ndup

!  Local variables

        CHARACTER ( 50 ) :: context !  Procedure name (used when printing)

!  output unit ( set to - 1 if nout not present )

        INTEGER ( KIND = ip_ ) :: nout

        context = 'clean_cscl_oop'

        nout = - 1
        IF ( PRESENT( lp ) ) nout = lp

!  note: have to change this test for complex code

        IF ( matrix_type < 0 .OR. matrix_type == 5 .OR. matrix_type > 6 ) THEN
          flag = ERROR_MATRIX_TYPE
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        CALL clean_cscl_oop_main( context, 1_ip_, matrix_type, m, n, ptr_in,   &
                                  row_in, ptr_out, row_out, flag, val_in,      &
                                  val_out, lmap, map, lp, noor, ndup )
        RETURN

        END SUBROUTINE clean_cscl_oop_ptr64_precision

!-*-*-  G A L A H A D - clean_cscl_oop_main_ptr32   S U B R O U T I N E  -*-*-

        SUBROUTINE clean_cscl_oop_main_ptr32( context, multiplier,             &
                                              matrix_type, m, n, ptr_in,       &
                                              row_in, ptr_out,                 &
                                              row_out, flag, val_in, val_out,  &
                                              lmap, map, lp, noor, ndup )

!  converts CSC (with lower entries only for symmetric, skew-symmetric and
!  Hermitian matrices) to standard format. Also used for symmetric, 
!  skew-symmetric and Hermitian matrices in upper CSR format

!  Procedure name ( used when printing ).

        CHARACTER ( 50 ), INTENT ( IN ) :: context

!  - 1 or 1, differs for csc/csr

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: multiplier

!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column pointers on input

        INTEGER ( KIND = i4_ ), DIMENSION ( * ), INTENT ( IN ) :: ptr_in

!  row indices on input. These may be unordered within each column and may
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row_in

!  col ptr output

        INTEGER ( KIND = i4_ ), DIMENSION ( * ), INTENT ( OUT ) :: ptr_out

!  row indices out. Duplicates and out-of-range entries are dealt with and
!  the entries within each column are ordered by increasing row index

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ),                  &
                                             INTENT ( OUT ) :: row_out

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  values input

        REAL ( KIND = rp_ ), OPTIONAL, DIMENSION ( * ), INTENT ( IN ) :: val_in

!  values on output

        REAL ( KIND = rp_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: val_out
        INTEGER ( KIND = i4_ ), OPTIONAL, INTENT ( OUT ) :: lmap

!  map(1:size(val_out)) gives src: map(i) = j means val_out(i)=val_in(j).
!  map(size(val_out)+1:) gives pairs: map(i:i+1) = (j,k) means
!  val_out(j) = val_out(j) + val_in(k)

        INTEGER ( KIND = i4_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: map

!  unit for printing output if wanted

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( IN ) :: lp

!  number of out-of-range entries

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: noor

!  number of duplicates summed

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: ndup


!  local variables

        INTEGER ( KIND = ip_ ) :: col !  current column
        INTEGER ( KIND = ip_ ) :: i, idiag, idup, ioor, j, k

!  output unit (set to - 1 if nout not present)

        INTEGER ( KIND = ip_ ) :: nout
        INTEGER ( KIND = ip_ ) :: st !  stat parameter
        INTEGER ( KIND = ip_ ) :: minidx

        TYPE ( dup_list ), POINTER :: dup
        TYPE ( dup_list ), POINTER :: duphead

!  check that restrictions are adhered to

        NULLIFY ( dup, duphead )
        flag = SUCCESS
        nout = - 1
        IF ( PRESENT( lp ) ) nout = lp

        IF ( n < 0 .OR. m < 0 ) THEN
          flag = ERROR_N_OOR
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( ptr_in( 1 ) < 1 ) THEN
          flag = ERROR_PTR_1
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( val_in ) .NEQV. PRESENT( val_out ) ) THEN
          flag = ERROR_VAL_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( map ) .NEQV. PRESENT( lmap ) ) THEN
          flag = ERROR_LMAP_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

!  ensure output arrays are not allocated

        DEALLOCATE ( row_out, STAT = st )
        IF ( PRESENT( val_out ) ) DEALLOCATE ( val_out, STAT = st )
        IF ( PRESENT( map ) ) DEALLOCATE ( map, STAT = st )

        idup = 0
        ioor = 0
        idiag = 0

        ALLOCATE ( row_out( ptr_in( n + 1 ) - 1 ), STAT = st )
        IF ( st /= 0 ) GO TO 100

!  allocate map for worst case where all bar one are duplicates

        IF ( PRESENT( map ) ) THEN
          ALLOCATE ( map( 2 * ptr_in( n + 1 ) - 2 ), STAT = st )
          k = 1 !  insert location
          DO col = 1, n
            ptr_out( col ) = INT( k, KIND = i4_ )
            IF ( ptr_in( col + 1 ) < ptr_in( col ) ) THEN
              flag = ERROR_PTR_MONO
              CALL print_matrix_flag( context, nout, flag )
              CALL cleanup_dup( duphead )
              RETURN
            END IF
            minidx = 1
            IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF )           &
              minidx = col
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW )                &
              minidx = col + 1

!  loop over column, copy across while dropping any out of range entries

            DO i = ptr_in( col ), ptr_in( col + 1 ) - 1
              j = row_in( i )
              IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                ioor = ioor + 1
                CYCLE
              END IF
              row_out( k ) = row_in( i )
              map( k ) = INT( multiplier * i, KIND = i4_ )
              k = k + 1
            END DO

!  sort entries into order

            i = k - ptr_out( col )
            IF ( i == 0 .AND. ptr_in( col + 1 ) - ptr_in( col ) /= 0 ) THEN
              flag = ERROR_ALL_OOR
              CALL print_matrix_flag( context, nout, flag )
              CALL cleanup_dup( duphead )
              RETURN
            END IF
            IF ( i /= 0 ) THEN
              CALL sort32( row_out( ptr_out( col ) : k - 1 ), i,               &
                           map = map( ptr_out( col ):k - 1 ) )

!  loop over sorted list and drop duplicates

              i = k - 1 !  last entry in column
              k = ptr_out( col ) + 1 !  insert position

!  note: we are skipping the first entry as it cannot be a duplicate

              IF ( row_out( ptr_out( col ) ) == col ) THEN
                idiag = idiag + 1
              ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                CALL cleanup_dup( duphead )
                RETURN
              END IF
              DO i = ptr_out( col ) + 1, i
                IF ( row_out( i ) == row_out( i - 1 ) ) THEN !  duplicate, drop
                  idup = idup + 1
                  ALLOCATE ( dup, STAT = st )
                  IF ( st /= 0 ) GO TO 100
                  dup%next => duphead
                  duphead => dup
                  dup%src = map( i )
                  dup%dest = INT( k - 1, KIND = i4_ )
                  CYCLE
                END IF
                IF ( row_out( i ) == col ) idiag = idiag + 1
                row_out( k ) = row_out( i )
                map( k ) = map( i )
                k = k + 1
              END DO
            ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
              flag = ERROR_MISSING_DIAGONAL
              CALL print_matrix_flag( context, nout, flag )
              CALL cleanup_dup( duphead )
              RETURN
            END IF
          END DO
          ptr_out( n + 1 ) = INT( k, KIND = i4_ )
          lmap = INT( k - 1, KIND = i4_ )
        ELSE IF ( PRESENT( val_out ) ) THEN
          ALLOCATE ( val_out( ptr_in( n + 1 ) - 1 ), STAT = st )
          IF ( st /= 0 ) GO TO 100
          k = 1 !  insert location
          DO col = 1, n
            ptr_out( col ) = INT( k, KIND = i4_ )
            IF ( ptr_in( col + 1 ) < ptr_in( col ) ) THEN
              flag = ERROR_PTR_MONO
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            minidx = 1
            IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF )           &
              minidx = col
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW )                &
              minidx = col + 1

!  Loop over column, copy across while dropping any out of range entries

            SELECT CASE ( matrix_type )
            CASE ( SSIDS_MATRIX_REAL_SKEW )
              DO i = ptr_in( col ), ptr_in( col + 1 ) - 1
                j = row_in( i )
                IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                  ioor = ioor + 1
                  CYCLE
                END IF
                row_out( k ) = row_in( i )
                val_out( k ) = REAL( multiplier, KIND = rp_ ) * val_in( i )
                k = k + 1
              END DO
            CASE DEFAULT
              DO i = ptr_in( col ), ptr_in( col + 1 ) - 1
                j = row_in( i )
                IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                  ioor = ioor + 1
                  CYCLE
                END IF
                row_out( k ) = row_in( i )
                val_out( k ) = val_in( i )
                k = k + 1
              END DO
            END SELECT

!  sort entries into order

            i = k - ptr_out( col )
            IF ( i == 0 .AND. ptr_in( col + 1 ) - ptr_in( col ) /= 0 ) THEN
              flag = ERROR_ALL_OOR
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            IF ( i /= 0 ) THEN
              CALL sort32( row_out( ptr_out( col ) : k - 1 ), i,               &
                           val = val_out( ptr_out( col ) : k - 1 ) )

!  loop over sorted list and drop duplicates

              i = k - 1 !  last entry in column
              k = ptr_out( col ) + 1 !  insert position

!  note: we are skipping the first entry as it cannot be a duplicate

              IF ( row_out( ptr_out( col ) ) == col ) THEN
                idiag = idiag + 1
              ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
              DO i = ptr_out( col ) + 1, i
                IF ( row_out( i ) == row_out( i- 1 ) ) THEN
!  duplicate, sum then drop from pattern
                  idup = idup + 1
                  val_out( i- 1 ) = val_out( i- 1 ) + val_out( i )
                  CYCLE
                END IF
                IF ( row_out( i ) == col ) idiag = idiag + 1
                row_out( k ) = row_out( i )
                val_out( k ) = val_out( i )
                k = k + 1
              END DO
            ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
              flag = ERROR_MISSING_DIAGONAL
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
          END DO
          ptr_out( n + 1 ) = INT( k, KIND = i4_ )
        ELSE !  pattern only
          k = 1 !  insert location
          DO col = 1, n
            ptr_out( col ) = INT( k, KIND = i4_ )
            IF ( ptr_in( col + 1 ) < ptr_in( col ) ) THEN
              flag = ERROR_PTR_MONO
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            minidx = 1
            IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF )           &
              minidx = col
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW )                &
              minidx = col + 1

!  loop over column, copy across while dropping any out of range entries

            DO i = ptr_in( col ), ptr_in( col + 1 ) - 1
              j = row_in( i )
              IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                ioor = ioor + 1
                CYCLE
              END IF
              row_out( k ) = row_in( i )
              k = k + 1
            END DO

!  sort entries into order

            i = k - ptr_out( col )
            IF ( i == 0 .AND. ptr_in( col + 1 ) - ptr_in( col ) /= 0 ) THEN
              flag = ERROR_ALL_OOR
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            IF ( i /= 0 ) THEN
              CALL sort32( row_out( ptr_out( col ) : k - 1 ), i )

!  loop over sorted list and drop duplicates

              i = k - 1 !  last entry in column
              k = ptr_out( col ) + 1 !  insert position

!  note: we are skipping the first entry as it cannot be a duplicate

              IF ( row_out( ptr_out( col ) ) == col ) THEN
                idiag = idiag + 1
              ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
              DO i = ptr_out( col ) + 1, i
                IF ( row_out( i ) == row_out( i - 1 ) ) THEN !  duplicate, drop
                  idup = idup + 1
                  CYCLE
                END IF
                IF ( row_out( i ) == col ) idiag = idiag + 1
                row_out( k ) = row_out( i )
                k = k + 1
              END DO
            ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
              flag = ERROR_MISSING_DIAGONAL
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
          END DO
          ptr_out( n + 1 ) = INT( k, KIND = i4_ )
        END IF

!  append duplicates to map

        IF ( PRESENT( map ) ) THEN
          DO WHILE ( associated( duphead ) )
            idup = idup + 1
            map( lmap + 1 ) = duphead%dest
            map( lmap + 2 ) = duphead%src
            lmap = lmap + 2
            dup => duphead%next
            DEALLOCATE( duphead )
            duphead => dup
          END DO
          IF ( PRESENT( val_out ) ) THEN
            ALLOCATE ( val_out( ptr_out( n + 1 ) - 1 ), STAT = st )
            IF ( st /= 0 ) GO TO 100
            CALL MU_apply_conversion_map( matrix_type, lmap, map, val_in,      &
                                          ptr_out( n + 1 ) - 1, val_out )
          END IF
        END IF

        IF ( PRESENT( val_out ) ) THEN
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF )

!  check for positive diagonal entries

            DO j = 1, n
              k = ptr_out( j )

!  positive definite case - can't reach here unless all entries have diagonal

              IF ( REAL( val_out( k ) ) <= zero ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
            END DO
          END SELECT
        END IF

!  check whether a warning needs to be raised

        IF ( ioor > 0 .OR. idup > 0 .OR. idiag < n ) THEN
          IF ( ioor > 0 ) flag = WARNING_IDX_OOR
          IF ( idup > 0 ) flag = WARNING_DUP_IDX
          IF ( idup > 0 .AND. ioor > 0 ) flag = WARNING_DUP_AND_OOR
          IF ( ABS( matrix_type ) /= SSIDS_MATRIX_REAL_SKEW ) THEN
            IF ( idiag < n .AND. ioor > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n .AND. idup > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n ) THEN
              flag = WARNING_MISSING_DIAGONAL
            END IF
          END IF
          CALL print_matrix_flag( context, nout, flag )
        END IF

        IF ( PRESENT( noor ) ) noor = ioor
        IF ( PRESENT( ndup ) ) ndup = idup
        RETURN

 100    IF ( st /= 0 ) THEN
          flag = ERROR_ALLOCATION
          CALL print_matrix_flag( context, nout, flag )
        END IF
        RETURN

        END SUBROUTINE clean_cscl_oop_main_ptr32

!-*-*-  G A L A H A D - clean_cscl_oop_main   S U B R O U T I N E  -*-*-

        SUBROUTINE clean_cscl_oop_main( context, multiplier, matrix_type,      &
                                        m, n, ptr_in, row_in, ptr_out,         &
                                        row_out, flag, val_in, val_out,        &
                                        lmap, map, lp, noor, ndup )

!  converts CSC ( with lower entries only for symmetric, skew-symmetric and
!  Hermitian matrices ) to standard format. Also used for symmetric, 
!  skew-symmetric and Hermitian matrices in upper CSR format


!  procedure name (used when printing).

        CHARACTER ( 50 ), INTENT ( IN ) :: context

!  - 1 or 1, differs for csc/csr

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: multiplier

!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m !  number of rows
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  number of columns

!  column pointers on input

        INTEGER ( KIND = i8_ ), DIMENSION ( * ), INTENT ( IN ) :: ptr_in

!  row indices on input. !  These may be unordered within each column and may
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( * ), INTENT ( IN ) :: row_in

!  col ptr output

        INTEGER ( KIND = i8_ ), DIMENSION ( * ), INTENT ( OUT ) :: ptr_out

!  row indices out. Duplicates and out-of-range entries are dealt with and
!  the entries within each column are ordered by increasing row index.

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ),                  &
                                             INTENT ( OUT ) :: row_out
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  values input

        REAL ( KIND = rp_ ), OPTIONAL, DIMENSION ( * ), INTENT ( IN ) :: val_in

!  values on output

        REAL ( KIND = rp_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: val_out
        INTEGER ( KIND = i8_ ), OPTIONAL, INTENT ( OUT ) :: lmap

!  map(1:size(val_out)) gives src: map(i) = j means val_out(i)=val_in(j).
!  map(size(val_out)+1:) gives pairs: map(i:i+1) = (j,k) means
!  val_out(j) = val_out(j) + val_in(k)

        INTEGER ( KIND = i8_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: map

!  unit for printing output if wanted

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( IN ) :: lp

!  number of out-of-range entries

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: noor

!  number of duplicates summed

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: ndup

!  local variables

        INTEGER ( KIND = ip_ ) :: col !  current column
        INTEGER ( KIND = ip_ ) :: i, idiag, idup, ioor, j
        INTEGER ( KIND = i8_ ) :: ii, kk

!  output unit (set to - 1 if nout not present)

        INTEGER ( KIND = ip_ ) :: nout
        INTEGER ( KIND = ip_ ) :: st !  stat parameter
        INTEGER ( KIND = ip_ ) :: minidx

        TYPE ( dup_list64 ), POINTER :: dup
        TYPE ( dup_list64 ), POINTER :: duphead

!  check that restrictions are adhered to

        NULLIFY ( dup, duphead )
        flag = SUCCESS
        nout = - 1
        IF ( PRESENT( lp ) ) nout = lp

        IF ( n < 0 .OR. m < 0 ) THEN
          flag = ERROR_N_OOR
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( ptr_in( 1 ) < 1 ) THEN
          flag = ERROR_PTR_1
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( val_in ) .NEQV. PRESENT( val_out ) ) THEN
          flag = ERROR_VAL_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( map ) .NEQV. PRESENT( lmap ) ) THEN
          flag = ERROR_LMAP_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

!  ensure output arrays are not allocated

        DEALLOCATE ( row_out, STAT = st )
        IF ( PRESENT( val_out ) ) DEALLOCATE ( val_out, STAT = st )
        IF ( PRESENT( map ) ) DEALLOCATE ( map, STAT = st )

        idup = 0
        ioor = 0
        idiag = 0

        ALLOCATE ( row_out( ptr_in( n + 1 ) - 1 ), STAT = st )
        IF ( st /= 0 ) GO TO 100

!  allocate map for worst case where all bar one are duplicates

        IF ( PRESENT( map ) ) THEN
          ALLOCATE ( map( 2 * ptr_in( n + 1 ) - 2 ), STAT = st )
          kk = 1 !  insert location
          DO col = 1, n
            ptr_out( col ) = kk
            IF ( ptr_in( col + 1 ) < ptr_in( col ) ) THEN
              flag = ERROR_PTR_MONO
              CALL print_matrix_flag( context, nout, flag )
              CALL cleanup_dup( duphead )
              RETURN
            END IF
            minidx = 1
            IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF )           &
              minidx = col
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW )                &
              minidx = col + 1

!  loop over column, copy across while dropping any out of range entries

            DO ii = ptr_in( col ), ptr_in( col + 1 ) - 1
              j = row_in( ii )
              IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                ioor = ioor + 1
                CYCLE
              END IF
              row_out( kk ) = row_in( ii )
              map( kk ) = multiplier * ii
              kk = kk + 1
            END DO

!  sort entries into order

            i = INT( kk -ptr_out( col ) )
            IF ( i == 0 .AND. ptr_in( col + 1 ) -ptr_in( col ) /= 0 ) THEN
              flag = ERROR_ALL_OOR
              CALL print_matrix_flag( context, nout, flag )
              CALL cleanup_dup( duphead )
              RETURN
            END IF
            IF ( i /= 0 ) THEN
              CALL sort64( row_out( ptr_out( col ) : kk - 1 ), i,              &
                           map = map( ptr_out( col ) : kk - 1 ) )

!  loop over sorted list and drop duplicates

              ii = kk - 1 !  last entry in column
              kk = ptr_out( col ) + 1 !  insert position

!  note: we are skipping the first entry as it cannot be a duplicate

              IF ( row_out( ptr_out( col ) ) == col ) THEN
                idiag = idiag + 1
              ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                CALL cleanup_dup( duphead )
                RETURN
              END IF
              DO ii = ptr_out( col ) + 1, ii
                IF ( row_out( ii ) == row_out( ii- 1 ) ) THEN !  duplicate, drop
                  idup = idup + 1
                  ALLOCATE ( dup, STAT = st )
                  IF ( st /= 0 ) GO TO 100
                  dup%next => duphead
                  duphead => dup
                  dup%src = map( ii )
                  dup%dest = kk - 1
                  CYCLE
                END IF
                IF ( row_out( ii ) == col ) idiag = idiag + 1
                row_out( kk ) = row_out( ii )
                map( kk ) = map( ii )
                kk = kk + 1
              END DO
            ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
              flag = ERROR_MISSING_DIAGONAL
              CALL print_matrix_flag( context, nout, flag )
              CALL cleanup_dup( duphead )
              RETURN
            END IF
          END DO
          ptr_out( n + 1 ) = kk
          lmap = kk - 1
        ELSE IF ( PRESENT( val_out ) ) THEN
          ALLOCATE ( val_out( ptr_in( n + 1 ) - 1 ), STAT = st )
          IF ( st /= 0 ) GO TO 100
          kk = 1 !  insert location
          DO col = 1, n
            ptr_out( col ) = kk
            IF ( ptr_in( col + 1 ) < ptr_in( col ) ) THEN
              flag = ERROR_PTR_MONO
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            minidx = 1
            IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF )           &
              minidx = col
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW )                &
              minidx = col + 1

!  loop over column, copy across while dropping any out of range entries

            SELECT CASE ( matrix_type )
            CASE ( SSIDS_MATRIX_REAL_SKEW )
              DO ii = ptr_in( col ), ptr_in( col + 1 ) - 1
                j = row_in( ii )
                IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                  ioor = ioor + 1
                  CYCLE
                END IF
                row_out( kk ) = row_in( ii )
                val_out( kk ) = real( multiplier, KIND = rp_ )*val_in( ii )
                kk = kk + 1
              END DO
            CASE DEFAULT
              DO ii = ptr_in( col ), ptr_in( col + 1 ) - 1
                j = row_in( ii )
                IF ( j < minidx .OR. j > m ) THEN
!  out of range, ignore
                  ioor = ioor + 1
                  CYCLE
                END IF
                row_out( kk ) = row_in( ii )
                val_out( kk ) = val_in( ii )
                kk = kk + 1
              END DO
            END SELECT
!  Sort entries into order
            i = INT( kk -ptr_out( col ) )
            IF ( i == 0 .AND. ptr_in( col + 1 ) - ptr_in( col ) /= 0 ) THEN
              flag = ERROR_ALL_OOR
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            IF ( i /= 0 ) THEN
              CALL sort32( row_out( ptr_out( col ) : kk - 1 ), i,              &
                           val = val_out( ptr_out( col ) : kk - 1 ) )

!  loop over sorted list and drop duplicates

              ii = kk - 1 !  last entry in column
              kk = ptr_out( col ) + 1 !  insert position

!  note: we are skipping the first entry as it cannot be a duplicate

              IF ( row_out( ptr_out( col ) ) == col ) THEN
                idiag = idiag + 1
              ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
              DO ii = ptr_out( col ) + 1, ii
                IF ( row_out( ii ) == row_out( ii- 1 ) ) THEN

!  duplicate, sum then drop from pattern

                  idup = idup + 1
                  val_out( ii- 1 ) = val_out( ii- 1 ) + val_out( ii )
                  CYCLE
                END IF
                IF ( row_out( ii ) == col ) idiag = idiag + 1
                row_out( kk ) = row_out( ii )
                val_out( kk ) = val_out( ii )
                kk = kk + 1
              END DO
            ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
              flag = ERROR_MISSING_DIAGONAL
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
          END DO
          ptr_out( n + 1 ) = kk
        ELSE !  pattern only
          kk = 1 !  insert location
          DO col = 1, n
            ptr_out( col ) = kk
            IF ( ptr_in( col + 1 ) < ptr_in( col ) ) THEN
              flag = ERROR_PTR_MONO
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            minidx = 1
            IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_SYM_PSDEF )           &
              minidx = col
            IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW )                &
              minidx = col + 1

!  loop over column, copy across while dropping any out of range entries

            DO ii = ptr_in( col ), ptr_in( col + 1 ) - 1
              j = row_in( ii )
              IF ( j < minidx .OR. j > m ) THEN !  out of range, ignore
                ioor = ioor + 1
                CYCLE
              END IF
              row_out( kk ) = row_in( ii )
              kk = kk + 1
            END DO

!  sort entries into order

            i = INT( kk -ptr_out( col ) )
            IF ( i == 0 .AND. ptr_in( col + 1 ) -ptr_in( col ) /= 0 ) THEN
              flag = ERROR_ALL_OOR
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
            IF ( i /= 0 ) THEN
              CALL sort32( row_out( ptr_out( col ) : kk - 1 ), i )

!  loop over sorted list and drop duplicates

              ii = kk - 1 !  last entry in column
              kk = ptr_out( col ) + 1 !  insert position

!  note: we are skipping the first entry as it cannot be a duplicate

              IF ( row_out( ptr_out( col ) ) == col ) THEN
                idiag = idiag + 1
              ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
              DO ii = ptr_out( col ) + 1, ii
                IF ( row_out( ii ) == row_out( ii- 1 ) ) THEN !  duplicate, drop
                  idup = idup + 1
                  CYCLE
                END IF
                IF ( row_out( ii ) == col ) idiag = idiag + 1
                row_out( kk ) = row_out( ii )
                kk = kk + 1
              END DO
            ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
              flag = ERROR_MISSING_DIAGONAL
              CALL print_matrix_flag( context, nout, flag )
              RETURN
            END IF
          END DO
          ptr_out( n + 1 ) = kk
        END IF

!  append duplicates to map

        IF ( PRESENT( map ) ) THEN
          DO WHILE ( associated( duphead ) )
            idup = idup + 1
            map( lmap + 1 ) = duphead%dest
            map( lmap + 2 ) = duphead%src
            lmap = lmap + 2
            dup => duphead%next
            DEALLOCATE ( duphead )
            duphead => dup
          END DO
          IF ( PRESENT( val_out ) ) THEN
            ALLOCATE ( val_out( ptr_out( n + 1 ) - 1 ), STAT = st )
            IF ( st /= 0 ) GO TO 100
            CALL MU_apply_conversion_map( matrix_type, lmap, map, val_in,      &
                                          ptr_out( n + 1 ) - 1, val_out )
          END IF
        END IF

        IF ( PRESENT( val_out ) ) THEN
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF )

!  check for positive diagonal entries

            DO j = 1, n
              kk = ptr_out( j )

!  positive definite case - can't reach here unless all entries have diagonal

              IF ( REAL( val_out( kk ) ) <= zero ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
            END DO
          END SELECT
        END IF

!  check whether a warning needs to be raised

        IF ( ioor > 0 .OR. idup > 0 .OR. idiag < n ) THEN
          IF ( ioor > 0 ) flag = WARNING_IDX_OOR
          IF ( idup > 0 ) flag = WARNING_DUP_IDX
          IF ( idup > 0 .AND. ioor > 0 ) flag = WARNING_DUP_AND_OOR
          IF ( ABS( matrix_type ) /= SSIDS_MATRIX_REAL_SKEW ) THEN
            IF ( idiag < n .AND. ioor > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n .AND. idup > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n ) THEN
              flag = WARNING_MISSING_DIAGONAL
            END IF
          END IF
          CALL print_matrix_flag( context, nout, flag )
        END IF

        IF ( PRESENT( noor ) ) noor = ioor
        IF ( PRESENT( ndup ) ) ndup = idup
        RETURN

100     IF ( st /= 0 ) THEN
          flag = ERROR_ALLOCATION
          CALL print_matrix_flag( context, nout, flag )
        END IF
        RETURN

        END SUBROUTINE clean_cscl_oop_main

!-*-*-  G A L A H A D - convert_coord_to_cscl_ptr32   S U B R O U T I N E  -*-*-

        SUBROUTINE convert_coord_to_cscl_ptr32_precision( matrix_type, m, n,   &
           ne, row, col, ptr_out, row_out, flag, val_in, val_out, lmap, map,   &
           lp, noor, ndup )

!  converts COOR format to CSC with only lower entries present for
!  (skew-)symmetric problems. Entries within each column ordered by increasing
!  row index (standard format)

!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m ! number of rows in matrix
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n ! number of columns in matrix

!  number of input nonzero entries

        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ne

!  row indices on input. These may be unordered within each column and may 
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( ne ), INTENT ( IN ) :: row

!  column indices on input.  These may be unordered within each column and may
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( ne ), INTENT ( IN ) :: col

!  col ptr output

        INTEGER ( KIND = i4_ ), DIMENSION ( n + 1 ), INTENT ( OUT ) :: ptr_out

!  row indices out. Duplicates and out-of-range entries are dealt with and
!  the entries within each column are ordered by increasing row index.

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ),                  &
                                             INTENT ( OUT ) :: row_out
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  values input

        REAL ( KIND = rp_ ), OPTIONAL, DIMENSION ( * ), INTENT ( IN ) :: val_in

!  values on output

        REAL ( KIND = rp_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: val_out
        INTEGER ( KIND = i4_ ), OPTIONAL, INTENT ( OUT ) :: lmap

!  map(1:size(val_out)) gives src: map(i) = j means val_out(i)=val_in(j).
!  map(size(val_out)+1:) gives pairs: map(i:i+1) = (j,k) means
!  val_out(j) = val_out(j) + val_in(k)

        INTEGER ( KIND = i4_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: map

!  unit for printing output if wanted

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( IN ) :: lp

!  number of out-of-range entries

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: noor

!  number of duplicates summed

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: ndup

!  local variables

        CHARACTER ( 50 ) :: context !  Procedure name (used when printing)
        INTEGER ( KIND = ip_ ) :: i, idiag, idup, ioor, j, l
        INTEGER ( KIND = i4_ ) :: k, l1, l2
        INTEGER ( KIND = i4_ ) :: ne_new

!  output unit (set to - 1 if lp not present)

        INTEGER ( KIND = ip_ ) :: nout
        INTEGER ( KIND = ip_ ) :: st !  stat parameter

        TYPE ( dup_list ), POINTER :: dup
        TYPE ( dup_list ), POINTER :: duphead

        NULLIFY ( dup, duphead )

        context = 'convert_coord_to_cscl'

        flag = SUCCESS
        nout = - 1
        IF ( PRESENT( lp ) ) nout = lp

!  check that restrictions are adhered to

!  note: have to change this test for complex code

        IF ( matrix_type < 0 .OR. matrix_type == 5 .OR. matrix_type > 6 ) THEN
          flag = ERROR_MATRIX_TYPE
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( m < 0 .OR. n < 0 ) THEN
          flag = ERROR_N_OOR
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_UNSYM .AND. m /= n ) THEN
          flag = ERROR_M_NE_N
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( val_in ) .NEQV. PRESENT( val_out ) ) THEN
          flag = ERROR_VAL_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( map ) .NEQV. PRESENT( lmap ) ) THEN
          flag = ERROR_LMAP_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

!  allocate output and work arrays

        DEALLOCATE ( row_out, STAT = st )
        IF ( PRESENT( val_out ) ) DEALLOCATE ( val_out, STAT = st )
        IF ( PRESENT( map ) ) DEALLOCATE ( map, STAT = st )

!  check for duplicates and/or out-of-range. check diagonal present.
!  first do the case where values not present

        idup = 0
        ioor = 0
        idiag = 0

!  first pass, count number of entries in each col of the matrix
!  matrix. Count is at an offset of 1 to allow us to play tricks
!  (ie. ptr_out(i+1) set to hold number of entries in column i of expanded 
!  matrix). Excludes out of range entries. Includes duplicates
!
        ptr_out( 1:n + 1 ) = 0
        DO l = 1, ne
          i = row( l )
          j = col( l )
          IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) THEN
            ioor = ioor + 1
            CYCLE
          END IF

          IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW .AND. i == j ) THEN
            ioor = ioor + 1
            CYCLE
          END IF

          SELECT CASE ( ABS( matrix_type ) )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF: )
            IF ( i >= j ) THEN
              ptr_out( j + 1 ) = ptr_out( j + 1 ) + 1
            ELSE
              ptr_out( i + 1 ) = ptr_out( i + 1 ) + 1
            END IF
          CASE DEFAULT
            ptr_out( j + 1 ) = ptr_out( j + 1 ) + 1
          END SELECT
        END DO

!  determine column starts for transposed expanded matrix such
!  that column i starts at ptr_out( i )

        ne_new = 0
        ptr_out( 1 ) = 1
        DO i = 2, n + 1
          ne_new = ne_new + ptr_out( i )
          ptr_out( i ) = ptr_out( i ) + ptr_out( i- 1 )
        END DO

!  check whether all entries out of range

        IF ( ne > 0 .AND. ne_new == 0 ) THEN
          flag = ERROR_ALL_OOR
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

!  second pass, drop entries into place for conjugate of transposed 
!  expanded matrix

        ALLOCATE ( row_out( ne_new ), STAT = st )
        IF ( st /= 0 ) GO TO 100
        IF ( PRESENT( map ) ) THEN
          IF ( allocated( map ) ) DEALLOCATE ( map, STAT = st )
          ALLOCATE ( map( 2 * ne ), STAT = st )
          IF ( st /= 0 ) GO TO 100
          map( : ) = 0
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SKEW )
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m .OR. i == j ) CYCLE
              IF ( i >= j ) THEN
                k = ptr_out( j )
                ptr_out( j ) = k + 1
                row_out( k ) = i
                map( k ) = INT( l, KIND = i4_ )
              ELSE
                k = ptr_out( i )
                ptr_out( i ) = k + 1
                row_out( k ) = j
                map( k ) = INT( - l, KIND = i4_ )
              END IF
            END DO

          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF )
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              IF ( i >= j ) THEN
                k = ptr_out( j )
                ptr_out( j ) = k + 1
                row_out( k ) = i
                map( k ) = INT( l, KIND = i4_ )
              ELSE
                k = ptr_out( i )
                ptr_out( i ) = k + 1
                row_out( k ) = j
                map( k ) = INT( l, KIND = i4_ )
              END IF
            END DO
          CASE DEFAULT
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              k = ptr_out( j )
              ptr_out( j ) = k + 1
              row_out( k ) = i
              map( k ) = INT( l, KIND = i4_ )
            END DO
          END SELECT
        ELSE IF ( PRESENT( val_out ) ) THEN
          ALLOCATE ( val_out( ne_new ), STAT = st )
          IF ( st /= 0 ) GO TO 100
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SKEW )
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m .OR. i == j ) CYCLE
              IF ( i >= j ) THEN
                k = ptr_out( j )
                ptr_out( j ) = k + 1
                row_out( k ) = i
                val_out( k ) = val_in( l )
              ELSE
                k = ptr_out( i )
                ptr_out( i ) = k + 1
                row_out( k ) = j
                val_out( k ) = -val_in( l )
              END IF
            END DO

          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF )
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              IF ( i >= j ) THEN
                k = ptr_out( j )
                ptr_out( j ) = k + 1
                row_out( k ) = i
                val_out( k ) = val_in( l )
              ELSE
                k = ptr_out( i )
                ptr_out( i ) = k + 1
                row_out( k ) = j
                val_out( k ) = val_in( l )
              END IF
            END DO
          CASE DEFAULT
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              k = ptr_out( j )
              ptr_out( j ) = k + 1
              row_out( k ) = i
              val_out( k ) = val_in( l )
            END DO
          END SELECT

!  neither val_out or map present

        ELSE
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SKEW )
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m .OR. i == j ) CYCLE
              IF ( i >= j ) THEN
                k = ptr_out( j )
                ptr_out( j ) = k + 1
                row_out( k ) = i
              ELSE
                k = ptr_out( i )
                ptr_out( i ) = k + 1
                row_out( k ) = j
              END IF
            END DO

          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF )
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              IF ( i >= j ) THEN
                k = ptr_out( j )
                ptr_out( j ) = k + 1
                row_out( k ) = i
              ELSE
                k = ptr_out( i )
                ptr_out( i ) = k + 1
                row_out( k ) = j
              END IF
            END DO
          CASE DEFAULT
            DO l = 1, ne
              i = row( l )
              j = col( l )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              k = ptr_out( j )
              ptr_out( j ) = k + 1
              row_out( k ) = i
            END DO
          END SELECT
        END IF

        DO j = n, 2, - 1
          ptr_out( j ) = ptr_out( j - 1 )
        END DO
        ptr_out( 1 ) = 1

!  third pass, in place sort and removal of duplicates
!  Also checks for diagonal entries in pos. def. case.
!  Uses a modified insertion sort for best speed on already ordered data

        idup = 0
        IF ( PRESENT( map ) ) THEN
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = l2 - l1 + 1
            IF ( l > 1 ) CALL sort32( row_out( l1 : l2 ), l,                   &
                                      map = map( l1 : l2 ) )
          END DO

!  work through removing duplicates

          k = 1 !  insert position
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            ptr_out( j ) = k

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = l2 - l1 + 1
            IF ( l == 0 ) CYCLE !  no entries

!  move first entry of column forward

            IF ( row_out( l1 ) == j ) idiag = idiag + 1
            row_out( k ) = row_out( l1 )
            map( k ) = map( l1 )
            k = k + 1

!  loop over remaining entries

            DO i = l1 + 1, l2
              IF ( row_out( i ) == row_out( k - 1 ) ) THEN !  Duplicate
                idup = idup + 1
                ALLOCATE ( dup, STAT = st )
                IF ( st /= 0 ) GO TO 100
                dup%next => duphead
                duphead => dup
                dup%src = map( i )
                dup%dest = k - 1
                CYCLE
              END IF

!  pull entry forwards

              IF ( row_out( i ) == j ) idiag = idiag + 1
              row_out( k ) = row_out( i )
              map( k ) = map( i )
              k = k + 1
            END DO
          END DO
          ptr_out( n + 1 ) = k
          lmap = k - 1
        ELSE IF ( PRESENT( val_out ) ) THEN
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            l = l2 - l1 + 1
            IF ( l > 1 ) CALL sort32( row_out( l1 : l2 ), l,                   &
                                      val = val_out( l1 : l2 ) )
          END DO

!  work through removing duplicates

          k = 1 !  insert position
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            ptr_out( j ) = k

!  sort( row_out( l1:l2 ) ) and permute map( l1:l2 ) accordingly

            l = l2 - l1 + 1
            IF ( l == 0 ) CYCLE !  no entries

!  move first entry of column forward

            IF ( row_out( l1 ) == j ) idiag = idiag + 1
            row_out( k ) = row_out( l1 )
            val_out( k ) = val_out( l1 )
            k = k + 1

!  loop over remaining entries

            DO i = l1 + 1, l2
              IF ( row_out( i ) == row_out( k - 1 ) ) THEN
                idup = idup + 1
                val_out( k - 1 ) = val_out( k - 1 ) + val_out( i )
                CYCLE
              END IF

!  pull entry forwards

              IF ( row_out( i ) == j ) idiag = idiag + 1
              row_out( k ) = row_out( i )
              val_out( k ) = val_out( i )
              k = k + 1
            END DO
          END DO
          ptr_out( n + 1 ) = k
        ELSE
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            l = l2 - l1 + 1
            IF ( l > 1 ) CALL sort32( row_out( l1 : l2 ), l )
          END DO

!  work through removing duplicates

          k = 1 !  insert position
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            ptr_out( j ) = k

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = l2 - l1 + 1
            IF ( l == 0 ) CYCLE !  no entries

!  move first entry of column forward

            IF ( row_out( l1 ) == j ) idiag = idiag + 1
            row_out( k ) = row_out( l1 )
            k = k + 1

!  loop over remaining entries

            DO i = l1 + 1, l2
              IF ( row_out( i ) == row_out( k - 1 ) ) THEN
                idup = idup + 1
                CYCLE
              END IF

!  pull entry forwards

              IF ( row_out( i ) == j ) idiag = idiag + 1
              row_out( k ) = row_out( i )
              k = k + 1
            END DO
          END DO
          ptr_out( n + 1 ) = k

        END IF

!  check for missing diagonals in pos def and indef cases
!  Note: change this test for complex case

        IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
          DO j = 1, n
            IF ( ptr_out( j ) < ptr_out( n + 1 ) ) THEN
              IF ( row_out( ptr_out( j ) ) /= j ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
            END IF
          END DO
        END IF

!  append duplicates to map

        IF ( PRESENT( map ) ) THEN
          DO WHILE ( associated( duphead ) )
            map( lmap + 1 ) = duphead%dest
            map( lmap + 2 ) = duphead%src
            lmap = lmap + 2
            dup => duphead%next
            DEALLOCATE ( duphead )
            duphead => dup
          END DO
          IF ( PRESENT( val_out ) ) THEN
            ALLOCATE ( val_out( ptr_out( n + 1 ) - 1 ), STAT = st )
            IF ( st /= 0 ) GO TO 100
            CALL MU_apply_conversion_map( matrix_type, lmap, map, val_in,      &
                                          ptr_out( n + 1 ) - 1, val_out )
          END IF
        END IF

        IF ( PRESENT( val_out ) ) THEN
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF )

!  check for positive diagonal entries

            DO j = 1, n
              k = ptr_out( j )

!  positive definite case - can't reach here unless all entries have diagonal

              IF ( real( val_out( k ) ) <= zero ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
            END DO
          END SELECT
        END IF

!  check whether a warning needs to be raised

        IF ( ioor > 0 .OR. idup > 0 .OR. idiag < n ) THEN
          IF ( ioor > 0 ) flag = WARNING_IDX_OOR
          IF ( idup > 0 ) flag = WARNING_DUP_IDX
          IF ( idup > 0 .AND. ioor > 0 ) flag = WARNING_DUP_AND_OOR
          IF ( ABS( matrix_type ) /= SSIDS_MATRIX_REAL_SKEW ) THEN
            IF ( idiag < n .AND. ioor > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n .AND. idup > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n ) THEN
              flag = WARNING_MISSING_DIAGONAL
            END IF
          END IF
          CALL print_matrix_flag( context, nout, flag )
        END IF

!  check whether a warning needs to be raised

        IF ( ioor > 0 .OR. idup > 0 .OR. idiag < n ) THEN
          IF ( ioor > 0 ) flag = WARNING_IDX_OOR
          IF ( idup > 0 ) flag = WARNING_DUP_IDX
          IF ( idup > 0 .AND. ioor > 0 ) flag = WARNING_DUP_AND_OOR
          IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_INDEF .AND.         &
               idiag < n .AND. ioor > 0 ) THEN
            flag = WARNING_MISS_DIAG_OORDUP
          ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_INDEF .AND.    &
                    idiag < n .AND. idup > 0 ) THEN
            flag = WARNING_MISS_DIAG_OORDUP
          ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_INDEF .AND.    &
                    idiag < n ) THEN
            flag = WARNING_MISSING_DIAGONAL
          END IF
          CALL print_matrix_flag( context, nout, flag )
        END IF

        IF ( PRESENT( noor ) ) noor = ioor
        IF ( PRESENT( ndup ) ) ndup = idup
        RETURN

 100    CONTINUE
        IF ( st /= 0 ) THEN
          flag = ERROR_ALLOCATION
          CALL print_matrix_flag( context, nout, flag )
        END IF
        RETURN

        END SUBROUTINE convert_coord_to_cscl_ptr32_precision

!-*-*-  G A L A H A D -  convert_coord_to_cscl_ptr64  S U B R O U T I N E  -*-*-

        SUBROUTINE convert_coord_to_cscl_ptr64_precision( matrix_type, m, n,   &
                     ne, row, col, ptr_out, row_out, flag, val_in, val_out,    &
                     lmap, map, lp, noor, ndup )

!  converts COOR format to CSC with only lower entries present for
!  (skew-)symmetric problems. Entries within each column ordered by increasing
!  row index (standard format)


!  what sort of symmetry is there?

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: m ! number of rows in matrix
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n ! number of columns in matrix

!  number of input nonzero entries

        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ne

!  row indices on input. These may be unordered within each column and may
!  contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( ne ), INTENT ( IN ) :: row 

!  column indices on input. These may be unordered within each column and 
!  may contain duplicates and/or out-of-range entries

        INTEGER ( KIND = ip_ ), DIMENSION ( ne ), INTENT ( IN ) :: col

!  col ptr output

        INTEGER ( KIND = i8_ ), DIMENSION ( n + 1 ), INTENT ( OUT ) :: ptr_out

!  row indices out. Duplicates and out-of-range entries are dealt with and
!  the entries within each column are ordered by increasing row index.

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ),                  &
                                             INTENT ( OUT ) :: row_out
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag !  return code

!  values input

        REAL ( KIND = rp_ ), OPTIONAL, DIMENSION ( * ), INTENT ( IN ) :: val_in

!  values on output

        REAL ( KIND = rp_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: val_out
        INTEGER ( KIND = i8_ ), OPTIONAL, INTENT ( OUT ) :: lmap

!  map(1:size(val_out)) gives src: map(i) = j means val_out(i)=val_in(j).
!  map(size(val_out)+1:) gives pairs: map(i:i+1) = (j,k) means
!  val_out(j) = val_out(j) + val_in(k)

        INTEGER ( KIND = i8_ ), OPTIONAL, ALLOCATABLE, DIMENSION ( : ) :: map

!  unit for printing output if wanted

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( IN ) :: lp

!  number of out-of-range entries

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: noor

!  number of duplicates summed

        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT ( OUT ) :: ndup

!  local variables

        CHARACTER ( 50 ) :: context !  procedure name (used when printing)
        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = ip_ ) :: idiag
        INTEGER ( KIND = ip_ ) :: idup
        INTEGER ( KIND = ip_ ) :: ioor
        INTEGER ( KIND = ip_ ) :: j
        INTEGER ( KIND = ip_ ) :: l
        INTEGER ( KIND = i8_ ) :: ii, ll, l1, l2, kk
        INTEGER ( KIND = i8_ ) :: ne_new

!  output unit (set to - 1 if lp not present)

        INTEGER ( KIND = ip_ ) :: nout
        INTEGER ( KIND = ip_ ) :: st !  stat parameter

        TYPE ( dup_list64 ), POINTER :: dup
        TYPE ( dup_list64 ), POINTER :: duphead

        NULLIFY ( dup, duphead )

        context = 'convert_coord_to_cscl'

        flag = SUCCESS

        nout = - 1
        IF ( PRESENT( lp ) ) nout = lp

!  check that restrictions are adhered to.
!  Note: have to change this test for complex code

        IF ( matrix_type < 0 .OR. matrix_type == 5 .OR. matrix_type>6 ) THEN
          flag = ERROR_MATRIX_TYPE
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( m < 0 .OR. n < 0 ) THEN
          flag = ERROR_N_OOR
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( ABS( matrix_type ) >= SSIDS_MATRIX_REAL_UNSYM .AND. m /= n ) THEN
          flag = ERROR_M_NE_N
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( val_in ) .NEQV. PRESENT( val_out ) ) THEN
          flag = ERROR_VAL_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

        IF ( PRESENT( map ) .NEQV. PRESENT( lmap ) ) THEN
          flag = ERROR_LMAP_MISS
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

!  allocate output and work arrays

        DEALLOCATE ( row_out, STAT = st )
        IF ( PRESENT( val_out ) ) DEALLOCATE ( val_out, STAT = st )
        IF ( PRESENT( map ) ) DEALLOCATE ( map, STAT = st )

!  check for duplicates and/or out-of-range. check diagonal present.
!  first do the case where values not present

        idup = 0
        ioor = 0
        idiag = 0

!  first pass, count number of entries in each col of the matrix
!  matrix. Count is at an offset of 1 to allow us to play tricks
!  (ie. ptr_out(i+1) set to hold number of entries in column i of expanded 
!  matrix).  Excludes out of range entries. Includes duplicates
!
        ptr_out( 1:n + 1 ) = 0
        DO ll = 1, ne
          i = row( ll )
          j = col( ll )
          IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) THEN
            ioor = ioor + 1
            CYCLE
          END IF

          IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SKEW .AND. i == j ) THEN
            ioor = ioor + 1
            CYCLE
          END IF

          SELECT CASE ( ABS( matrix_type ) )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF: )
            IF ( i >= j ) THEN
              ptr_out( j + 1 ) = ptr_out( j + 1 ) + 1
            ELSE
              ptr_out( i + 1 ) = ptr_out( i + 1 ) + 1
            END IF
          CASE DEFAULT
            ptr_out( j + 1 ) = ptr_out( j + 1 ) + 1
          END SELECT
        END DO

!  determine column starts for transposed expanded matrix such that column 
!  i starts at ptr_out(i)

        ne_new = 0
        ptr_out( 1 ) = 1
        DO i = 2, n + 1
          ne_new = ne_new + ptr_out( i )
          ptr_out( i ) = ptr_out( i ) + ptr_out( i- 1 )
        END DO

!  check whether all entries out of range

        IF ( ne > 0 .AND. ne_new == 0 ) THEN
          flag = ERROR_ALL_OOR
          CALL print_matrix_flag( context, nout, flag )
          RETURN
        END IF

!  second pass, drop entries into place for conjugate of transposed
!  expanded matrix

        ALLOCATE ( row_out( ne_new ), STAT = st )
        IF ( st /= 0 ) GO TO 100
        IF ( PRESENT( map ) ) THEN
          IF ( allocated( map ) ) DEALLOCATE ( map, STAT = st )
          ALLOCATE ( map( 2 * ne ), STAT = st )
          IF ( st /= 0 ) GO TO 100
          map( : ) = 0
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SKEW )
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m .OR. i == j ) CYCLE
              IF ( i >= j ) THEN
                kk = ptr_out( j )
                ptr_out( j ) = kk + 1
                row_out( kk ) = i
                map( kk ) = ll
              ELSE
                kk = ptr_out( i )
                ptr_out( i ) = kk + 1
                row_out( kk ) = j
                map( kk ) = -ll
              END IF
            END DO

          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF )
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              IF ( i >= j ) THEN
                kk = ptr_out( j )
                ptr_out( j ) = kk + 1
                row_out( kk ) = i
                map( kk ) = ll
              ELSE
                kk = ptr_out( i )
                ptr_out( i ) = kk + 1
                row_out( kk ) = j
                map( kk ) = ll
              END IF
            END DO
          CASE DEFAULT
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              kk = ptr_out( j )
              ptr_out( j ) = kk + 1
              row_out( kk ) = i
              map( kk ) = ll
            END DO
          END SELECT
        ELSE IF ( PRESENT( val_out ) ) THEN
          ALLOCATE ( val_out( ne_new ), STAT = st )
          IF ( st /= 0 ) GO TO 100
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SKEW )
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m .OR. i == j ) CYCLE
              IF ( i >= j ) THEN
                kk = ptr_out( j )
                ptr_out( j ) = kk + 1
                row_out( kk ) = i
                val_out( kk ) = val_in( ll )
              ELSE
                kk = ptr_out( i )
                ptr_out( i ) = kk + 1
                row_out( kk ) = j
                val_out( kk ) = -val_in( ll )
              END IF
            END DO

          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF )
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              IF ( i >= j ) THEN
                kk = ptr_out( j )
                ptr_out( j ) = kk + 1
                row_out( kk ) = i
                val_out( kk ) = val_in( ll )
              ELSE
                kk = ptr_out( i )
                ptr_out( i ) = kk + 1
                row_out( kk ) = j
                val_out( kk ) = val_in( ll )
              END IF
            END DO
          CASE DEFAULT
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              kk = ptr_out( j )
              ptr_out( j ) = kk + 1
              row_out( kk ) = i
              val_out( kk ) = val_in( ll )
            END DO
          END SELECT

!  neither val_out or map present

        ELSE
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SKEW )
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m .OR. i == j ) CYCLE
              IF ( i >= j ) THEN
                kk = ptr_out( j )
                ptr_out( j ) = kk + 1
                row_out( kk ) = i
              ELSE
                kk = ptr_out( i )
                ptr_out( i ) = kk + 1
                row_out( kk ) = j
              END IF
            END DO

          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF, SSIDS_MATRIX_REAL_SYM_INDEF )
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              IF ( i >= j ) THEN
                kk = ptr_out( j )
                ptr_out( j ) = kk + 1
                row_out( kk ) = i
              ELSE
                kk = ptr_out( i )
                ptr_out( i ) = kk + 1
                row_out( kk ) = j
              END IF
            END DO
          CASE DEFAULT
            DO ll = 1, ne
              i = row( ll )
              j = col( ll )
              IF ( j < 1 .OR. j > n .OR. i < 1 .OR. i > m ) CYCLE
              kk = ptr_out( j )
              ptr_out( j ) = kk + 1
              row_out( kk ) = i
            END DO
          END SELECT
        END IF

        DO j = n, 2, - 1
          ptr_out( j ) = ptr_out( j - 1 )
        END DO
        ptr_out( 1 ) = 1

!  third pass, in place sort and removal of duplicates
!  Also checks for diagonal entries in pos. def. case.
!  Uses a modified insertion sort for best speed on already ordered data

        idup = 0
        IF ( PRESENT( map ) ) THEN
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = INT( l2 - l1 + 1 )
            IF ( l > 1 ) CALL sort64( row_out( l1 : l2 ), l,                   &
                                      map = map( l1 : l2 ) )
          END DO

!  work through removing duplicates

          kk = 1 !  insert position
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            ptr_out( j ) = kk

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = INT( l2-l1 + 1 )
            IF ( l == 0 ) CYCLE !  no entries

!  move first entry of column forward

            IF ( row_out( l1 ) == j ) idiag = idiag + 1
            row_out( kk ) = row_out( l1 )
            map( kk ) = map( l1 )
            kk = kk + 1

!  loop over remaining entries

            DO ii = l1 + 1, l2
              IF ( row_out( ii ) == row_out( kk - 1 ) ) THEN !  Duplicate
                idup = idup + 1
                ALLOCATE ( dup, STAT = st )
                IF ( st /= 0 ) GO TO 100
                dup%next => duphead
                duphead => dup
                dup%src = map( ii )
                dup%dest = kk - 1
                CYCLE
              END IF

!  pull entry forwards

              IF ( row_out( ii ) == j ) idiag = idiag + 1
              row_out( kk ) = row_out( ii )
              map( kk ) = map( ii )
              kk = kk + 1
            END DO
          END DO
          ptr_out( n + 1 ) = kk
          lmap = kk - 1
        ELSE IF ( PRESENT( val_out ) ) THEN
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            l = INT( l2-l1 + 1 )
            IF ( l > 1 ) CALL sort32( row_out( l1 : l2 ), l,                   &
                                      val = val_out( l1 : l2 ) )
          END DO

!  work through removing duplicates

          kk = 1 !  insert position
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            ptr_out( j ) = kk

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = INT( l2-l1 + 1 )
            IF ( l == 0 ) CYCLE !  no entries

!  move first entry of column forward

            IF ( row_out( l1 ) == j ) idiag = idiag + 1
            row_out( kk ) = row_out( l1 )
            val_out( kk ) = val_out( l1 )
            kk = kk + 1
!  loop over remaining entries

            DO ii = l1 + 1, l2
              IF ( row_out( ii ) == row_out( kk - 1 ) ) THEN
                idup = idup + 1
                val_out( kk - 1 ) = val_out( kk - 1 ) + val_out( ii )
                CYCLE
              END IF

!  pull entry forwards

              IF ( row_out( ii ) == j ) idiag = idiag + 1
              row_out( kk ) = row_out( ii )
              val_out( kk ) = val_out( ii )
              kk = kk + 1
            END DO
          END DO
          ptr_out( n + 1 ) = kk
        ELSE
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            l = INT( l2-l1 + 1 )
            IF ( l > 1 ) CALL sort32( row_out( l1 : l2 ), l )
          END DO

!  work through removing duplicates

          kk = 1 !  insert position
          DO j = 1, n
            l1 = ptr_out( j )
            l2 = ptr_out( j + 1 ) - 1
            ptr_out( j ) = kk

!  sort(row_out(l1:l2)) and permute map(l1:l2) accordingly

            l = INT( l2 - l1 + 1 )
            IF ( l == 0 ) CYCLE !  no entries

!  move first entry of column forward

            IF ( row_out( l1 ) == j ) idiag = idiag + 1
            row_out( kk ) = row_out( l1 )
            kk = kk + 1

!  loop over remaining entries

            DO ii = l1 + 1, l2
              IF ( row_out( ii ) == row_out( kk - 1 ) ) THEN
                idup = idup + 1
                CYCLE
              END IF

!  pull entry forwards

              IF ( row_out( ii ) == j ) idiag = idiag + 1
              row_out( kk ) = row_out( ii )
              kk = kk + 1
            END DO
          END DO
          ptr_out( n + 1 ) = kk
        END IF

!  check for missing diagonals in pos def and indef cases
!  Note: change this test for complex case

        IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_PSDEF ) THEN
          DO j = 1, n
            IF ( ptr_out( j ) < ptr_out( n + 1 ) ) THEN
              IF ( row_out( ptr_out( j ) ) /= j ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
            END IF
          END DO
        END IF

!  append duplicates to map

        IF ( PRESENT( map ) ) THEN
          DO WHILE ( associated( duphead ) )
            map( lmap + 1 ) = duphead%dest
            map( lmap + 2 ) = duphead%src
            lmap = lmap + 2
            dup => duphead%next
            DEALLOCATE ( duphead )
            duphead => dup
          END DO
          IF ( PRESENT( val_out ) ) THEN
            ALLOCATE ( val_out( ptr_out( n + 1 ) - 1 ), STAT = st )
            IF ( st /= 0 ) GO TO 100
            CALL MU_apply_conversion_map( matrix_type, lmap, map, val_in, &
              ptr_out( n + 1 ) - 1, val_out )
          END IF
        END IF

        IF ( PRESENT( val_out ) ) THEN
          SELECT CASE ( matrix_type )
          CASE ( SSIDS_MATRIX_REAL_SYM_PSDEF )

!  check for positive diagonal entries

            DO j = 1, n
              kk = ptr_out( j )

!  positive definite - can't reach here unless all entries have diagonal

              IF ( REAL( val_out( kk ) ) <= zero ) THEN
                flag = ERROR_MISSING_DIAGONAL
                CALL print_matrix_flag( context, nout, flag )
                RETURN
              END IF
            END DO
          END SELECT
        END IF

!  check whether a warning needs to be raised

        IF ( ioor > 0 .OR. idup > 0 .OR. idiag < n ) THEN
          IF ( ioor > 0 ) flag = WARNING_IDX_OOR
          IF ( idup > 0 ) flag = WARNING_DUP_IDX
          IF ( idup > 0 .AND. ioor > 0 ) flag = WARNING_DUP_AND_OOR
          IF ( ABS( matrix_type ) /= SSIDS_MATRIX_REAL_SKEW ) THEN
            IF ( idiag < n .AND. ioor > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n .AND. idup > 0 ) THEN
              flag = WARNING_MISS_DIAG_OORDUP
            ELSE IF ( idiag < n ) THEN
              flag = WARNING_MISSING_DIAGONAL
            END IF
          END IF
          CALL print_matrix_flag( context, nout, flag )
        END IF

!  check whether a warning needs to be raised

        IF ( ioor > 0 .OR. idup > 0 .OR. idiag < n ) THEN
          IF ( ioor > 0 ) flag = WARNING_IDX_OOR
          IF ( idup > 0 ) flag = WARNING_DUP_IDX
          IF ( idup > 0 .AND. ioor > 0 ) flag = WARNING_DUP_AND_OOR
          IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_INDEF .AND.         &
               idiag < n .AND. ioor > 0 ) THEN
            flag = WARNING_MISS_DIAG_OORDUP
          ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_INDEF .AND.    &
                    idiag < n .AND. idup > 0 ) THEN
            flag = WARNING_MISS_DIAG_OORDUP
          ELSE IF ( ABS( matrix_type ) == SSIDS_MATRIX_REAL_SYM_INDEF .AND.    &
                    idiag < n ) THEN
            flag = WARNING_MISSING_DIAGONAL
          END IF
          CALL print_matrix_flag( context, nout, flag )
        END IF

        IF ( PRESENT( noor ) ) noor = ioor
        IF ( PRESENT( ndup ) ) ndup = idup
        RETURN

100     CONTINUE
        IF ( st /= 0 ) THEN
          flag = ERROR_ALLOCATION
          CALL print_matrix_flag( context, nout, flag )
        END IF
        RETURN

        END SUBROUTINE convert_coord_to_cscl_ptr64_precision

!-*-*-  G A L A H A D - apply_conversion_map_ptr32   S U B R O U T I N E  -*-*-

        SUBROUTINE apply_conversion_map_ptr32_precision( matrix_type, lmap,    &
                                                         map, val, ne, val_out )

!  this subroutine will use map to translate the values of val to val_out

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: lmap
        INTEGER ( KIND = i4_ ), DIMENSION ( lmap ), INTENT ( IN ) :: map
        REAL ( KIND = rp_ ), DIMENSION ( * ), INTENT ( IN ) :: val
        INTEGER ( KIND = i4_ ), INTENT ( IN ) :: ne
        REAL ( KIND = rp_ ), DIMENSION ( ne ), INTENT ( OUT ) :: val_out

        INTEGER ( KIND = i4_ ) :: i, j, k

        SELECT CASE ( matrix_type )

!  rectangular, unsymmetric or symmetric matrix

        CASE DEFAULT

!  first set val_out using first part of map

          DO i = 1, ne
            j = ABS( map( i ) )
            val_out( i ) = val( j )
          END DO

!  second examine list of duplicates

          DO i = ne + 1, lmap, 2
            j = ABS( map( i ) )
            k = ABS( map( i + 1 ) )
            val_out( j ) = val_out( j ) + val( k )
          END DO

!  skew symmetric matrix

        CASE ( SSIDS_MATRIX_REAL_SKEW )

!  first set val_out using first part of map

          DO i = 1, ne
            j = ABS( map( i ) )
            val_out( i ) = SIGN( 1.0, real( map( i ) ) ) * val( j )
          END DO

!  second examine list of duplicates

          DO i = ne + 1, lmap, 2
            j = ABS( map( i ) )
            k = ABS( map( i + 1 ) )
            val_out( j )                                                       &
              = val_out( j ) + SIGN( 1.0, REAL( map( i + 1 ) ) ) * val( k )
          END DO
        END SELECT
        RETURN

        END SUBROUTINE apply_conversion_map_ptr32_precision

!-*-*-  G A L A H A D - apply_conversion_map_ptr64   S U B R O U T I N E  -*-*-

        SUBROUTINE apply_conversion_map_ptr64_precision( matrix_type, lmap,    &
                                                         map, val, ne, val_out )

!  this subroutine will use map to translate the values of val to val_out

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: matrix_type
        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: lmap
        INTEGER ( KIND = i8_ ), DIMENSION ( lmap ), INTENT ( IN ) :: map
        REAL ( KIND = rp_ ), DIMENSION ( * ), INTENT ( IN ) :: val
        INTEGER ( KIND = i8_ ), INTENT ( IN ) :: ne
        REAL ( KIND = rp_ ), DIMENSION ( ne ), INTENT ( OUT ) :: val_out

        INTEGER ( KIND = i8_ ) :: i, j, k

        SELECT CASE ( matrix_type )

!  rectangular, unsymmetric or symmetric matrix

        CASE DEFAULT

!  first set val_out using first part of map

          DO i = 1, ne
            j = ABS( map( i ) )
            val_out( i ) = val( j )
          END DO

!  second examine list of duplicates

          DO i = ne + 1, lmap, 2
            j = ABS( map( i ) )
            k = ABS( map( i + 1 ) )
            val_out( j ) = val_out( j ) + val( k )
          END DO

!  skew symmetric matrix

        CASE ( SSIDS_MATRIX_REAL_SKEW )

!  first set val_out using first part of map

          DO i = 1, ne
            j = ABS( map( i ) )
            val_out( i ) = SIGN( 1.0, real( map( i ) ) ) * val( j )
          END DO

!  second examine list of duplicates

          DO i = ne + 1, lmap, 2
            j = ABS( map( i ) )
            k = ABS( map( i + 1 ) )
            val_out( j )                                                       &
              = val_out( j ) + SIGN( 1.0, REAL( map( i + 1 ) ) ) * val( k )
          END DO
        END SELECT
        RETURN

        END SUBROUTINE apply_conversion_map_ptr64_precision

!-*-  G A L A H A D - p r i n t _ m a t r i x _f l a g   S U B R O U T I N E  -

        SUBROUTINE print_matrix_flag( context, nout, flag )
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: flag, nout
        CHARACTER ( LEN=* ), OPTIONAL, INTENT ( IN ) :: context

        IF ( nout < 0 ) RETURN
        IF ( flag < 0 ) THEN
          WRITE ( nout, '( / 3A, I3 )' ) ' Error return from ',                &
            trim( context ), '. Error flag = ', flag
        ELSE
          WRITE ( nout, '( / 3A, I3 )' ) ' Warning from ', trim( context ),    &
            '. Warning flag = ', flag
        END IF

        SELECT CASE ( flag )

!   errors

        CASE ( ERROR_ALLOCATION )
          WRITE ( nout, '( A )' ) ' Allocation error'
        CASE ( ERROR_MATRIX_TYPE )
          WRITE ( nout, '( A )' ) ' matrix_type has invalid value'
        CASE ( ERROR_N_OOR )
          WRITE ( nout, '( A )' ) ' m or n is out-of-range'
        CASE ( ERROR_ALL_OOR )
          WRITE ( nout, '( A )' ) ' All entries in a column out-of-range'
        CASE ( ERROR_PTR_MONO )
          WRITE ( nout, '( A )' ) ' ptr not monotonic'
        CASE ( ERROR_PTR_1 )
          WRITE ( nout, '( A )' ) ' ptr( 1 )  <  1'
        CASE ( ERROR_IMAG_DIAGONAL )
          WRITE ( nout, '( A )' ) ' one or more diagonal entries is not real'
        CASE ( ERROR_MISSING_DIAGONAL )
          WRITE ( nout, '( A )' ) ' one or more diagonal entries are not +ve'
        CASE ( ERROR_VAL_MISS )
          WRITE ( nout, '( A )' ) ' Only one of val and val_out is present'
        CASE ( ERROR_LMAP_MISS )
          WRITE ( nout, '( A )' ) ' Only one of lmap and map is present'
        CASE ( ERROR_UPR_ENTRY )
          WRITE ( nout, '( A )' ) ' Entry in upper triangle'
        CASE ( ERROR_M_NE_N )
          WRITE ( nout, '( A )' ) ' m is not equal to n'

!  warnings

        CASE ( WARNING_IDX_OOR )
          WRITE ( nout, '( A )' ) ' out-of-range indices detected'
        CASE ( WARNING_DUP_IDX )
          WRITE ( nout, '( A )' ) ' duplicate entries detected'
        CASE ( WARNING_DUP_AND_OOR )
          WRITE ( nout, '( A )' ) &
            ' out-of-range indices detected and duplicate entries detected'
        CASE ( WARNING_MISSING_DIAGONAL )
          WRITE ( nout, '( A )' ) ' one or more diagonal entries is missing'
        CASE ( WARNING_MISS_DIAG_OORDUP )
          WRITE ( nout, '( A )' ) ' one or more diagonal entries is missing and'
          WRITE ( nout, '( A )' )                                             &
            ' out-of-range and/or duplicate entries detected'
        END SELECT

        END SUBROUTINE print_matrix_flag

!-*-*-*-*-*-*-  G A L A H A D - s o r t 3 2   S U B R O U T I N E  -*-*-*-*-*-*-

        SUBROUTINE sort32( array, n, map, val )

!  sort an integer array by heapsort into ascending order.

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  Size of array to be sorted

!  array to be sorted

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( INOUT ) :: array
        INTEGER ( KIND = i4_ ), DIMENSION ( n ), OPTIONAL,                     &
                                                 INTENT ( INOUT ) :: map

!  apply same permutation to val

        REAL ( KIND = rp_ ), DIMENSION ( n ), OPTIONAL, INTENT ( INOUT ) :: val
        INTEGER ( KIND = ip_ ) :: i, temp
        INTEGER ( KIND = i4_ ) :: temp4
        REAL ( KIND = rp_ ) :: vtemp
        INTEGER ( KIND = ip_ ) :: root

        IF ( n <= 1 ) RETURN !  nothing to do

!  turn array into a heap with largest element on top (as this will be pushed
!  on the end of the array in the next phase). Wwe start at the bottom of the 
!  heap (i.e. 3 above) and work our way upwards ensuring the new "root" of 
!  each subtree is in the correct location

        root = n / 2
        DO root = root, 1, - 1
          CALL pushdown32( root, n, array, val = val, map = map )
        END DO

!  repeatedly take the largest value and swap it to the back of the array
!  then repeat above code to sort the array

        DO i = n, 2, - 1

!  swap a(i) and head of heap a(1)

          temp = array( 1 )
          array( 1 ) = array( i )
          array( i ) = temp
          IF ( PRESENT( val ) ) THEN
            vtemp = val( 1 )
            val( 1 ) = val( i )
            val( i ) = vtemp
          END IF
          IF ( PRESENT( map ) ) THEN
            temp4 = map( 1 )
            map( 1 ) = map( i )
            map( i ) = temp4
          END IF
          CALL pushdown32( 1_ip_, i- 1_ip_, array, val=val, map=map )
        END DO
        END SUBROUTINE sort32

!-*-*-*-*-*-*-  G A L A H A D - s o r t 6 4   S U B R O U T I N E  -*-*-*-*-*-*-

        SUBROUTINE sort64( array, n, map, val )

!  sort an integer array by heapsort into ascending order.

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  Size of array to be sorted

!  array to be sorted

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( INOUT ) :: array
        INTEGER ( KIND = i8_ ), DIMENSION ( n ), OPTIONAL,                     &
                                                 INTENT ( INOUT ) :: map

!  apply same permutation to val

        REAL ( KIND = rp_ ), DIMENSION ( n ), OPTIONAL, INTENT ( INOUT ) :: val

        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = ip_ ) :: temp
        INTEGER ( KIND = i8_ ) :: ltemp
        REAL ( KIND = rp_ ) :: vtemp
        INTEGER ( KIND = ip_ ) :: root

        IF ( n <= 1 ) RETURN !  nothing to do

!  turn array into a heap with largest element on top (as this will be pushed
!  on the end of the array in the next phase). Wwe start at the bottom of the 
!  heap (i.e. 3 above) and work our way upwards ensuring the new "root" of 
!  each subtree is in the correct location

        root = n / 2
        DO root = root, 1, - 1
          CALL pushdown64( root, n, array, val=val, map=map )
        END DO

!  repeatedly take the largest value and swap it to the back of the array
!  then repeat above code to sort the array

        DO i = n, 2, - 1

!  swap a(i) and head of heap a(1)

          temp = array( 1 )
          array( 1 ) = array( i )
          array( i ) = temp
          IF ( PRESENT( val ) ) THEN
            vtemp = val( 1 )
            val( 1 ) = val( i )
            val( i ) = vtemp
          END IF
          IF ( PRESENT( map ) ) THEN
            ltemp = map( 1 )
            map( 1 ) = map( i )
            map( i ) = ltemp
          END IF
          CALL pushdown64( 1_ip_, i- 1_ip_, array, val=val, map=map )
        END DO
        END SUBROUTINE sort64

!-*-*-*-*-  G A L A H A D - p u s h d o w n 3 2   S U B R O U T I N E  -*-*-*-

        SUBROUTINE pushdown32( root, last, array, val, map )

!  this subroutine will assume everything below head is a heap and will
!  push head downwards into the correct location for it

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: root
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: last
        INTEGER ( KIND = ip_ ), DIMENSION ( last ), INTENT ( INOUT ) :: array
        REAL ( KIND = rp_ ), DIMENSION ( last ), OPTIONAL,                     &
                                                 INTENT ( INOUT ) :: val
        INTEGER ( KIND = i4_ ), DIMENSION ( last ), OPTIONAL,                  &
                                                 INTENT ( INOUT ) :: map

        INTEGER ( KIND = ip_ ) :: insert !  current insert position
        INTEGER ( KIND = ip_ ) :: test !  current position to test

!  value of array( root ) at start of iteration

        INTEGER ( KIND = ip_ ) :: root_idx

!  value of val( root ) at start of iteration

        REAL ( KIND = rp_ ) :: root_val

!  value of map( root ) at start of iteration

        INTEGER ( KIND = i4_ ) :: root_map

!  A heap is a ( partial ) binary tree with the property that given a
!  parent and a child, array(child) >= array(parent). If we label as
!                       1
!                     /   \
!                    2     3
!                   / \   / \
!                  4   5 6   7
!  Then node i has nodes 2*i and 2*i + 1 as its children

        IF ( PRESENT( val ) .AND. PRESENT( map ) ) THEN !  both val and map
          root_idx = array( root )
          root_val = val( root )
          root_map = map( root )
          insert = root
          test = 2 * insert

!  first check for largest child branch to descend

          DO WHILE ( test <= last )
            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating value up

            array( insert ) = array( test )
            val( insert ) = val( test )
            map( insert ) = map( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root value into location found

          array( insert ) = root_idx
          val( insert ) = root_val
          map( insert ) = root_map
        ELSE IF ( PRESENT( val ) ) THEN !  val only, not map
          root_idx = array( root )
          root_val = val( root )
          insert = root
          test = 2 * insert
          DO WHILE ( test <= last )

!  first check for largest child branch to descend

            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating value up

            array( insert ) = array( test )
            val( insert ) = val( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root value into location found

          array( insert ) = root_idx
          val( insert ) = root_val
        ELSE IF ( PRESENT( map ) ) THEN !  map only, not val
          root_idx = array( root )
          root_map = map( root )
          insert = root
          test = 2 * insert
          DO WHILE ( test <= last )

!  first check for largest child branch to descend

            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating mapue up

            array( insert ) = array( test )
            map( insert ) = map( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root mapue into location found

          array( insert ) = root_idx
          map( insert ) = root_map
        ELSE !  neither map nor val
          root_idx = array( root )
          insert = root
          test = 2 * insert
          DO WHILE ( test <= last )

!  first check for largest child branch to descend

            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating value up

            array( insert ) = array( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root value into location found

          array( insert ) = root_idx
        END IF

        END SUBROUTINE pushdown32

!-*-*-*-*-  G A L A H A D - p u s h d o w n 6 4   S U B R O U T I N E  -*-*-*-

        SUBROUTINE pushdown64( root, last, array, val, map )

!  this subroutine will assume everything below head is a heap and will
!  push head downwards into the correct location for it

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: root
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: last
        INTEGER ( KIND = ip_ ), DIMENSION ( last ), INTENT ( INOUT ) :: array
        REAL ( KIND = rp_ ), DIMENSION ( last ), OPTIONAL,                     &
                                                 INTENT ( INOUT ) :: val
        INTEGER ( KIND = i8_ ), DIMENSION ( last ), OPTIONAL,                  &
                                                 INTENT ( INOUT ) :: map

        INTEGER ( KIND = ip_ ) :: insert !  current insert position
        INTEGER ( KIND = ip_ ) :: test !  current position to test

!  value of array( root ) at start of iteration

        INTEGER ( KIND = ip_ ) :: root_idx

!  value of val( root ) at start of iteration

        REAL ( KIND = rp_ ) :: root_val

!  value of map( root ) at start of iteration

        INTEGER ( KIND = i8_ ) :: root_map

!  A heap is a ( partial ) binary tree with the property that given a
!  parent and a child, array(child) >= array(parent). If we label as
!                       1
!                     /   \
!                    2     3
!                   / \   / \
!                  4   5 6   7
!  Then node i has nodes 2*i and 2*i + 1 as its children

        IF ( PRESENT( val ) .AND. PRESENT( map ) ) THEN !  both val and map
          root_idx = array( root )
          root_val = val( root )
          root_map = map( root )
          insert = root
          test = 2 * insert

!  first check for largest child branch to descend

          DO WHILE ( test <= last )
            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating value up

            array( insert ) = array( test )
            val( insert ) = val( test )
            map( insert ) = map( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root value into location found

          array( insert ) = root_idx
          val( insert ) = root_val
          map( insert ) = root_map
        ELSE IF ( PRESENT( val ) ) THEN !  val only, not map
          root_idx = array( root )
          root_val = val( root )
          insert = root
          test = 2 * insert
          DO WHILE ( test <= last )

!  first check for largest child branch to descend

            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating value up

            array( insert ) = array( test )
            val( insert ) = val( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root value into location found

          array( insert ) = root_idx
          val( insert ) = root_val
        ELSE IF ( PRESENT( map ) ) THEN !  map only, not val
          root_idx = array( root )
          root_map = map( root )
          insert = root
          test = 2 * insert
          DO WHILE ( test <= last )

!  first check for largest child branch to descend

            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating mapue up

            array( insert ) = array( test )
            map( insert ) = map( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root mapue into location found

          array( insert ) = root_idx
          map( insert ) = root_map
        ELSE !  neither map nor val
          root_idx = array( root )
          insert = root
          test = 2 * insert
          DO WHILE ( test <= last )

!  first check for largest child branch to descend

            IF ( test /= last ) THEN
              IF ( array( test + 1 )>array( test ) ) test = test + 1
            END IF
            IF ( array( test ) <= root_idx ) EXIT !  root gets tested here

!  otherwise, move on to next level down, percolating value up

            array( insert ) = array( test )
            insert = test
            test = 2 * insert
          END DO

!  finally drop root value into location found

          array( insert ) = root_idx
        END IF

        END SUBROUTINE pushdown64

!-*-*-  G A L A H A D - c l e a n u p _ d u p 3 2   S U B R O U T I N E  -*-*-

        SUBROUTINE cleanup_dup32( duphead )

!  NB: can't have both intent() and pointer

        TYPE ( dup_list ), POINTER :: duphead

        TYPE ( dup_list ), POINTER :: dup

        DO WHILE ( associated( duphead ) )
          dup => duphead%next
          DEALLOCATE ( duphead )
          duphead => dup
        END DO
        END SUBROUTINE cleanup_dup32

!-*-*-  G A L A H A D - c l e a n u p _ d u p 6 4   S U B R O U T I N E  -*-*-

        SUBROUTINE cleanup_dup64( duphead )

!  NB: can't be both intent() and pointer

        TYPE ( dup_list64 ), POINTER :: duphead

        TYPE ( dup_list64 ), POINTER :: dup

        DO WHILE ( associated( duphead ) )
          dup => duphead%next
          DEALLOCATE ( duphead )
          duphead => dup
        END DO
        END SUBROUTINE cleanup_dup64

!-  G A L A H A D - h a l f _ t o _ f u l l _ i n t 3 2   S U B R O U T I N E  -

        SUBROUTINE half_to_full_int32( n, row, ptr, iw, a, cbase )

!  generate the expanded structure for a  matrix a with a symmetric sparsity 
!  pattern given the structure for the lower triangular part.
!  Diagonal entries need not be present

        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  holds the order of A.

!  row must be set by the user to hold the row indices of the lower triangular 
!  part of A. the entries of a single column must be contiguous. The entries 
!  of column j must precede those of column j + 1, and there must be no wasted
!  space between columns. row indices within a column may be in any order. On
!  exit, it will have the same meaning but will be changed to hold the row 
!  indices of the entries in the expanded structure.  diagonal entries need 
!  not be present. the new row indices added in the upper triangular part 
!  will be in order for each column and will precede the row indices for the 
!  lower triangular part which will remain in the input order

        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: row( * ) 

!  ptr must be set  by the user so that ptr(j) is the position in row of the 
!  first entry in column j and  ptr(n+1) must be set to one  more than 
!  the total number of entries.  on exit, ptr(j) will have the same meaning 
!  but will be changed to point to the position of the first entry of column
!  j in the expanded structure. the new value of ptr(n+1) will be one greater 
!  than the number of entries in the expanded structure

        INTEGER ( KIND = i4_ ), INTENT ( INOUT ) :: ptr( n + 1 )
        INTEGER ( KIND = ip_ ) :: iw( n ) !  workspace

!  if present, a(1:ptr(n+1)-1) must be set by the user so that a(k) holds the 
!  value of the entry in row(k). on exit, a will hold the values of the entries
!  in the expanded structure corresponding to the output values of row

        REAL ( KIND = rp_ ), OPTIONAL, INTENT ( INOUT ) :: a( * )
        LOGICAL, OPTIONAL, INTENT ( IN ) :: cbase

        INTEGER ( KIND = i4_ ) :: ckp1 !  used as running pointer
        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = i4_ ) :: i1, i2, ii, ipkp1, ipos
        INTEGER ( KIND = ip_ ) :: j
        INTEGER ( KIND = i4_ ) :: jstart

!  number of entries in col. j of original structure

        INTEGER ( KIND = ip_ ) :: lenk

!  number diagonal entries present

        INTEGER ( KIND = ip_ ) :: ndiag

!  number of entries in expanded storage

        INTEGER ( KIND = i4_ ) :: newtau

!  number of entries in symmetric storage

        INTEGER ( KIND = i4_ ) :: oldtau

!  Added to ptr and row to get Fortran base

        INTEGER ( KIND = i4_ ) :: rebase

        rebase = 0
        IF ( PRESENT( cbase ) ) THEN
          IF ( cbase ) rebase = 1
        END IF

        oldtau = ptr( n + 1 ) - 1 + rebase
        iw( 1:n ) = 0

!  iw(j) set to total number entries in col. j of expanded mx.

        ndiag = 0
        DO j = 1, n
          i1 = ptr( j ) + rebase
          i2 = ptr( j + 1 ) - 1 + rebase
          iw( j ) = iw( j ) + i2 - i1 + 1
          DO ii = i1, i2
            i = row( ii ) + rebase
            IF ( i /= j ) THEN
              iw( i ) = iw( i ) + 1
            ELSE
              ndiag = ndiag + 1
            END IF
          END DO
        END DO

        newtau = 2 * oldtau - INT( ndiag, KIND = i4_ )

!  ipkp1 points to position  after end of column being currently processed

        ipkp1 = oldtau + 1

!  ckp1 points to position  after end of same column in expanded structure
        ckp1 = newtau + 1

!  go through the array in the reverse order placing lower triangular
!  elements in appropriate slots

        DO j = n, 1, - 1
          i1 = ptr( j ) + rebase
          i2 = ipkp1
          lenk = i2 - i1

!  jstart is running pointer to position in new structure

          jstart = ckp1

!  set ikp1 for next column

          ipkp1 = i1
          i2 = i2 - 1
!  run through columns in reverse order, lower triangular part of column 
!  moved to end of same column in expanded form

          IF ( PRESENT( a ) ) THEN
            DO ii = i2, i1, - 1
              jstart = jstart - 1
              a( jstart ) = a( ii )
              row( jstart ) = row( ii ) !  rebase cancels
            END DO
          ELSE
            DO ii = i2, i1, - 1
              jstart = jstart - 1
              row( jstart ) = row( ii ) !  rebase cancels
            END DO
          END IF

!  ptr is set to position of first entry in lower triangular part of
!  column j in expanded form

          ptr( j ) = jstart - rebase

!  set ckp1 for next column

          ckp1 = ckp1 - INT( iw( j ), KIND = i4_ )

!  reset iw( j ) to number of entries in lower triangle of column.

          iw( j ) = lenk
        END DO

!  again sweep through the columns in the reverse order, this time when one is 
!  handling column j the upper triangular elements a(j,i) are put in position.

        DO j = n, 1, - 1
          i1 = ptr( j ) + rebase
          i2 = ptr( j ) + INT( iw( j ), KIND = i4_ ) - 1 + rebase

!  run down column in order, note that i is always greater than or equal to j

          IF ( PRESENT( a ) ) THEN
            DO ii = i1, i2
              i = row( ii ) + rebase
              IF ( i == j ) CYCLE
              ptr( i ) = ptr( i ) - 1 !  rebase cancels
              ipos = ptr( i )
              a( ipos ) = a( ii )
              row( ipos ) = j - rebase
            END DO
          ELSE
            DO ii = i1, i2
              i = row( ii ) + rebase
              IF ( i == j ) CYCLE
              ptr( i ) = ptr( i ) - 1 !  rebase cancels
              ipos = ptr( i )
              row( ipos ) = j - rebase
            END DO
          END IF
        END DO
        ptr( n + 1 ) = newtau + 1 - rebase

        END SUBROUTINE half_to_full_int32

!-  G A L A H A D - h a l f _ t o _ f u l l _ i n t 6 4   S U B R O U T I N E  -

        SUBROUTINE half_to_full_int64( n, row, ptr, iw, a, cbase )
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n !  holds the order of A.

!  row must be set by the user to hold the row indices of the lower triangular 
!  part of A. the entries of a single column must be contiguous. The entries 
!  of column j must precede those of column j + 1, and there must be no wasted
!  space between columns. row indices within a column may be in any order. On
!  exit, it will have the same meaning but will be changed to hold the row 
!  indices of the entries in the expanded structure.  diagonal entries need 
!  not be present. the new row indices added in the upper triangular part 
!  will be in order for each column and will precede the row indices for the 
!  lower triangular part which will remain in the input order

        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: row( * ) 

!  ptr must be set  by the user so that ptr(j) is the position in row of the 
!  first entry in column j and  ptr(n+1) must be set to one  more than 
!  the total number of entries.  on exit, ptr(j) will have the same meaning 
!  but will be changed to point to the position of the first entry of column
!  j in the expanded structure. the new value of ptr(n+1) will be one greater 
!  than the number of entries in the expanded structure

        INTEGER ( KIND = i8_ ), INTENT ( INOUT ) :: ptr( n + 1 )
        INTEGER ( KIND = ip_ ) :: iw( n ) !  workspace

!  if present, a(1:ptr(n+1)-1) must be set by the user so that a(k) holds the 
!  value of the entry in row(k). on exit, a will hold the values of the entries
!  in the expanded structure corresponding to the output values of row

        REAL ( KIND = rp_ ), OPTIONAL, INTENT ( INOUT ) :: a( * )
        LOGICAL, OPTIONAL, INTENT ( IN ) :: cbase

        INTEGER ( KIND = i8_ ) :: ckp1 !  used as running pointer
        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = i8_ ) :: i1, i2, ii, ipkp1, ipos
        INTEGER ( KIND = ip_ ) :: j
        INTEGER ( KIND = i8_ ) :: jstart

!  number of entries in col. j of original structure

        INTEGER ( KIND = ip_ ) :: lenk
        INTEGER ( KIND = ip_ ) :: ndiag !  number diagonal entries present

!  number of entries in expanded storage

        INTEGER ( KIND = i8_ ) :: newtau

!  number of entries in symmetric storage

        INTEGER ( KIND = i8_ ) :: oldtau

!  added to ptr and row to get Fortran base

        INTEGER ( KIND = ip_ ) :: rebase

        rebase = 0
        IF ( PRESENT( cbase ) ) THEN
          IF ( cbase ) rebase = 1
        END IF

        oldtau = ptr( n + 1 ) - 1 + rebase
        iw( 1:n ) = 0

!  iw(j) set to total number entries in column j of expanded matrix

        ndiag = 0
        DO j = 1, n
          i1 = ptr( j ) + rebase
          i2 = ptr( j + 1 ) - 1 + rebase
          iw( j ) = iw( j ) + INT( i2-i1 ) + 1
          DO ii = i1, i2
            i = row( ii ) + rebase
            IF ( i /= j ) THEN
              iw( i ) = iw( i ) + 1
            ELSE
              ndiag = ndiag + 1
            END IF
          END DO
        END DO

        newtau = 2 * oldtau - ndiag

!  ipkp1 points to position  after end of column being currently processed

        ipkp1 = oldtau + 1

!  ckp1 points to position  after end of same column in expanded structure

        ckp1 = newtau + 1

!  go through the array in the reverse order placing lower triangular
!  elements in  appropriate slots.

        DO j = n, 1, - 1
          i1 = ptr( j ) + rebase
          i2 = ipkp1
          lenk = INT( i2 - i1 )

!  jstart is running pointer to position in new structure

          jstart = ckp1

!  set ikp1 for next column

          ipkp1 = i1
          i2 = i2 - 1

!  run through columns in reverse order; lower triangular part of column
!  moved to the end of same column in expanded form

          IF ( PRESENT( a ) ) THEN
            DO ii = i2, i1, - 1
              jstart = jstart - 1
              a( jstart ) = a( ii )
              row( jstart ) = row( ii ) !  rebase cancels
            END DO
          ELSE
            DO ii = i2, i1, - 1
              jstart = jstart - 1
              row( jstart ) = row( ii ) !  rebase cancels
            END DO
          END IF

!  ptr is set to position of first entry in lower triangular part of
!  column j in expanded form

          ptr( j ) = jstart - rebase

!  set ckp1 for next column

          ckp1 = ckp1 - iw( j )

!  reset iw( j ) to number of entries in lower triangle of column.

          iw( j ) = lenk
        END DO

!  again sweep through the columns in the reverse order, this
!  time when one is handling column j the upper triangular
!  elements a(j,i) are put in position.

        DO j = n, 1, - 1
          i1 = ptr( j ) + rebase
          i2 = ptr( j ) + iw( j ) - 1 + rebase

!  run down column in order
!  note that i is always greater than or equal to j

          IF ( PRESENT( a ) ) THEN
            DO ii = i1, i2
              i = row( ii ) + rebase
              IF ( i == j ) CYCLE
              ptr( i ) = ptr( i ) - 1 !  rebase cancels
              ipos = ptr( i ) + rebase
              a( ipos ) = a( ii )
              row( ipos ) = j - rebase
            END DO
          ELSE
            DO ii = i1, i2
              i = row( ii ) + rebase
              IF ( i == j ) CYCLE
              ptr( i ) = ptr( i ) - 1 !  rebase cancels
              ipos = ptr( i ) + rebase
              row( ipos ) = j - rebase
            END DO
          END IF
        END DO
        ptr( n + 1 ) = newtau + 1 - rebase
        RETURN

        END SUBROUTINE half_to_full_int64

      END MODULE GALAHAD_MU_precision

