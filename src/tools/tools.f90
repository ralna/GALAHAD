! THIS VERSION: GALAHAD 3.3 - 28/01/2020 AT 08:30 GMT.
! 30/06/2003: procedures _is_are and _s moved from QPA.
! 23/07/2003: invalid "END INTERFACE" arguments removed
! 17/11/2009: procedures _char_integer and _char_real imported from OTHERS.
! 20/05/2010: procedure y_ies added

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*        TOOLS   M O D U L E        *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Copyright reserved: Nick Gould and Philippe Toint, for GALAHAD productions
!  July 2001
!
!               +---------------------------------------------+
!               |                                             |
!               |         Provides various simple tools       |
!               |                                             |
!               +---------------------------------------------+
!

   MODULE GALAHAD_TOOLS

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

!     Make everything private by default

      PRIVATE

!     Make the tools public

      PUBLIC :: TOOLS_output_vector
      PUBLIC :: TOOLS_output_matrix_C, TOOLS_output_matrix_S,                  &
                TOOLS_output_matrix_D, TOOLS_is_are, TOOLS_y_ies, TOOLS_s,     &
                TOOLS_vs, TOOLS_expon, TOOLS_char_integer, TOOLS_char_real,    &
                TOOLS_ordinal

!     Outputs a vector with a reasonable layout.

      INTERFACE TOOLS_output_vector
         MODULE PROCEDURE TOOLS_output_vector_integer,                         &
                          TOOLS_output_vector_real_sp,                         &
                          TOOLS_output_vector_real_dp
      END INTERFACE

!     Outputs a matrix with a reasonable layout.

      INTERFACE TOOLS_output_matrix_C
         MODULE PROCEDURE TOOLS_output_matrix_real_C_sp,                       &
                          TOOLS_output_matrix_real_C_dp
      END INTERFACE

      INTERFACE TOOLS_output_matrix_S
         MODULE PROCEDURE TOOLS_output_matrix_real_S_sp,                       &
                          TOOLS_output_matrix_real_S_dp
      END INTERFACE

      INTERFACE TOOLS_output_matrix_D
         MODULE PROCEDURE TOOLS_output_matrix_real_D_sp,                       &
                          TOOLS_output_matrix_real_D_dp
      END INTERFACE

      INTERFACE TOOLS_expon
         MODULE PROCEDURE TOOLS_expon_sp, TOOLS_expon_dp
      END INTERFACE

      INTERFACE TOOLS_char_real
         MODULE PROCEDURE TOOLS_char_real_sp, TOOLS_char_real_dp
      END INTERFACE

!-------------------------------------------------------------------------------
!   P r e c i s i o n
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0E+0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )

!-------------------------------------------------------------------------------
!   O t h e r s
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: OK = 0

   CONTAINS

!==============================================================================
!==============================================================================


      SUBROUTINE TOOLS_output_vector_real_sp( n, x, out )

!     Print x

!     Arguments:

      INTEGER, INTENT( IN ) :: n

!             The dimension of x.

      REAL ( KIND = sp ), DIMENSION( n ), INTENT( IN ) :: x

!             The vector to print.

      INTEGER, INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!     Local variable

      INTEGER :: j, i

      WRITE( out, 101 )
      j = 1
      DO i = 1, n / 5
         WRITE( out, 100 ) j, x( j:j+4 )
         j = j + 5
      END DO
      IF ( j <= n ) WRITE( out, 100 ) j, x( j:n )
      WRITE( out, 101 )

      RETURN

100   FORMAT( 1x, i4, 5( 1x, 1pE14.6 ) )
101   FORMAT( / )

      END SUBROUTINE TOOLS_output_vector_real_sp

!==============================================================================
!==============================================================================
      SUBROUTINE TOOLS_output_vector_real_dp( n, x, out )

!     Print x

!     Arguments:

      INTEGER, INTENT( IN ) :: n

!             The dimension of x.

      REAL ( KIND = dp ), DIMENSION( n ), INTENT( IN ) :: x

!             The vector to print.

      INTEGER, INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!     Local variable

      INTEGER :: j, i

      WRITE( out, 101 )
      j = 1
      DO i = 1, n / 5
         WRITE( out, 100 ) j, x( j:j+4 )
         j = j + 5
      END DO
      IF ( j <= n ) WRITE( out, 100 ) j, x( j:n )
      WRITE( out, 101 )

      RETURN

100   FORMAT( 1x, i4, 5( 1x, 1pE14.6 ) )
101   FORMAT( / )

      END SUBROUTINE TOOLS_output_vector_real_dp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_vector_integer( n, ix, out )

!     Print ix

!     Arguments:

      INTEGER, INTENT( IN ) :: n

!             The dimension of ix.

      INTEGER, DIMENSION( n ), INTENT( IN ) :: ix

!             The vector to print.

      INTEGER, INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: j, i

      WRITE( out, 101 )
      j = 1
      DO i = 1, n / 10
         WRITE( out, 100 ) j, ix( j:j+9 )
         j = j + 10
      END DO
      IF ( j <= n ) WRITE( out, 100 ) j, ix( j:n )
      WRITE( out, 101 )

      RETURN

100   FORMAT( 1x, i4, 2x, 10( 1x, i5 ) )
101   FORMAT( / )

      END SUBROUTINE TOOLS_output_vector_integer

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_C_sp( nnz, A_val, A_row, A_col, out )

      INTEGER, INTENT( IN ) :: nnz
      REAL( KIND = sp ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_row
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER, INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, kk

      WRITE( out, 102 )
      k = 0
      DO kk = 1, nnz / 3
         WRITE( out, 100 ) A_row( k + 1 ), A_col( k + 1 ), A_val( k + 1 ),     &
                           A_row( k + 2 ), A_col( k + 2 ), A_val( k + 2 ),     &
                           A_row( k + 3 ), A_col( k + 3 ), A_val( k + 3 )
         k = k + 3
      END DO
      IF ( k < nnz ) THEN
         SELECT CASE ( nnz - k )
         CASE ( 1 )
            WRITE ( out, 100 ) A_row( nnz ), A_col( nnz ), A_val( nnz )
         CASE ( 2 )
            WRITE ( out, 100 ) A_row( k + 1 ), A_col( k + 1 ), A_val( k + 1  ),&
                               A_row( nnz ), A_col( nnz ), A_val( nnz )
         END SELECT
      END IF
      WRITE( out, 101 )

      RETURN

100   FORMAT( 2( 1x, i4), 2x, 1pE12.4, 2( 4x, 2( 1x, i4), 2x, 1pE12.4 ) )
101   FORMAT( / )
102   FORMAT( /,1x,'   i    j       value  ',2(5x,'   i    j       value  '),/)

      END SUBROUTINE TOOLS_output_matrix_real_C_sp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_S_sp( nnz, A_val, A_ptr, A_col, out )
      INTEGER, INTENT( IN ) :: nnz
      REAL( KIND = sp ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_ptr
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER, INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, kk, i, i1, i2, i3

      WRITE( out, 102 )
      k = 0
      i = 1
      DO kk = 1, nnz / 3
         DO
            IF ( k + 1  == A_ptr( i + 1 ) ) EXIT
            i = i + 1
         END DO
         i1 = i
         DO
            IF ( k + 2  == A_ptr( i + 1 ) ) EXIT
            i = i + 1
         END DO
         i2 = i
         DO
            IF ( k + 3  == A_ptr( i + 1 ) ) EXIT
            i = i + 1
         END DO
         i3 = i
         WRITE( out, 100 ) i1, A_col( k + 1 ), A_val( k + 1 ),                 &
                           i2, A_col( k + 2 ), A_val( k + 2 ),                 &
                           i3, A_col( k + 3 ), A_val( k + 3 )
         k = k + 3
      END DO
      IF ( k < nnz ) THEN
         SELECT CASE ( nnz - k )
         CASE ( 1 )
            DO
               IF ( nnz  == A_ptr( i + 1 ) ) EXIT
               i = i + 1
            END DO
            WRITE ( out, 100 ) i, A_col( nnz ), A_val( nnz )
         CASE ( 2 )
            DO
               IF ( k + 1  == A_ptr( i + 1 ) ) EXIT
               i = i + 1
            END DO
            i1 = i
            DO
               IF ( nnz == A_ptr( i + 1 ) ) EXIT
               i = i + 1
            END DO
            i2 = i
            WRITE ( out, 100 ) i1, A_col( k + 1 ), A_val( k + 1  ),            &
                               i2, A_col( nnz ), A_val( nnz )
         END SELECT
      END IF
      WRITE( out, 101 )

      RETURN

100   FORMAT( 2( 1x, i4), 2x, 1pE12.4, 2( 4x, 2( 1x, i4), 2x, 1pE12.4 ) )
101   FORMAT( / )
102   FORMAT(/,1x,'   i    j       value  ',2(5x,'   i    j       value  '),/ )

      END SUBROUTINE TOOLS_output_matrix_real_S_sp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_D_sp( nrow, ncol, A_val, sym, out )
      INTEGER, INTENT( IN ) :: nrow, ncol
      REAL( KIND = sp ), DIMENSION( : ), INTENT( IN ) :: A_val
      LOGICAL, INTENT( IN ) :: sym
      INTEGER, INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, kk, i, i1, i2, j, j1, j2, nval, lrow

      WRITE( out, 102 )
      nval = nrow * ncol
      k = 0
      i = 1
      j = 0
      IF ( sym ) THEN
         lrow = 1
      ELSE
         lrow = ncol
      END IF
      DO kk = 1, nval / 3
         j = j + 1
         IF ( j > lrow ) THEN
            i = i + 1
            IF ( sym ) lrow = i
            j = 1
         END IF
         i1 = i
         j1 = j
         j  = j + 1
         IF ( j > lrow ) THEN
            i = i + 1
            IF ( sym ) lrow = i
            j = 1
         END IF
         i2 = i
         j2 = j
         j  = j + 1
         IF ( j > lrow ) THEN
            i = i + 1
            IF ( sym ) lrow = i
            j = 1
         END IF
         WRITE( out, 100 ) i1, j1, A_val( k + 1 ), i2, j2, A_val( k + 2 ),     &
                           i , j , A_val( k + 3 )
         k = k + 3
      END DO
      IF ( k < nval ) THEN
         SELECT CASE ( nval - k )
         CASE ( 1 )
            j = j + 1
            IF ( j > lrow ) THEN
               i = i + 1
               IF ( sym ) lrow = i
               j = 1
            END IF
            WRITE ( out, 100 ) i, j, A_val( nval )
         CASE ( 2 )
            j = j + 1
            IF ( j > lrow ) THEN
               i = i + 1
               IF ( sym ) lrow = i
               j = 1
            END IF
            i1 = i
            j1 = j
            j  = j + 1
            IF ( j > lrow ) THEN
               i = i + 1
               IF ( sym ) lrow = i
               j = 1
            END IF
            WRITE ( out, 100 ) i1, j1, A_val( k + 1  ),            &
                               i , j , A_val( nval )
         END SELECT
      END IF
      WRITE( out, 101 )

      RETURN

100   FORMAT( 2( 1x, i4), 2x, 1pE12.4, 2( 4x, 2( 1x, i4), 2x, 1pE12.4 ) )
101   FORMAT( / )
102   FORMAT(/,1x,'   i    j       value  ',2(5x,'   i    j       value  '),/)

      END SUBROUTINE TOOLS_output_matrix_real_D_sp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_C_dp( nnz, A_val, A_row, A_col, out )
      INTEGER, INTENT( IN ) :: nnz
      REAL( KIND = dp ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_row
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER, INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, kk

      WRITE( out, 102 )
      k = 0
      DO kk = 1, nnz / 3
         WRITE( out, 100 ) A_row( k + 1 ), A_col( k + 1 ), A_val( k + 1 ),     &
                           A_row( k + 2 ), A_col( k + 2 ), A_val( k + 2 ),     &
                           A_row( k + 3 ), A_col( k + 3 ), A_val( k + 3 )
         k = k + 3
      END DO
      IF ( k < nnz ) THEN
         SELECT CASE ( nnz - k )
         CASE ( 1 )
            WRITE ( out, 100 ) A_row( nnz ), A_col( nnz ), A_val( nnz )
         CASE ( 2 )
            WRITE ( out, 100 ) A_row( k + 1 ), A_col( k + 1 ), A_val( k + 1  ),&
                               A_row( nnz ), A_col( nnz ), A_val( nnz )
         END SELECT
      END IF
      WRITE( out, 101 )

      RETURN

100   FORMAT( 2( 1x, i4), 2x, 1pD12.4, 2( 4x, 2( 1x, i4), 2x, 1pD12.4 ) )
101   FORMAT( / )
102   FORMAT( /,1x,'   i    j       value  ',2(5x,'   i    j       value  '),/)

      END SUBROUTINE TOOLS_output_matrix_real_C_dp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_S_dp( nnz, A_val, A_ptr, A_col, out )
      INTEGER, INTENT( IN ) :: nnz
      REAL( KIND = dp ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_ptr
      INTEGER, DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER, INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, kk, i, i1, i2, i3

      WRITE( out, 102 )
      k = 0
      i = 1
      DO kk = 1, nnz / 3
         DO
            IF ( k + 1  == A_ptr( i + 1 ) ) EXIT
            i = i + 1
         END DO
         i1 = i
         DO
            IF ( k + 2  == A_ptr( i + 1 ) ) EXIT
            i = i + 1
         END DO
         i2 = i
         DO
            IF ( k + 3  == A_ptr( i + 1 ) ) EXIT
            i = i + 1
         END DO
         i3 = i
         WRITE( out, 100 ) i1, A_col( k + 1 ), A_val( k + 1 ),                 &
                           i2, A_col( k + 2 ), A_val( k + 2 ),                 &
                           i3, A_col( k + 3 ), A_val( k + 3 )
         k = k + 3
      END DO
      IF ( k < nnz ) THEN
         SELECT CASE ( nnz - k )
         CASE ( 1 )
            DO
               IF ( nnz  == A_ptr( i + 1 ) ) EXIT
               i = i + 1
            END DO
            WRITE ( out, 100 ) i, A_col( nnz ), A_val( nnz )
         CASE ( 2 )
            DO
               IF ( k + 1  == A_ptr( i + 1 ) ) EXIT
               i = i + 1
            END DO
            i1 = i
            DO
               IF ( nnz == A_ptr( i + 1 ) ) EXIT
               i = i + 1
            END DO
            i2 = i
            WRITE ( out, 100 ) i1, A_col( k + 1 ), A_val( k + 1  ),            &
                               i2, A_col( nnz ), A_val( nnz )
         END SELECT
      END IF
      WRITE( out, 101 )

      RETURN

100   FORMAT( 2( 1x, i4), 2x, 1pD12.4, 2( 4x, 2( 1x, i4), 2x, 1pD12.4 ) )
101   FORMAT( / )
102   FORMAT(/,1x,'   i    j       value  ',2(5x,'   i    j       value  '),/ )

      END SUBROUTINE TOOLS_output_matrix_real_S_dp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_D_dp( nrow, ncol, A_val, sym, out )
      INTEGER, INTENT( IN ) :: nrow, ncol
      REAL( KIND = dp ), DIMENSION( : ), INTENT( IN ) :: A_val
      LOGICAL, INTENT( IN ) :: sym
      INTEGER, INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, kk, i, i1, i2, j, j1, j2, nval, lrow

      WRITE( out, 102 )
      nval = nrow * ncol
      k = 0
      i = 1
      j = 0
      IF ( sym ) THEN
         lrow = 1
      ELSE
         lrow = ncol
      END IF
      DO kk = 1, nval / 3
         j = j + 1
         IF ( j > lrow ) THEN
            i = i + 1
            IF ( sym ) lrow = i
            j = 1
         END IF
         i1 = i
         j1 = j
         j  = j + 1
         IF ( j > lrow ) THEN
            i = i + 1
            IF ( sym ) lrow = i
            j = 1
         END IF
         i2 = i
         j2 = j
         j  = j + 1
         IF ( j > lrow ) THEN
            i = i + 1
            IF ( sym ) lrow = i
            j = 1
         END IF
         WRITE( out, 100 ) i1, j1, A_val( k + 1 ), i2, j2, A_val( k + 2 ),     &
                           i , j , A_val( k + 3 )
         k = k + 3
      END DO
      IF ( k < nval ) THEN
         SELECT CASE ( nval - k )
         CASE ( 1 )
            j = j + 1
            IF ( j > lrow ) THEN
               i = i + 1
               IF ( sym ) lrow = i
               j = 1
            END IF
            WRITE ( out, 100 ) i, j, A_val( nval )
         CASE ( 2 )
            j = j + 1
            IF ( j > lrow ) THEN
               i = i + 1
               IF ( sym ) lrow = i
               j = 1
            END IF
            i1 = i
            j1 = j
            j  = j + 1
            IF ( j > lrow ) THEN
               i = i + 1
               IF ( sym ) lrow = i
               j = 1
            END IF
            WRITE ( out, 100 ) i1, j1, A_val( k + 1  ),            &
                               i , j , A_val( nval )
         END SELECT
      END IF
      WRITE( out, 101 )

      RETURN

100   FORMAT( 2( 1x, i4), 2x, 1pD12.4, 2( 4x, 2( 1x, i4), 2x, 1pD12.4 ) )
101   FORMAT( / )
102   FORMAT(/,1x,'   i    j       value  ',2(5x,'   i    j       value  '),/)

      END SUBROUTINE TOOLS_output_matrix_real_D_dp

!==============================================================================
!==============================================================================

!  Function that returns ordinal suffices (i.e., "st", "nd", "rd", "th")
!  This is used for i/o purposes
!  Nick Gould, 2010

      FUNCTION TOOLS_ordinal( num )
      CHARACTER ( len = 2 ) :: TOOLS_ordinal
      INTEGER :: num
      SELECT CASE( MOD( num, 100 ) )
      CASE( 1, 21, 31, 41, 51, 61, 71, 81, 91 )
        TOOLS_ordinal = 'st'
      CASE( 2, 22, 32, 42, 52, 62, 72, 82, 92 )
        TOOLS_ordinal = 'nd'
      CASE( 3, 23, 33, 43, 53, 63, 73, 83, 93 )
        TOOLS_ordinal = 'rd'
      CASE DEFAULT
        TOOLS_ordinal = 'th'
      END SELECT
      RETURN
      END FUNCTION TOOLS_ordinal

!==============================================================================

!  Function that returns "is" for 1 item, "are" for any other number of items.
!  This is used for i/o purposes
!  Nick Gould, 1999

      FUNCTION TOOLS_is_are( num )
      CHARACTER ( len = 3 ) :: TOOLS_is_are
      INTEGER, INTENT( IN ) :: num
      IF ( num /= 1 ) THEN
        TOOLS_is_are = 'are'
      ELSE
        TOOLS_is_are = 'is '
      END IF
      RETURN
      END FUNCTION TOOLS_is_are

!==============================================================================
!==============================================================================

!  Function that returns " " for 1 item, "s" for any other number of items
!  This is used for i/o purposes
!  Nick Gould, 1999

      FUNCTION TOOLS_s( num )
      CHARACTER ( len = 1 ) :: TOOLS_s
      INTEGER, INTENT( IN ) :: num
      IF ( num /= 1 ) THEN
        TOOLS_s = 's'
      ELSE
        TOOLS_s = ' '
      END IF
      RETURN
      END FUNCTION TOOLS_s

!==============================================================================
!==============================================================================

!  Function that returns "s" for 1 item, " " for any other number of items
!  This is used for i/o purposes
!  Nick Gould, 1999

      FUNCTION TOOLS_vs( num )
      CHARACTER ( len = 1 ) :: TOOLS_vs
      INTEGER, INTENT( IN ) :: num
      IF ( num /= 1 ) THEN
        TOOLS_vs = ' '
      ELSE
        TOOLS_vs = 's'
      END IF
      RETURN
      END FUNCTION TOOLS_vs

!==============================================================================
!==============================================================================

!  Function that returns "y" for 1 item, "ies" for any other number of items.
!  This is used for i/o purposes
!  Nick Gould, 2010

      FUNCTION TOOLS_y_ies( num )
      CHARACTER ( len = 3 ) :: TOOLS_y_ies
      INTEGER, INTENT( IN ) :: num
      IF ( num == 1 ) THEN
        TOOLS_y_ies = 'y  '
      ELSE
        TOOLS_y_ies = 'ies'
      END IF
      RETURN
      END FUNCTION TOOLS_y_ies

!==============================================================================
!==============================================================================

     FUNCTION TOOLS_char_real_sp( re )
     CHARACTER ( LEN = 7 ) :: TOOLS_char_real_sp

!  Obtain a 7 character representation of the single-precision real r
!  Extracted from LANCELOT module OTHERS
!  Nick Gould, 1995

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = sp ), INTENT( IN ) :: re

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: ir
     REAL ( KIND = sp ) :: r, rm, rh, rd
     CHARACTER :: cr * 7

     TOOLS_char_real_sp( 1 : 7 ) = '       '
     r = re
     rm = re / 60.0_sp
     rh = rm / 60.0_sp
     rd = rh / 24.0_sp
     IF ( r <= 9999.9_sp ) THEN
        r = re
        WRITE( UNIT = cr, FMT = 2000 ) r
        TOOLS_char_real_sp = cr
     ELSE IF ( r <= 99999.9_sp ) THEN
        r = re
        WRITE( UNIT = cr, FMT = 2000 ) r
        TOOLS_char_real_sp( 1 : 1 ) = ' '
        TOOLS_char_real_sp( 2 : 7 ) = cr( 1 : 6 )
     ELSE IF ( r <= 999999.0_sp ) THEN
        ir = INT(re)
        WRITE( UNIT = cr, FMT = 2010 ) ir
        TOOLS_char_real_sp = cr
     ELSE IF ( rm <= 99999.9_sp ) THEN
        ir = INT( rm )
        WRITE( UNIT = cr( 1 : 6 ), FMT = 2020 ) ir
        TOOLS_char_real_sp = cr( 1 : 6 ) // 'm'
     ELSE IF ( rh <= 99999.9_sp ) THEN
        ir = INT( rh )
        WRITE( UNIT = cr( 1 : 6 ), FMT = 2020 ) ir
        TOOLS_char_real_sp = cr( 1 : 6 ) // 'h'
     ELSE IF ( rd <= 99999.9_sp ) THEN
        ir = INT( rd )
        WRITE( UNIT = cr( 1 : 6 ), FMT = 2020 ) ir
        TOOLS_char_real_sp = cr( 1 : 6 ) // 'd'
     ELSE
        TOOLS_char_real_sp = ' ******'
     END IF

     RETURN

!  Non-executable statements

 2000 FORMAT( 0P, F7.1 )
 2010 FORMAT( I7 )
 2020 FORMAT( I6 )

!  End of TOOLS_char_real_sp

     END FUNCTION TOOLS_char_real_sp

!==============================================================================
!==============================================================================

     FUNCTION TOOLS_char_real_dp( re )
     CHARACTER ( LEN = 7 ) :: TOOLS_char_real_dp

!  Obtain a 7 character representation of the double-precision real r
!  Extracted from LANCELOT module OTHERS
!  Nick Gould, 1995

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = dp ), INTENT( IN ) :: re

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: ir
     REAL ( KIND = dp ) :: r, rm, rh, rd
     CHARACTER :: cr * 7

     TOOLS_char_real_dp( 1 : 7 ) = '       '
     r = re
     rm = re / 60.0_dp
     rh = rm / 60.0_dp
     rd = rh / 24.0_dp
     IF ( r <= 9999.9_dp ) THEN
        r = re
        WRITE( UNIT = cr, FMT = 2000 ) r
        TOOLS_char_real_dp = cr
     ELSE IF ( r <= 99999.9_dp ) THEN
        r = re
        WRITE( UNIT = cr, FMT = 2000 ) r
        TOOLS_char_real_dp( 1 : 1 ) = ' '
        TOOLS_char_real_dp( 2 : 7 ) = cr( 1 : 6 )
     ELSE IF ( r <= 999999.0_dp ) THEN
        ir = INT(re)
        WRITE( UNIT = cr, FMT = 2010 ) ir
        TOOLS_char_real_dp = cr
     ELSE IF ( rm <= 99999.9_dp ) THEN
        ir = INT( rm )
        WRITE( UNIT = cr( 1 : 6 ), FMT = 2020 ) ir
        TOOLS_char_real_dp = cr( 1 : 6 ) // 'm'
     ELSE IF ( rh <= 99999.9_dp ) THEN
        ir = INT( rh )
        WRITE( UNIT = cr( 1 : 6 ), FMT = 2020 ) ir
        TOOLS_char_real_dp = cr( 1 : 6 ) // 'h'
     ELSE IF ( rd <= 99999.9_dp ) THEN
        ir = INT( rd )
        WRITE( UNIT = cr( 1 : 6 ), FMT = 2020 ) ir
        TOOLS_char_real_dp = cr( 1 : 6 ) // 'd'
     ELSE
        TOOLS_char_real_dp = ' ******'
     END IF

     RETURN

!  Non-executable statements

 2000 FORMAT( 0P, F7.1 )
 2010 FORMAT( I7 )
 2020 FORMAT( I6 )

!  End of TOOLS_char_real_dp

     END FUNCTION TOOLS_char_real_dp

!==============================================================================
!==============================================================================

     FUNCTION TOOLS_expon_sp( r )
     CHARACTER ( LEN = 3 ) :: TOOLS_expon_sp

!  Obtain a 3 character representation of the exponent of the single precision
!  real r. exponents outside [-99,+99] will be represented as +in or -in as
!  relevant
!  Nick Gould, 2010

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = sp ) :: r

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

     IF ( r == 0.0_sp ) THEN
       WRITE( TOOLS_expon_sp, "( '-in' )" )
     ELSE
       i = FLOOR( LOG10( r ) )
       IF ( i <= 99 .AND. i >= - 99 ) THEN
         WRITE( TOOLS_expon_sp, "( SP, I3 )" ) i
       ELSE IF ( i < - 99 ) THEN
         WRITE( TOOLS_expon_sp, "( '-in' )" )
       ELSE
         WRITE( TOOLS_expon_sp, "( '+in' )" )
       END IF
     END IF
     RETURN

!  End of TOOLS_expon_sp

     END FUNCTION TOOLS_expon_sp

!==============================================================================
!==============================================================================

     FUNCTION TOOLS_expon_dp( r )
     CHARACTER ( LEN = 3 ) :: TOOLS_expon_dp

!  Obtain a 3 character representation of the exponent of the single precision
!  real r. exponents outside [-99,+99] will be represented as +in or -in as
!  relevant
!  Nick Gould, 2010

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = dp ) :: r

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

     IF ( r == 0.0_dp ) THEN
       WRITE( TOOLS_expon_dp, "( '-in' )" )
     ELSE
       i = FLOOR( LOG10( r ) )
       IF ( i <= 99 .AND. i >= - 99 ) THEN
         WRITE( TOOLS_expon_dp, "( SP, I3 )" ) i
       ELSE IF ( i < - 99 ) THEN
         WRITE( TOOLS_expon_dp, "( '-in' )" )
       ELSE
         WRITE( TOOLS_expon_dp, "( '+in' )" )
       END IF
     END IF
     RETURN

!  End of TOOLS_expon_dp

     END FUNCTION TOOLS_expon_dp

!==============================================================================
!==============================================================================

     FUNCTION TOOLS_char_integer( i )
     CHARACTER ( LEN = 6 ) :: TOOLS_char_integer

!  Obtain a 6 character representation of the integer i.
!  Extracted from LANCELOT module OTHERS
!  Nick Gould, 1995

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: i

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: ik, im, ig
     CHARACTER :: ci * 6

     TOOLS_char_integer( 1 : 6 ) = '       '
     ik = i / 1000
     im = ik / 1000
     ig = im / 1000
     IF ( i <= 99999 ) THEN
        WRITE( UNIT = ci, FMT = 2010 ) i
        TOOLS_char_integer = ci
     ELSE IF ( i <= 9999999 ) THEN
        WRITE( UNIT = ci( 1 : 5 ), FMT = 2020 ) ik
        TOOLS_char_integer = ci( 1 : 5 ) // 'k'
     ELSE IF ( i <= 999999999 ) THEN
        WRITE( UNIT = ci( 1 : 5 ), FMT = 2020 ) im
        TOOLS_char_integer = ci( 1 : 5 ) // 'm'
     ELSE
        WRITE( UNIT = ci( 1 : 5 ), FMT = 2020 ) ig
        TOOLS_char_integer = ci( 1 : 5 ) // 'g'
     END IF

     RETURN

!  Non-executable statements

 2010 FORMAT( I6 )
 2020 FORMAT( I5 )

!  End of TOOLS_char_integer

     END FUNCTION TOOLS_char_integer

!==============================================================================
!==============================================================================

   END MODULE GALAHAD_TOOLS

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*    END TOOLS  M O D U L E   *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
