! THIS VERSION: GALAHAD 5.1 - 2024-11-22 AT 09:20 GMT.

#include "galahad_modules.h"

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

#ifdef REAL_128
     USE GALAHAD_KINDS, ONLY : ip_, sp_, dp_, qp_
#else
     USE GALAHAD_KINDS, ONLY : ip_, sp_, dp_
#endif

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

!     Make everything private by default

      PRIVATE

!     Make the tools public

      PUBLIC :: TOOLS_output_matrix_C, TOOLS_output_matrix_S,                  &
                TOOLS_output_matrix_D, TOOLS_output_vector

!     Outputs a vector with a reasonable layout.

#ifdef REAL_128
      INTERFACE TOOLS_output_vector
         MODULE PROCEDURE TOOLS_output_vector_integer,                         &
                          TOOLS_output_vector_real_sp,                         &
                          TOOLS_output_vector_real_dp,                         &
                          TOOLS_output_vector_real_qp
      END INTERFACE

!     Outputs a matrix with a reasonable layout.

      INTERFACE TOOLS_output_matrix_C
         MODULE PROCEDURE TOOLS_output_matrix_real_C_sp,                       &
                          TOOLS_output_matrix_real_C_dp,                       &
                          TOOLS_output_matrix_real_C_qp
      END INTERFACE

      INTERFACE TOOLS_output_matrix_S
         MODULE PROCEDURE TOOLS_output_matrix_real_S_sp,                       &
                          TOOLS_output_matrix_real_S_dp,                       &
                          TOOLS_output_matrix_real_S_qp
      END INTERFACE

      INTERFACE TOOLS_output_matrix_D
         MODULE PROCEDURE TOOLS_output_matrix_real_D_sp,                       &
                          TOOLS_output_matrix_real_D_dp,                       &
                          TOOLS_output_matrix_real_D_qp
      END INTERFACE
#else
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
#endif

   CONTAINS

!==============================================================================
!==============================================================================


      SUBROUTINE TOOLS_output_vector_real_sp( n, x, out )

!     Print x

!     Arguments:

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n

!             The dimension of x.

      REAL ( KIND = sp_ ), DIMENSION( n ), INTENT( IN ) :: x

!             The vector to print.

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!     Local variable

      INTEGER ( KIND = ip_ ) :: j, i

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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n

!             The dimension of x.

      REAL ( KIND = dp_ ), DIMENSION( n ), INTENT( IN ) :: x

!             The vector to print.

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!     Local variable

      INTEGER ( KIND = ip_ ) :: j, i

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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n

!             The dimension of ix.

      INTEGER ( KIND = ip_ ), DIMENSION( n ), INTENT( IN ) :: ix

!             The vector to print.

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: j, i

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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz
      REAL( KIND = sp_ ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_row
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz
      REAL( KIND = sp_ ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_ptr
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk, i, i1, i2, i3

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nrow, ncol
      REAL( KIND = sp_ ), DIMENSION( : ), INTENT( IN ) :: A_val
      LOGICAL, INTENT( IN ) :: sym
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk, i, i1, i2, j, j1, j2, nval, lrow

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz
      REAL( KIND = dp_ ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_row
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz
      REAL( KIND = dp_ ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_ptr
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk, i, i1, i2, i3

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nrow, ncol
      REAL( KIND = dp_ ), DIMENSION( : ), INTENT( IN ) :: A_val
      LOGICAL, INTENT( IN ) :: sym
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk, i, i1, i2, j, j1, j2, nval, lrow

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

#ifdef REAL_128
!==============================================================================
!==============================================================================
      SUBROUTINE TOOLS_output_vector_real_qp( n, x, out )

!     Print x

!     Arguments:

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n

!             The dimension of x.

      REAL ( KIND = qp_ ), DIMENSION( n ), INTENT( IN ) :: x

!             The vector to print.

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!             The output device number

!     Programming: Ph. L. Toint, November 2002.

!     Local variable

      INTEGER ( KIND = ip_ ) :: j, i

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

      END SUBROUTINE TOOLS_output_vector_real_qp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_C_qp( nnz, A_val, A_row, A_col, out )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz
      REAL( KIND = qp_ ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_row
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk

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

      END SUBROUTINE TOOLS_output_matrix_real_C_qp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_S_qp( nnz, A_val, A_ptr, A_col, out )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz
      REAL( KIND = qp_ ), DIMENSION( nnz ), INTENT( IN ) :: A_val
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_ptr
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: A_col
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk, i, i1, i2, i3

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

      END SUBROUTINE TOOLS_output_matrix_real_S_qp

!==============================================================================
!==============================================================================

      SUBROUTINE TOOLS_output_matrix_real_D_qp( nrow, ncol, A_val, sym, out )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nrow, ncol
      REAL( KIND = qp_ ), DIMENSION( : ), INTENT( IN ) :: A_val
      LOGICAL, INTENT( IN ) :: sym
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out

!     Programming: Ph. L. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER ( KIND = ip_ ) :: k, kk, i, i1, i2, j, j1, j2, nval, lrow

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

      END SUBROUTINE TOOLS_output_matrix_real_D_qp

!==============================================================================
!==============================================================================

     FUNCTION TOOLS_expon_qp( r )
     CHARACTER ( LEN = 3 ) :: TOOLS_expon_qp

!  Obtain a 3 character representation of the exponent of the single precision
!  real r. exponents outside [-99,+99] will be represented as +in or -in as
!  relevant
!  Nick Gould, 2010

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = qp_ ) :: r

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i

     IF ( r == 0.0_qp_ ) THEN
       WRITE( TOOLS_expon_qp, "( '-in' )" )
     ELSE
       i = FLOOR( LOG10( r ) )
       IF ( i <= 99 .AND. i >= - 99 ) THEN
         WRITE( TOOLS_expon_qp, "( SP, I3 )" ) i
       ELSE IF ( i < - 99 ) THEN
         WRITE( TOOLS_expon_qp, "( '-in' )" )
       ELSE
         WRITE( TOOLS_expon_qp, "( '+in' )" )
       END IF
     END IF
     RETURN

!  End of TOOLS_expon_qp

     END FUNCTION TOOLS_expon_qp
#endif
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

