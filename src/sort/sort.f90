! THIS VERSION: GALAHAD 3.3 - 04/02/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*        SORT   M O D U L E         *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Philippe Toint

!  History -
!   originally released pre GALAHAD Version 1.0. July 1st 2001
!   update released with GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SORT_double

!
!               +---------------------------------------------+
!               |                                             |
!               |         Provides tools for sorting and      |
!               |         for applying and/or inverting       |
!               |              permutation vectors.           |
!               |                                             |
!               +---------------------------------------------+
!

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

!     Make everything private by default

      PRIVATE

!     Make the tools public

      PUBLIC :: SORT_inplace_invert

!            In place inversion of a given permutation.

      PUBLIC :: SORT_inplace_permute

!            Applies a given permutation to an integer vector and, optionally,
!            to a real vector, without resorting to extra storage.

      PUBLIC :: SORT_inverse_permute

!            Applies the inverse of a given permutation to an integer vector
!            and, optionally, to a real vector, without resorting to extra
!            storage.

      PUBLIC :: SORT_quicksort

!           Sorts a given integer/real vector in ascending order, optionally
!           applying the same permutation to an integer and/or to a real
!           vector(s).

      INTERFACE SORT_quicksort
         MODULE PROCEDURE SORT_quicksort_integer, SORT_quicksort_real
      END INTERFACE

      PUBLIC :: SORT_heapsort_build, SORT_heapsort_smallest

!           Partially sorts a given integer/real vector in ascending order,
!           optionally applying the same permutation to another integer vector.

      INTERFACE SORT_heapsort_build
         MODULE PROCEDURE SORT_heapsort_build_real, SORT_heapsort_build_integer
      END INTERFACE

      INTERFACE SORT_heapsort_smallest
         MODULE PROCEDURE SORT_heapsort_smallest_real,                         &
                          SORT_heapsort_smallest_integer
      END INTERFACE

      PUBLIC :: SORT_partition

!           Permute a given integer/real vector so that the values smaller
!           than a prescribed target value occur before those that exceed
!           that target, optionally applying the same permutation to another
!           integer vector.

      INTERFACE SORT_partition
         MODULE PROCEDURE SORT_partition_real, SORT_partition_integer
      END INTERFACE

      PUBLIC :: SORT_reorder_by_rows

      INTERFACE SORT_reorder_by_rows
         MODULE PROCEDURE SORT_reorder_by_rows, SORT_reorder_by_rows_no_vals
      END INTERFACE

!            Reorder a sparse matrix from arbitary coordinate order to row
!            order, that is so that the entries for row i appearing directly
!            before those for row i+1.

      PUBLIC :: SORT_reorder_by_cols

      INTERFACE SORT_reorder_by_cols
         MODULE PROCEDURE SORT_reorder_by_cols, SORT_reorder_by_cols_no_vals
      END INTERFACE

!            Reorder a sparse matrix from arbitary coordinate order to column
!            order, that is so that the entries for column j appearing directly
!            before those for column j+1.

!-------------------------------------------------------------------------------
!   P r e c i s i o n
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0E+0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PRIVATE, PARAMETER :: wp = dp

!-------------------------------------------------------------------------------
!   O t h e r s
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: OK            =   0
      INTEGER, PRIVATE, PARAMETER :: WRONG_N       =   1
      INTEGER, PRIVATE, PARAMETER :: SORT_TOO_LONG =   2

   CONTAINS

!==============================================================================
!==============================================================================

      SUBROUTINE SORT_inplace_invert( n, p )

!     Computes the inverse of the permutation p of the integers from 1 to n
!     in place and in linear time. The method exploits the fact that every
!     permutation is the product of disjoint cycles.

!     Arguments

      INTEGER,                 INTENT( IN )    :: n
      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: p

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: i, j, k, l

      DO i = 1, n
         l = p( i )
         IF ( l > 0 ) THEN
            j = i
            DO
               k = l
               l = p( k )
               IF ( l < 0 ) EXIT
               p( k ) = - j
               j = k
            END DO
         END IF
         p( i ) = - p( i )
      END DO

      RETURN

      END SUBROUTINE SORT_inplace_invert

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_inplace_permute( n, p, x, ix, iy )

!     Permute the entries of x so that x(i) appears in position p(i)
!     Do this without resorting to extra vector storage. Optionally,
!     permute the entries of ix so that ix(i) appears in position p(i).

!     Arguments

      INTEGER, INTENT( IN ) :: n

!            the dimension of the vectors p, x, and ix

      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: p

!            the permutation to apply to the content of x and ix

      INTEGER, DIMENSION( n ), OPTIONAL, INTENT( INOUT ) :: ix, iy

!            integer vector of size n, to which the permutation p
!            must be applied

      REAL ( KIND = wp ), DIMENSION( n ), OPTIONAL, INTENT( INOUT ) :: x

!            a real vector of size n, to which the permutation p
!            must be applied

!     Programming: N. Gould, November 1999, adapted by Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: i, pi, pi_old, ixpi, ixpi_old, iypi, iypi_old
      REAL ( KIND = wp ) :: xpi, xpi_old

      IF ( PRESENT( iy ) ) THEN

!        For x, ix and iy

         IF ( PRESENT( ix ) .AND. PRESENT( x ) ) THEN

            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in p

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i  moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again

               ELSE
                  xpi_old  = x( i )
                  ixpi_old = ix( i )
                  iypi_old = iy( i )
                  DO
                     xpi         = x( pi )
                     x( pi )     = xpi_old
                     xpi_old     = xpi
                     ixpi        = ix( pi )
                     ix( pi )    = ixpi_old
                     ixpi_old    = ixpi
                     iypi        = iy( pi )
                     iy( pi )    = iypi_old
                     iypi_old    = iypi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  x( i )  = xpi_old
                  ix( i ) = ixpi_old
                  iy( i ) = iypi_old
               END IF
            END DO

!     For x and iy only

         ELSE IF( PRESENT( x ) ) THEN

            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in p

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again.

               ELSE
                  xpi_old  = x( i )
                  iypi_old  = iy( i )
                  DO
                     xpi         = x( pi )
                     iypi        = iy( pi )
                     x( pi )     = xpi_old
                     iy( pi )    = iypi_old
                     xpi_old     = xpi
                     iypi_old    = iypi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  x( i ) = xpi_old
                  iy( i ) = iypi_old
               END IF
            END DO

!     For ix and iy only

         ELSE IF ( PRESENT( ix ) ) THEN
            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in P

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again.

               ELSE
                  ixpi_old  = ix( i )
                  iypi_old  = iy( i )
                  DO
                     ixpi        = ix( pi )
                     iypi        = iy( pi )
                     ix( pi )    = ixpi_old
                     iy( pi )    = iypi_old
                     ixpi_old    = ixpi
                     iypi_old    = iypi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  ix( i ) = ixpi_old
                  iy( i ) = ixpi_old
               END IF
            END DO

!     For iy only

         ELSE
            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in P

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again.

               ELSE
                  iypi_old  = iy( i )
                  DO
                     iypi        = iy( pi )
                     iy( pi )    = iypi_old
                     iypi_old    = iypi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  iy( i ) = iypi_old
               END IF
            END DO
         END IF

      ELSE

!        For x and ix

         IF ( PRESENT( ix ) .AND. PRESENT( x ) ) THEN

            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in p

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i  moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again

               ELSE
                  xpi_old  = x( i )
                  ixpi_old = ix( i )
                  DO
                     xpi         = x( pi )
                     x( pi )     = xpi_old
                     xpi_old     = xpi
                     ixpi        = ix( pi )
                     ix( pi )    = ixpi_old
                     ixpi_old    = ixpi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  x( i )  = xpi_old
                  ix( i ) = ixpi_old
               END IF
            END DO

!     For x only

         ELSE IF( PRESENT( x ) ) THEN

            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in p

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again.

               ELSE
                  xpi_old  = x( i )
                  DO
                     xpi         = x( pi )
                     x( pi )     = xpi_old
                     xpi_old     = xpi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  x( i ) = xpi_old
               END IF
            END DO
         ELSE IF ( PRESENT( ix ) ) THEN
            DO i = 1, n
               pi = p( i )

!              Skip any entry which is already in place

               IF ( pi == i ) CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in P

               IF ( pi < 0 ) THEN
                  p( i ) = - pi

!              The i-th entry is not in place. Chase through the list of entries
!              i, p( i ), p( p( i ) ), ... until
!              p( ... ( p( i ) ) ... ) = i moving entries into place.
!              Negate the relevant entries in p so that these entries will
!              not be moved again.

               ELSE
                  ixpi_old  = ix( i )
                  DO
                     ixpi        = ix( pi )
                     ix( pi )    = ixpi_old
                     ixpi_old    = ixpi
                     pi_old      = pi
                     pi          = p( pi_old )
                     p( pi_old ) = - pi
                     IF ( pi == i ) EXIT
                  END DO
                  ix( i ) = ixpi_old
               END IF
            END DO
         END IF
      END IF

      RETURN

      END SUBROUTINE SORT_inplace_permute

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_inverse_permute( n, p, x, ix )

!     Permute the entries of x so that x(p(i)) appears in position i
!     Do this without resorting to extra vector storage. Optionally,
!     permute the entries of ix so that ix(p(i)) appears in position i

!     Arguments

      INTEGER, INTENT( IN ) :: n

!            the dimension of the vectors p, x, and ix

      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: p

!            the permutation whose inverse must be applied to the content
!            of x and ix

      INTEGER, DIMENSION( n ), OPTIONAL, INTENT( INOUT ) :: ix

!            an integer vector of size n, to which the permutation inverse of
!            p must be applied

      REAL ( KIND = wp ), DIMENSION( n ), OPTIONAL, INTENT( INOUT ) :: x

!            a real vector of size n, to which the permutation inverse of p
!            must be applied

!     Programming: N. Gould, November 1999, adapted by Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: i, pi, pi_old, ixi
      REAL ( KIND = wp ) :: xi

!     For both x and ix:

      IF ( PRESENT( x ) .AND. PRESENT( ix ) ) THEN

!        Loop over the entries of x and ix

         DO i = 1, n
            pi = p( i )

!           Skip any entry which is already in place

            IF ( pi == i ) THEN
               CYCLE

!           Skip any entry which has already been moved into place, remembering
!           to un-negate the relevant entry in P_inverse

            ELSE IF ( pi < 0 ) THEN
               p( i ) = - pi

!           The i-th entry is not in place. Chase through the list of entries
!           i, p( i ), p( p( i ) ), ... until  p( ... ( p( i ) ) ... ) = i,
!           moving entries into place. Negate the relevant entries in p_inverse
!           so that these entries will not be moved again

            ELSE
               xi     = x( i )
               ixi    = ix( i )
               pi_old = i
               DO
                  x( pi_old )  = x( pi )
                  ix( pi_old ) = ix( pi )
                  pi_old       = pi
                  pi           = p( pi_old )
                  p( pi_old )  = - pi
                  IF ( pi == i ) EXIT
               END DO
               x( pi_old )  = xi
               ix( pi_old ) = ixi
            END IF
         END DO

!     For just x:

      ELSE IF ( PRESENT( x ) ) THEN

!        Loop over the entries of x

         DO i = 1, n
            pi = p( i )

!           Skip any entry which is already in place

            IF ( pi == i ) THEN
               CYCLE

!              Skip any entry which has already been moved into place,
!              remembering to un-negate the relevant entry in p_inverse

            ELSE IF ( pi < 0 ) THEN
               p( i ) = - pi

!           The i-th entry is not in place. Chase through the list of entries
!           i, p( i ), p( p_inverse( i ) ),... until p( ...( p( i ) ) ... ) = i,
!           moving entries into place.  Negate the relevant entries in p so
!           that these entries will  not be moved again

            ELSE
               xi     = x( i )
               pi_old = i
               DO
                  x( pi_old )  = x( pi )
                  pi_old       = pi
                  pi           = p( pi_old )
                  p( pi_old )  = - pi
                  IF ( pi == i ) EXIT
               END DO
               x( pi_old ) = xi
            END IF
         END DO

      ELSE IF ( PRESENT( ix ) ) THEN

!        Loop over the entries of ix

         DO i = 1, n
            pi = p( i )

!           Skip any entry which is already in place

            IF ( pi == i ) THEN
               CYCLE

!           Skip any entry which has already been moved into place, remembering
!           to un-negate the relevant entry in P_inverse

            ELSE IF ( pi < 0 ) THEN
               p( i ) = - pi

!           The i-th entry is not in place. Chase through the list of entries
!           i, p( i ), p( p_inverse( i ) ),... until p( ...( p( i ) ) ... ) = i,
!           moving entries into place. Negate the relevant entries in p so that
!           these entries will not be moved again

            ELSE
               ixi     = ix( i )
               pi_old = i
               DO
                  ix( pi_old ) = ix( pi )
                  pi_old       = pi
                  pi           = p( pi_old )
                  p( pi_old )  = - pi
                  IF ( pi == i ) EXIT
               END DO
               ix( pi_old ) = ixi
            END IF
         END DO

      END IF

      RETURN

      END SUBROUTINE SORT_inverse_permute

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_quicksort_integer( size, x, exitcode, ix, rx )

!     Sorts the content of integer vector x in ascending order, optionally
!     applying the same permutation to the integer vector ix and the
!     real vector rx.  This is an implementation of quicksort (Hoare's
!     partition-exchange sorting).

!     Arguments:

      INTEGER, INTENT( IN ) :: size

!              the size of x

      INTEGER, DIMENSION ( size ), INTENT( INOUT ) :: x

!              the integfer vector x for sorting (in ascending order)

      INTEGER, INTENT( OUT ) :: exitcode

!              the exitcode can take one of the two following values
!                 OK            ( 0 ) : successful sorting,
!                 WRONG_N       ( 1 ) : size is non-positive
!                 SORT_TOO_LONG ( 2 ) : the sorting request involves
!                                       more than 2** 32 numbers

      INTEGER, DIMENSION ( size ), OPTIONAL, INTENT( INOUT ) :: ix

!              an (optional) integer vector to be sorted according to x

      REAL ( KIND = wp ), &
               DIMENSION ( size ), OPTIONAL, INTENT( INOUT ) :: rx

!              an (optional) real vector to be sorted according to x

!     Programming: Ph. Toint, Spring 2001

!===============================================================================

!     Local variables

      INTEGER, PARAMETER :: log2s = 32  !  This parameter must be increased
                                        !  if more than 2**32 numbers are to
                                        !  to be sorted

      INTEGER            :: s, l, r, i, j, mid, stack( log2s, 2 ), p, itmp
      REAL ( KIND = wp ) :: rtmp

      IF ( size <= 0 ) THEN
         exitcode = WRONG_N
         RETURN
      END IF
      p     = 0
      itmp  = 0
      l     = size
      DO
         IF ( l <= 1 ) EXIT
         p    = p + 1
         itmp = itmp + MOD( l, 2 )
         l    = l / 2
      END DO
      IF ( itmp > 0 ) p = p + 1
      IF ( p <= log2s ) THEN
         s = 1
         stack( s, 1 ) = 1
         stack( s, 2 ) = size

!        Case where x, ix and rx must be sorted

         IF ( PRESENT( ix ) .AND. PRESENT( rx ) ) THEN
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        itmp    = x( i )
                        x( i )  = x( j )
                        x( j )  = itmp
                        itmp    = ix( i )
                        ix( i ) = ix( j )
                        ix( j ) = itmp
                        rtmp    = rx( i )
                        rx( i ) = rx( j )
                        rx( j ) = rtmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO

!        Case where x and ix must be sorted

         ELSE IF ( PRESENT ( ix ) ) THEN
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        itmp    = x( i )
                        x( i )  = x( j )
                        x( j )  = itmp
                        itmp    = ix( i )
                        ix( i ) = ix( j )
                        ix( j ) = itmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO

!        Case where x and rx must be sorted

         ELSE IF ( PRESENT ( rx ) ) THEN
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        itmp    = x( i )
                        x( i )  = x( j )
                        x( j )  = itmp
                        rtmp    = rx( i )
                        rx( i ) = rx( j )
                        rx( j ) = rtmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO

!        Case where only the x must be sorted

         ELSE
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        itmp   = x( i )
                        x( i ) = x( j )
                        x( j ) = itmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO
         END IF
         exitcode = OK
      ELSE
         exitcode = SORT_TOO_LONG
      ENDIF

      RETURN

      END SUBROUTINE SORT_quicksort_integer

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_quicksort_real( size, x, exitcode, ix, rx )

!     Sorts the content of real vector x in ascending order, optionally
!     applying the same permutation to the integer vector ix and the
!     real vector rx.  This is an implementation of quicksort (Hoare's
!     partition-exchange sorting).

!     Arguments:

      INTEGER, INTENT( IN ) :: size

!              the size of x

      REAL ( KIND = wp ), DIMENSION ( size ), INTENT( INOUT ) :: x

!              the vector x for sorting (in ascending order)

      INTEGER, INTENT( OUT ) :: exitcode

!              the exitcode can take one of the two following values
!                 OK            ( 0 ) : successful sorting,
!                 WRONG_N       ( 1 ) : size is non-positive
!                 SORT_TOO_LONG ( 2 ) : the sorting request involves
!                                       more than 2** 32 numbers

      INTEGER, DIMENSION ( size ), OPTIONAL, INTENT( INOUT ) :: ix

!              an (optional) integer vector to be sorted according to x

      REAL ( KIND = wp ), &
               DIMENSION ( size ), OPTIONAL, INTENT( INOUT ) :: rx

!              an (optional) real vector to be sorted according to x

!     Programming: Ph. Toint, Spring 2001

!===============================================================================

!     Local variables

      INTEGER, PARAMETER :: log2s = 32  !  This parameter must be increased
                                        !  if more than 2**32 numbers are to
                                        !  to be sorted

      INTEGER            :: s, l, r, i, j, stack( log2s, 2 ), p, itmp
      REAL ( KIND = wp ) :: rtmp, mid

      IF ( size <= 0 ) THEN
         exitcode = WRONG_N
         RETURN
      END IF
      p     = 0
      itmp  = 0
      l     = size
      DO
         IF ( l <= 1 ) EXIT
         p    = p + 1
         itmp = itmp + MOD( l, 2 )
         l    = l / 2
      END DO
      IF ( itmp > 0 ) p = p + 1
      IF ( p <= log2s ) THEN
         s = 1
         stack( s, 1 ) = 1
         stack( s, 2 ) = size

!        Case where x, ix and rx must be sorted

         IF ( PRESENT( ix ) .AND. PRESENT( rx ) ) THEN
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        rtmp    = x( i )
                        x( i )  = x( j )
                        x( j )  = rtmp
                        itmp    = ix( i )
                        ix( i ) = ix( j )
                        ix( j ) = itmp
                        rtmp    = rx( i )
                        rx( i ) = rx( j )
                        rx( j ) = rtmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO

!        Case where x and ix must be sorted

         ELSE IF ( PRESENT ( ix ) ) THEN
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        rtmp    = x( i )
                        x( i )  = x( j )
                        x( j )  = rtmp
                        itmp    = ix( i )
                        ix( i ) = ix( j )
                        ix( j ) = itmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO

!        Case where x and rx must be sorted

         ELSE IF ( PRESENT ( rx ) ) THEN
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        rtmp    = x( i )
                        x( i )  = x( j )
                        x( j )  = rtmp
                        rtmp    = rx( i )
                        rx( i ) = rx( j )
                        rx( j ) = rtmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO

!        Case where only the x must be sorted

         ELSE
            DO
               l = stack( s, 1 )
               r = stack( s, 2 )
               s = s - 1
               DO
                  i = l
                  j = r
                  mid = x( ( l + r ) / 2 )
                  DO
                     DO
                        IF ( x( i ) >= mid ) EXIT
                        i = i + 1
                     END DO
                     DO
                        IF ( x( j ) <= mid ) EXIT
                        j = j - 1
                     END DO
                     IF ( i <= j ) THEN
                        rtmp      = x( i )
                        x( i )    = x( j )
                        x( j )    = rtmp
                        i = i + 1
                        j = j - 1
                     ENDIF
                     IF ( i > j ) EXIT
                  END DO
                  IF ( j - l  <  r - i  ) THEN
                     IF ( i < r ) THEN
                        s = s + 1
                        stack( s, 1 ) = i
                        stack( s, 2 ) = r
                     ENDIF
                     r = j
                  ELSE
                     IF ( l < j ) THEN
                        s = s + 1
                        stack( s, 1 ) = l
                        stack( s, 2 ) = j
                     ENDIF
                     l = i
                  ENDIF
                  IF ( l >= r ) EXIT
               END DO
               IF ( s <= 0 ) EXIT
            END DO
         END IF
         exitcode = OK
      ELSE
         exitcode = SORT_TOO_LONG
      ENDIF

      RETURN

      END SUBROUTINE SORT_quicksort_real

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_heapsort_build_real( n, x, inform, ix, largest )

!  Given an array x, elements x(1), ...., x(n), subroutine SORT_heapsort_build
!  rearranges the elements to form a heap in which each parent has a smaller
!  (optionally larger) value than either of its children.

!  Algorithm 232 of CACM (J. W. J. Williams): a combination of
!  ALGOL procedures SETHEAP and INHEAP

!  Programming: Nick Gould, January 26th 1995.

!  ------------------------- dummy arguments --------------------------
!
!  n      integer, which gives the number of values to be sorted.
!         n must be positive
!
!  x      real array of length n. On input, x must contain the
!         values which are to be sorted. On output, these values
!         will have been permuted so as to form a heap
!
!  inform integer, which informs the user of the success of SORT_heapsort_build.
!         If inform = 0 on exit, the heap has been formed.
!         If inform = 1 on exit, n was input with a value less than
!                       or equal to 0 and the heap has not been formed.
!
!  ix     is an OPTIONAL integer array of length n. On input, ix may
!         be used to hold indexing information (such as the original
!         order of the values) about x. On output, ix will have been
!         permuted so that ix(k) still refers to x(k).
!
!  largest is an OPTIONAL logical argument that if present and .TRUE. specifies
!         that the heap will instead be arranged so that each parent has a
!         larger (rather than smaller) value than either of its children

!  ------------------ end of dummy arguments --------------------------

      INTEGER, INTENT( IN ) :: n
      INTEGER, INTENT( OUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: ix
      LOGICAL, INTENT( IN ), OPTIONAL :: largest

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, k, ix_in
      REAL ( KIND = wp )  :: x_in
      LOGICAL :: index, find_largest

!  check for and use optional arguments

      index = PRESENT( ix )
      IF ( n <= 0 ) THEN
        inform = WRONG_N
        RETURN
      ENDIF

      find_largest = PRESENT( largest )
      IF ( find_largest ) find_largest = largest

!  Add the elements to the heap one at a time

!  * form the heap in deceasing order with the largest entry at the root

      IF ( find_largest ) THEN

!  the index array is present

        IF ( index ) THEN
          DO k = 2, n
            x_in = x( k )
            ix_in = ix( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j )
              ix( i ) = ix( j )
              i = j
            END DO
            x( i ) = x_in
            ix( i ) = ix_in
          END DO

!  the index array is absent

        ELSE
          DO k = 2, n
            x_in = x( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO
            x( i ) = x_in
          END DO
        END IF

!  * form the heap in increasing order with the smallest entry at the root

      ELSE

!  the index array is present

        IF ( index ) THEN
          DO k = 2, n
            x_in = x( k )
            ix_in = ix( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j )
              ix( i ) = ix( j )
              i = j
            END DO
            x( i ) = x_in
            ix( i ) = ix_in
          END DO

!  the index array is absent

        ELSE
          DO k = 2, n
            x_in = x( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO
            x( i ) = x_in
          END DO
        END IF

      END IF
      inform = OK

      RETURN

!  End of subroutine SORT_heapsort_build_real

      END SUBROUTINE SORT_heapsort_build_real

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_heapsort_build_integer( n, x, inform, ix, largest )

!  Given an array x, elements x(1), ...., x(n), subroutine SORT_heapsort_build
!  re-arranges the elements to form a heap in which each parent has a smaller
!  (optionally larger) value than either of its children.

!  Algorithm 232 of CACM (J. W. J. Williams): a combination of
!  ALGOL procedures SETHEAP and INHEAP

!  Programming: Nick Gould, January 26th 1995.

!  ------------------------- dummy arguments --------------------------
!
!  n      integer, which gives the number of values to be sorted.
!         n must be positive
!
!  x      integer array of length n. On input, x must contain the
!         values which are to be sorted. On output, these values
!         will have been permuted so as to form a heap
!
!  inform integer, which informs the user of the success of SORT_heapsort_build.
!         If inform = 0 on exit, the heap has been formed.
!         If inform = 1 on exit, n was input with a value less than
!                       or equal to 0 and the heap has not been formed.
!
!  ix     is an OPTIONAL integer array of length n. On input, ix may
!         be used to hold indexing information (such as the original
!         order of the values) about x. On output, ix will have been
!         permuted so that ix(k) still refers to x(k).
!
!  largest is an OPTIONAL logical argument that if present and .TRUE. specifies
!         that the heap will instead be arranged so that each parent has a
!         larger (rather than smaller) value than either of its children

!  ------------------ end of dummy arguments --------------------------

      INTEGER, INTENT( IN ) :: n
      INTEGER, INTENT( OUT ) :: inform
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: x
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( n ) :: ix
      LOGICAL, INTENT( IN ), OPTIONAL :: largest

!===============================================================================

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, k, ix_in
      INTEGER :: x_in
      LOGICAL :: index, find_largest

!  check for and use optional arguments

      index = PRESENT( ix )
      IF ( n <= 0 ) THEN
        inform = WRONG_N
        RETURN
      ENDIF

      find_largest = PRESENT( largest )
      IF ( find_largest ) find_largest = largest

!  Add the elements to the heap one at a time

!  * form the heap in deceasing order with the largest entry at the root

      IF ( find_largest ) THEN

!  the index array is present

        IF ( index ) THEN
          DO k = 2, n
            x_in = x( k )
            ix_in = ix( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j )
              ix( i ) = ix( j )
              i = j
            END DO
            x( i ) = x_in
            ix( i ) = ix_in
          END DO

!  the index array is absent

        ELSE
          DO k = 2, n
            x_in = x( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO
            x( i ) = x_in
          END DO
        END IF

!  * form the heap in increasing order with the smallest entry at the root

      ELSE

!  the index array is present

        IF ( index ) THEN
          DO k = 2, n
            x_in = x( k )
            ix_in = ix( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j )
              ix( i ) = ix( j )
              i = j
            END DO
            x( i ) = x_in
            ix( i ) = ix_in
          END DO

!  the index array is absent

        ELSE
          DO k = 2, n
            x_in = x( k )

!  The cycle may be repeated log2(k) times, but on average is repeated
!  merely twice

            i = k
            DO
              IF ( i <= 1 ) EXIT
              j = i / 2
              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO
            x( i ) = x_in
          END DO
        END IF

      END IF
      inform = OK

      RETURN

!  End of subroutine SORT_heapsort_build_integer

      END SUBROUTINE SORT_heapsort_build_integer

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_heapsort_smallest_real( m, x, inform, ix, largest )

!  Given an array x, elements x(1), ...., x(m) forming a heap,
!  SORT_heapsort_smallest assigns to x_out the value of x(1), the smallest
!  (optionally largest) member of the heap, and arranges the remaining
!  members as elements 1 to m - 1 of x. x_out is then placed in x(m)

!  Algorithm 232 of CACM (J. W. J. Williams): a combination of
!  ALGOL procedures OUTHEAP and SWOPHEAP

!  Programming: Nick Gould, January 26th 1995.

!  ------------------------- dummy arguments --------------------------
!
!  m      integer, which gives the number of values to be sorted.
!         m must be positive
!
!  x      real array of length m. On input, x must contain the values which
!         are to be sorted stored in a heap. On output, the smallest value
!         will have been moved into x(m) and the remaining values x(k),
!         k = 1,..., m-1 will have been restored to a heap
!
!  inform integer, which informs the user of the success of
!         SORT_heapsort_smallest.
!         If inform = 0 on exit, the smallest value has been found.
!         If inform = 1 on exit, m was input with a value less than
!                       or equal to 0 and the heap has not been formed
!
!  ix   optional integer array of length m. On input, ix may be used
!         to hold indexing information (see SORT_heapsort_build) about x.
!         On output, ix will have been permuted so that ix(k) still
!         refers to x(k). This argument is only permitted if it was present
!         when calling SORT_heapsort_build
!
!  largest optional logical argument that if present and .TRUE. specifies
!         that the heap will instead be arranged so that each parent has a
!         larger (rather than smaller) value than either of its children
!         This argument is only permitted if it was present when calling
!         SORT_heapsort_build
!
!  ------------------ end of dummy arguments --------------------------

      INTEGER, INTENT( IN ) :: m
      INTEGER, INTENT( OUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: x
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( m ) :: ix
      LOGICAL, INTENT( IN ), OPTIONAL :: largest

!===============================================================================

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, ix_in, ix_out
      REAL ( KIND = wp ) :: x_in, x_out
      LOGICAL :: index, find_largest

!  check for and use optional arguments

      index = PRESENT( ix )
      IF ( m <= 0 ) THEN
        inform = WRONG_N
        RETURN
      ENDIF

      find_largest = PRESENT( largest )
      IF ( find_largest ) find_largest = largest

!  * find the largest entry and rebuild the remaining heap

      IF ( find_largest ) THEN

!  Add the element x_in to the heap, extract and assign to x_out
!  the value of the largest member of the resulting set, and
!  leave the remaining elements in a heap of the original size.
!  In this process, elements 1 to m+1 of the array x may be disturbed

        IF ( m > 1 ) THEN
          i = 1
          IF ( index ) THEN
            x_out = x( 1 ) ; ix_out = ix( 1 )
            x_in = x( m ) ; ix_in = ix( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is largest, the heap has been
!  restored. If one of the children is largest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is largest

              IF ( x( j + 1 ) > x( j ) ) j = j + 1

!  Determine if the larger daughter is greater than the value from node i

              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j ) ; ix( i ) = ix( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in ; ix( i ) = ix_in

!  Store the largest value in the now vacated m-th position of the list

            x( m ) = x_out ; ix( m ) = ix_out
          ELSE
            x_out = x( 1 )
            x_in = x( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is largest, the heap has been
!  restored. If one of the children is largest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is largest

              IF ( x( j + 1 ) > x( j ) ) j = j + 1

!  Determine if the larger daughter is greater than the value from node i

              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in

!  Store the largest value in the now vacated m-th position of the list

            x( m ) = x_out
          END IF
        END IF

!  * find the smallest entry and rebuild the remaining heap

      ELSE

!  Add the element x_in to the heap, extract and assign to x_out
!  the value of the smallest member of the resulting set, and
!  leave the remaining elements in a heap of the original size.
!  In this process, elements 1 to m+1 of the array x may be disturbed

        IF ( m > 1 ) THEN
          i = 1
          IF ( index ) THEN
            x_out = x( 1 ) ; ix_out = ix( 1 )
            x_in = x( m ) ; ix_in = ix( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is smallest, the heap has been
!  restored. If one of the children is smallest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is smallest

              IF ( x( j + 1 ) < x( j ) ) j = j + 1

!  Determine if the smaller daughter is less than the value from node i

              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j ) ; ix( i ) = ix( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in ; ix( i ) = ix_in

!  Store the smallest value in the now vacated m-th position of the list

            x( m ) = x_out ; ix( m ) = ix_out
          ELSE
            x_out = x( 1 )
            x_in = x( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is smallest, the heap has been
!  restored. If one of the children is smallest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is smallest

              IF ( x( j + 1 ) < x( j ) ) j = j + 1

!  Determine if the smaller daughter is less than the value from node i

              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in

!  Store the smallest value in the now vacated m-th position of the list

            x( m ) = x_out
          END IF

        END IF

      END IF
      inform = OK

      RETURN

!  End of subroutine SORT_heapsort_smallest_real

      END SUBROUTINE SORT_heapsort_smallest_real

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_heapsort_smallest_integer( m, x, inform, ix, largest )

!  Given an array x, elements x(1), ...., x(m) forming a heap,
!  SORT_heapsort_smallest assigns to x_out the value of x(1), the smallest
!  (optionally largest) member of the heap, and arranges the remaining
!  members as elements 1 to m - 1 of x. x_out is then placed in x(m)

!  Algorithm 232 of CACM (J. W. J. Williams): a combination of
!  ALGOL procedures OUTHEAP and SWOPHEAP

!  Programming: Nick Gould, January 26th 1995.

!  ------------------------- dummy arguments --------------------------
!
!  m      integer, which gives the number of values to be sorted.
!         m must be positive
!
!  x      integer array of length m. On input, x must contain the values
!         that are to be sorted stored in a heap. On output, the smallest
!         value will have been moved into x(m) and the remaining values
!         x(k), k = 1,..., m-1 will have been restored to a heap
!
!  inform integer, which informs the user of the success of
!         SORT_heapsort_smallest.
!         If inform = 0 on exit, the smallest value has been found.
!         If inform = 1 on exit, m was input with a value less than
!                       or equal to 0 and the heap has not been formed
!
!  ix     optional integer array of length m. On input, ix may be used
!         to hold indexing information (see SORT_heapsort_build) about x.
!         On output, ix will have been permuted so that ix(k) still
!         refers to x(k). This argument is only permitted if it was present
!         when calling SORT_heapsort_build
!
!  largest optional logical argument that if present and .TRUE. specifies
!         that the heap will instead be arranged so that each parent has a
!         larger (rather than smaller) value than either of its children
!         This argument is only permitted if it was present when calling
!         SORT_heapsort_build
!
!  ------------------ end of dummy arguments --------------------------

      INTEGER, INTENT( IN ) :: m
      INTEGER, INTENT( OUT ) :: inform
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: x
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( m ) :: ix
      LOGICAL, INTENT( IN ), OPTIONAL :: largest

!===============================================================================

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, ix_in, ix_out
      INTEGER :: x_in, x_out
      LOGICAL :: index, find_largest

!  check for and use optional arguments

      index = PRESENT( ix )
      IF ( m <= 0 ) THEN
        inform = WRONG_N
        RETURN
      ENDIF

      find_largest = PRESENT( largest )
      IF ( find_largest ) find_largest = largest

!  * find the largest entry and rebuild the remaining heap

      IF ( find_largest ) THEN

!  Add the element x_in to the heap, extract and assign to x_out
!  the value of the largest member of the resulting set, and
!  leave the remaining elements in a heap of the original size.
!  In this process, elements 1 to m+1 of the array x may be disturbed

        IF ( m > 1 ) THEN
          i = 1
          IF ( index ) THEN
            x_out = x( 1 ) ; ix_out = ix( 1 )
            x_in = x( m ) ; ix_in = ix( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is largest, the heap has been
!  restored. If one of the children is largest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is largest

              IF ( x( j + 1 ) > x( j ) ) j = j + 1

!  Determine if the larger daughter is greater than the value from node i

              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j ) ; ix( i ) = ix( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in ; ix( i ) = ix_in

!  Store the largest value in the now vacated m-th position of the list

            x( m ) = x_out ; ix( m ) = ix_out
          ELSE
            x_out = x( 1 )
            x_in = x( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is largest, the heap has been
!  restored. If one of the children is largest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is largest

              IF ( x( j + 1 ) > x( j ) ) j = j + 1

!  Determine if the larger daughter is greater than the value from node i

              IF ( x( j ) <= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in

!  Store the largest value in the now vacated m-th position of the list

            x( m ) = x_out
          END IF
        END IF

!  * find the smallest entry and rebuild the remaining heap

      ELSE

!  Add the element x_in to the heap, extract and assign to x_out
!  the value of the smallest member of the resulting set, and
!  leave the remaining elements in a heap of the original size.
!  In this process, elements 1 to m+1 of the array x may be disturbed

        IF ( m > 1 ) THEN
          i = 1
          IF ( index ) THEN
            x_out = x( 1 ) ; ix_out = ix( 1 )
            x_in = x( m ) ; ix_in = ix( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is smallest, the heap has been
!  restored. If one of the children is smallest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is smallest

              IF ( x( j + 1 ) < x( j ) ) j = j + 1

!  Determine if the smaller daughter is less than the value from node i

              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j ) ; ix( i ) = ix( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in ; ix( i ) = ix_in

!  Store the smallest value in the now vacated m-th position of the list

            x( m ) = x_out ; ix( m ) = ix_out
          ELSE
            x_out = x( 1 )
            x_in = x( m )

!  Move from the top of the heap comparing the value of node i
!  with its two daughters. If node i is smallest, the heap has been
!  restored. If one of the children is smallest, promote this child
!  in the heap and move to the now vacated node.
!  This cycle may be repeated log2(m) times

            DO
              j = i + i
              IF ( j > m - 1 ) EXIT

!  Determine which of the two daughters is smallest

              IF ( x( j + 1 ) < x( j ) ) j = j + 1

!  Determine if the smaller daughter is less than the value from node i

              IF ( x( j ) >= x_in ) EXIT
              x( i ) = x( j )
              i = j
            END DO

!  The heap has been restored

            x( i ) = x_in

!  Store the smallest value in the now vacated m-th position of the list

            x( m ) = x_out
          END IF
        END IF

      END IF
      inform = OK

      RETURN

!  End of subroutine SORT_heapsort_smallest_integer

      END SUBROUTINE SORT_heapsort_smallest_integer

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_partition_real( n, X, x_division, less, IX )

!  given a real array x of length n and an associated integer array ix,
!  reorder the arrays so that the first less components in the new ordering
!  of x < x_division, and the remainder are >= x_division

!  dummy arguments

      INTEGER, INTENT( IN ):: n
      INTEGER, INTENT( OUT ):: less
      REAL( kind = wp ), INTENT( IN ) :: x_division
      REAL ( KIND = WP ), INTENT( INOUT ), DIMENSION( n ) :: X
      INTEGER, OPTIONAL, INTENT( INOUT ), DIMENSION( n ) :: IX

!  local variables

      INTEGER :: more, ix_less
      REAL ( KIND = WP ) :: x_less, x_more

      less = 0 ; more = n + 1
!     write(6,*) ' n ', n
!     write(6,*) ' X', X
!      write(6,*) ' x_division ', x_division

!  march forward through x until a component violates x <= x_division

      DO
        x_less = X( less + 1 )
!       write(6,"( 'less ', I6, 2ES12.4 )" ) less + 1, x_less, x_division
        IF ( x_less < x_division ) THEN
          less = less + 1
          IF ( less >= more - 1 ) EXIT
          CYCLE
        END IF

!  march backwards through x until a component violates x > x_division

        DO
          IF ( less + 1 == more - 1 ) RETURN
          x_more = X( more - 1 )
!         write(6,"( 'more ', I6, 2ES12.4 )" ) more - 1, x_more, x_division
          IF ( x_more < x_division ) EXIT
          more = more - 1
        END DO

!  swap the two violated x, and their attendant ix, components

        less = less + 1 ; more = more - 1
!       write(6,*) ' swapping ', less, more
        X( less ) = x_more ; X( more ) = x_less
        IF ( PRESENT( ix ) ) THEN
          ix_less = IX( less ) ; IX( less ) = IX( more ) ; IX( more ) = ix_less
        END IF
        IF ( less + 1 == more ) EXIT
      END DO

      RETURN

!  end of subroutine SORT_partition_real

      END SUBROUTINE SORT_partition_real

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_partition_integer( n, X, x_division, less, IX )

!  given an integer array x of length n and an associated integer array ix,
!  reorder the arrays so that the first less components in the new ordering
!  of x <= x_division, and the remainder are > x_division

!  dummy arguments

      INTEGER, INTENT( IN ):: n
      INTEGER, INTENT( OUT ):: less
      INTEGER, INTENT( IN ) :: x_division
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: X
      INTEGER, OPTIONAL, INTENT( INOUT ), DIMENSION( n ) :: IX

!  local variables

      INTEGER :: more, ix_less
      INTEGER :: x_less, x_more

      less = 0 ; more = n + 1

!  march forward through x until a component violates x <= x_division

      DO
        x_less = X( less + 1 )
!       write(6,"( 'less ', I6, 2ES12.4 )" ) less + 1, x_less, x_division
        IF ( x_less < x_division ) THEN
          less = less + 1
          IF ( less >= more - 1 ) EXIT
          CYCLE
        END IF

!  march backwards through x until a component violates x > x_division

        DO
          IF ( less + 1 == more - 1 ) RETURN
          x_more = X( more - 1 )
!         write(6,"( 'more ', I6, 2ES12.4 )" ) more - 1, x_more, x_division
          IF ( x_more < x_division ) EXIT
          more = more - 1
        END DO

!  swap the two violated x, and their attendant ix, components

        less = less + 1 ; more = more - 1
!       write(6,*) ' swapping ', less, more
        X( less ) = x_more ; X( more ) = x_less
        IF ( PRESENT( ix ) ) THEN
          ix_less = IX( less ) ; IX( less ) = IX( more ) ; IX( more ) = ix_less
        END IF
        IF ( less + 1 == more ) EXIT
      END DO

      RETURN

!  end of subroutine SORT_partition_integer

      END SUBROUTINE SORT_partition_integer

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_reorder_by_rows( nr, nc, nnz, A_row, A_col, la, A_val,   &
                                       A_ptr, lptr, IW, liw, error, warning,   &
                                       inform )

!  Reorder a sparse matrix A from arbitary coordinate order to row order.

!  Programming: Nick Gould, 8th August 2002.

!  ------------------------- dummy arguments --------------------------
!
!  nr     integer, which gives the number of rows in A.
!         nr must be non-negative.
!
!  nc     integer, which gives the number of columns in A.
!         nc must be non-negative.
!
!  nnz    integer, which gives the number of nonzeros in A.
!         nnz must be non-negative.
!
!  A_row  integer array of length la. On entry, A_row(k), k = 1, ..., nnz give
!         the row indices of A. On exit, A_row will have been reordered, but
!         A_row(k) will still be the row index corresponding to the
!         entry with column index A_col(k).
!
!  A_col  integer array of length la. On entry, A_col(k), k = 1, ..., nnz give
!         the column indices of A. On exit, A_col will have been reordered so
!         that entries in row i appear directly before those in row i+1 for
!         i = 1, ..., nr-1.
!
!  la    integer, which gives the actual dimension of A_val.
!        la must be at least nnz.
!
!  A_val  real array of length la. On entry, A_val(k), k = 1, ..., nnz give
!         the values of A. On exit, A_val will have been reordered so that
!         entries in row i appear directly before those in row i+1 for
!         i = 1, ..., nr-1 and correspond to those in A_row and A_col.
!
!  A_ptr  integer array of length lptr. On exit, A_ptr(i), i = 1, ..., nr give
!         the starting addresses for the entries in A_row/A_col/A_val in
!         row i, while A_ptr(nr+1) gives the index of the first non-occupied
!         component of A.
!
!  lptr  integer, which gives the actual dimension of A_ptr.
!        lptr must be at least nr + 1.
!
!  IW     workspace integer array of length liw.
!
!  liw   integer, which gives the actual dimension of IW
!        liw  must be at least MAX( nr, nc ) + 1.
!
!  error integer, which gives the output unit number for error messages.
!        Error messages only occur if error > 0.
!
!  warning integer, which gives the output unit number for warning messages.
!        Warning messages only occur if warning > 0.
!
!  inform integer, which gives the exit status of SORT_reorder_by_rows.
!         Possible values are:
!
!          0   A has been successfully re-orderered.
!         -1   A has been successfully re-orderered, but there were duplicate
!              entries which have been summed.
!         -2   A has been successfully re-orderered, but there were row entries
!              out of range which were ignored.
!         -3   A has been successfully re-orderered, but there were column
!              entries out of range which were ignored.
!         -4   A has been successfully re-orderered, but there were no rows
!         -5   A has been successfully re-orderered, but there were no entries
!          1   nr, nc or nnz is too small. The reordering was unsuccessful.
!          2   la < nnz. The reordering was unsuccessful.
!          3   liw < MAX( nr, nc ) + 1. The reordering was unsuccessful.
!          4   lptr < nr + 1. The reordering was unsuccessful.
!          5   All entries were out of order. The reordering was unsuccessful.
!
!  ------------------ end of dummy arguments --------------------------
!
!  Dummy arguments

      INTEGER, INTENT( IN ) :: nr, nc, nnz, la, lptr, liw, error, warning
      INTEGER, INTENT( OUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( la ) :: A_val
      INTEGER, INTENT( INOUT ), DIMENSION( la ) :: A_row, A_col
      INTEGER, INTENT( OUT ), DIMENSION( lptr ) :: A_ptr
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW

!  Local variables

      INTEGER :: i, j, k, k1, k2, l, nzi, ie, iep, je, jep
      INTEGER :: loc, idup, iout, jout, nzout
      REAL ( KIND = wp ) :: ae, aep

! Initialize data

      inform = 0
      nzout = 0 ; iout = 0 ; jout = 0 ; idup = 0

!  Check for faulty input data

      IF ( nr < 0 .OR. nc < 0 .OR. nnz < 0 ) THEN
        inform = 1
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' nr, nc, or nnz is out of range', /,           &
         &           ' nr = ',I0,' nc = ',I0,' nnz = ', I0 )" ) nr, nc, nnz
        END IF
        RETURN
      END IF

      IF ( la < nnz ) THEN
        inform = 2
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' increase la from', I0, ' to at least ', I0 )")&
            la, nnz
        END IF
        RETURN
      END IF

      IF ( liw < MAX( nr, nc ) + 1 ) THEN
        inform = 3
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'liw ', liw, MAX( nr, nc ) + 1
        END IF
        RETURN
      END IF

      IF ( lptr < nr + 1 ) THEN
        inform = 4
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'lptr', lptr, nr + 1
        END IF
        RETURN
      END IF

!  If the matrix has no rows or no entries, exit accordingly

      IF ( nr == 0 ) THEN
        inform = - 4
        A_ptr( 1 ) = 0
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no rows' )" )
        END IF
        RETURN
      END IF

      IF ( nnz == 0 ) THEN
        inform = - 5
        A_ptr( 1 : nr + 1 ) = 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no entries' )" )
        END IF
        RETURN
      END IF

!  Record the number of column and row indices out of order in iout and jout
!  and then remove them from consideration

      DO k = 1,nnz
        i = A_row( k ); j = A_col( k )
        IF ( i > nr .OR. i < 1 ) THEN
          iout = iout + 1
        ELSE IF ( j > nc .OR. j < 1 ) THEN
          jout = jout + 1
        ELSE
          nzout = nzout + 1
          A_row( nzout ) = i ; A_col( nzout ) = j
          A_val( nzout ) = A_val( k )
        END IF
      END DO

!  Inform the user if there has been faulty data

      IF ( iout > 0 ) THEN
        inform = - 2
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_row were out of',     &
         &              ' range and have been ignored by the routine')" ) iout
        END IF
      END IF

      IF ( jout > 0 ) THEN
        inform = - 3
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_col were out of',     &
         &              ' range and have been ignored by the routine')" ) jout
        END IF
      END IF

!  If all the data is faulty, exit

      IF ( iout + jout == nnz ) THEN
        inform = 5
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 4X, ' All entries input in A were out of range ' )" )
        END IF
        nzout = 0
        RETURN
      END IF

!  nzout gives the number of nonzero entries following removals. Now sort the
!  pattern of a sparse matrix from arbitary order to row order. The
!  order within each row is unimportant

!  Record the number of elements in each row in IW

      IW( : nr + 1 ) = 0
      DO k = 1, nzout
        i = A_row( k )
        IW( i ) = IW( i ) + 1
      END DO

!  Record the positions where each row would begin, in a compressed format
!  with the rows in natural order, in A_ptr and IW

      A_ptr( 1 ) = 1
      DO i = 2, nr + 1
        A_ptr( i ) = IW( i - 1 ) + A_ptr( i - 1 )
        IW( i - 1 ) = A_ptr( i - 1 )
      END DO

!  Reorder the elements into row order. Fill in each row from the front,
!  and increase the pointer IW( k ) by 1 as a new entry is placed in row k

      DO l = 1, nr
        DO k = IW( l ), A_ptr( l + 1 ) - 1
          ie = A_row( k ) ; je = A_col( k ) ; ae = A_val( k )
          DO j = 1, nzout
            IF ( ie == l ) EXIT
            loc = IW( ie )
            iep = A_row( loc ) ; jep = A_col( loc ) ; aep = A_val( loc )
            IW( ie ) = loc + 1
            A_row( loc ) = ie ; A_col( loc ) = je ; A_val( loc ) = ae
            ie = iep ; je = jep ; ae = aep
          END DO
          A_row( k ) = ie ; A_col( k ) = je ; A_val( k ) = ae
        END DO
      END DO

!  Check for duplicates

      nzout = 0
      k1 = 1
      nzi = 0
      IW( : nc ) = 0
      DO i = 1, nr
        k2 = A_ptr( i + 1 ) - 1
        A_ptr( i + 1 ) = A_ptr( i )
        DO k = k1, k2
          j = A_col( k )
          IF ( IW( j ) <= nzi ) THEN
            nzout = nzout + 1
            A_row( nzout ) = A_row( k )
            A_col( nzout ) = j ; A_val( nzout ) = A_val( k )
            A_ptr( i + 1 ) = A_ptr( i + 1 ) + 1
            IW( j ) = nzout

!  There is a duplicate in row i; sum the values

          ELSE
            idup = idup + 1
            A_val( IW( j ) ) = A_val( IW( j ) ) + A_val( k )
          END IF
        END DO
        k1 = k2 + 1
        nzi = nzout
      END DO
      IF ( idup > 0 ) THEN
        inform = - 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 3X, I0,' duplicate input entries summed' )" ) idup
        END IF
      END IF

!  Non-executable statements

 2000 FORMAT( /,' - Error return from SORT_reorder_by_rows - inform = ', I0 )
 2010 FORMAT( /,' - Warning message from SORT_reorder_by_rows - inform = ', I0 )
 2020 FORMAT( 1X, ' increase ', A4, ' from ', I0,' to at least ', I0 )

!  End of subroutine SORT_reorder_by_rows

        END SUBROUTINE SORT_reorder_by_rows

!===============================================================================

      SUBROUTINE SORT_reorder_by_rows_no_vals( nr, nc, nnz, A_row, A_col, la,  &
                                               A_ptr, lptr, IW, liw, error,    &
                                               warning, inform )

!  Reorder a sparse matrix A from arbitary coordinate order to row order.

!  Programming: Nick Gould, 8th August 2002.

!  ------------------------- dummy arguments --------------------------
!
!  nr     integer, which gives the number of rows in A.
!         nr must be non-negative.
!
!  nc     integer, which gives the number of columns in A.
!         nc must be non-negative.
!
!  nnz    integer, which gives the number of nonzeros in A.
!         nnz must be non-negative.
!
!  A_row  integer array of length la. On entry, A_row(k), k = 1, ..., nnz give
!         the row indices of A. On exit, A_row will have been reordered, but
!         A_row(k) will still be the row index corresponding to the
!         entry with column index A_col(k).
!
!  A_col  integer array of length la. On entry, A_col(k), k = 1, ..., nnz give
!         the column indices of A. On exit, A_col will have been reordered so
!         that entries in row i appear directly before those in row i+1 for
!         i = 1, ..., nr-1.
!
!  la    integer, which gives the actual dimension of A_row/A_col.
!        la must be at least nnz.
!
!  A_ptr  integer array of length lptr. On exit, A_ptr(i), i = 1, ..., nr give
!         the starting addresses for the entries in A_row/A_col/A_val in
!         row i, while A_ptr(nr+1) gives the index of the first non-occupied
!         component of A.
!
!  lptr  integer, which gives the actual dimension of A_ptr.
!        lptr must be at least nr + 1.
!
!  IW     workspace integer array of length liw.
!
!  liw   integer, which gives the actual dimension of IW
!        liw  must be at least MAX( nr, nc ) + 1.
!
!  error integer, which gives the output unit number for error messages.
!        Error messages only occur if error > 0.
!
!  warning integer, which gives the output unit number for warning messages.
!        Warning messages only occur if warning > 0.
!
!  inform integer, which gives the exit status of SORT_reorder_by_rows.
!         Possible values are:
!
!          0   A has been successfully re-orderered.
!         -1   A has been successfully re-orderered, but there were duplicate
!              entries which have been summed.
!         -2   A has been successfully re-orderered, but there were row entries
!              out of range which were ignored.
!         -3   A has been successfully re-orderered, but there were column
!              entries out of range which were ignored.
!         -4   A has been successfully re-orderered, but there were no rows
!         -5   A has been successfully re-orderered, but there were no entries
!          1   nr, nc or nnz is too small. The reordering was unsuccessful.
!          2   la < nnz. The reordering was unsuccessful.
!          3   liw < MAX( nr, nc ) + 1. The reordering was unsuccessful.
!          4   lptr < nr + 1. The reordering was unsuccessful.
!          5   All entries were out of order. The reordering was unsuccessful.
!
!  ------------------ end of dummy arguments --------------------------
!
!  Dummy arguments

      INTEGER, INTENT( IN ) :: nr, nc, nnz, la, lptr, liw, error, warning
      INTEGER, INTENT( OUT ) :: inform
      INTEGER, INTENT( INOUT ), DIMENSION( la ) :: A_row, A_col
      INTEGER, INTENT( OUT ), DIMENSION( lptr ) :: A_ptr
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW

!  Local variables

      INTEGER :: i, j, k, k1, k2, l, nzi, ie, iep, je, jep
      INTEGER :: loc, idup, iout, jout, nzout

! Initialize data

      inform = 0
      nzout = 0 ; iout = 0 ; jout = 0 ; idup = 0

!  Check for faulty input data

      IF ( nr < 0 .OR. nc < 0 .OR. nnz < 0 ) THEN
        inform = 1
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' nr, nc, or nnz is out of range', /,           &
         &           ' nr = ',I0,' nc = ',I0,' nnz = ', I0 )" ) nr, nc, nnz
        END IF
        RETURN
      END IF

      IF ( la < nnz ) THEN
        inform = 2
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' increase la from', I0, ' to at least ', I0 )")&
            la, nnz
        END IF
        RETURN
      END IF

      IF ( liw < MAX( nr, nc ) + 1 ) THEN
        inform = 3
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'liw ', liw, MAX( nr, nc ) + 1
        END IF
        RETURN
      END IF

      IF ( lptr < nr + 1 ) THEN
        inform = 4
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'lptr', lptr, nr + 1
        END IF
        RETURN
      END IF

!  If the matrix has no rows or no entries, exit accordingly

      IF ( nr == 0 ) THEN
        inform = - 4
        A_ptr( 1 ) = 0
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no rows' )" )
        END IF
        RETURN
      END IF

      IF ( nnz == 0 ) THEN
        inform = - 5
        A_ptr( 1 : nr + 1 ) = 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no entries' )" )
        END IF
        RETURN
      END IF

!  Record the number of column and row indices out of order in iout and jout
!  and then remove them from consideration

      DO k = 1,nnz
        i = A_row( k ); j = A_col( k )
        IF ( i > nr .OR. i < 1 ) THEN
          iout = iout + 1
        ELSE IF ( j > nc .OR. j < 1 ) THEN
          jout = jout + 1
        ELSE
          nzout = nzout + 1
          A_row( nzout ) = i ; A_col( nzout ) = j
        END IF
      END DO

!  Inform the user if there has been faulty data

      IF ( iout > 0 ) THEN
        inform = - 2
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_row were out of',     &
         &              ' range and have been ignored by the routine')" ) iout
        END IF
      END IF

      IF ( jout > 0 ) THEN
        inform = - 3
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_col were out of',     &
         &              ' range and have been ignored by the routine')" ) jout
        END IF
      END IF

!  If all the data is faulty, exit

      IF ( iout + jout == nnz ) THEN
        inform = 5
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 4X, ' All entries input in A were out of range ' )" )
        END IF
        nzout = 0
        RETURN
      END IF

!  nzout gives the number of nonzero entries following removals. Now sort the
!  pattern of a sparse matrix from arbitary order to row order. The
!  order within each row is unimportant

!  Record the number of elements in each row in IW

      IW( : nr + 1 ) = 0
      DO k = 1, nzout
        i = A_row( k )
        IW( i ) = IW( i ) + 1
      END DO

!  Record the positions where each row would begin, in a compressed format
!  with the rows in natural order, in A_ptr and IW

      A_ptr( 1 ) = 1
      DO i = 2, nr + 1
        A_ptr( i ) = IW( i - 1 ) + A_ptr( i - 1 )
        IW( i - 1 ) = A_ptr( i - 1 )
      END DO

!  Reorder the elements into row order. Fill in each row from the front,
!  and increase the pointer IW( k ) by 1 as a new entry is placed in row k

      DO l = 1, nr
        DO k = IW( l ), A_ptr( l + 1 ) - 1
          ie = A_row( k ) ; je = A_col( k )
          DO j = 1, nzout
            IF ( ie == l ) EXIT
            loc = IW( ie )
            iep = A_row( loc ) ; jep = A_col( loc )
            IW( ie ) = loc + 1
            A_row( loc ) = ie ; A_col( loc ) = je
            ie = iep ; je = jep
          END DO
          A_row( k ) = ie ; A_col( k ) = je
        END DO
      END DO

!  Check for duplicates

      nzout = 0
      k1 = 1
      nzi = 0
      IW( : nc ) = 0
      DO i = 1, nr
        k2 = A_ptr( i + 1 ) - 1
        A_ptr( i + 1 ) = A_ptr( i )
        DO k = k1, k2
          j = A_col( k )
          IF ( IW( j ) <= nzi ) THEN
            nzout = nzout + 1
            A_row( nzout ) = A_row( k )
            A_col( nzout ) = j
            A_ptr( i + 1 ) = A_ptr( i + 1 ) + 1
            IW( j ) = nzout

!  There is a duplicate in row i

          ELSE
            idup = idup + 1
          END IF
        END DO
        k1 = k2 + 1
        nzi = nzout
      END DO
      IF ( idup > 0 ) THEN
        inform = - 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 3X, I0,' duplicate input entries' )" ) idup
        END IF
      END IF

!  Non-executable statements

 2000 FORMAT( /,' - Error return from SORT_reorder_by_rows - inform = ', I0 )
 2010 FORMAT( /,' - Warning message from SORT_reorder_by_rows - inform = ', I0 )
 2020 FORMAT( 1X, ' increase', A4, ' from', I0,' to at least ', I0 )

!  End of subroutine SORT_reorder_by_rows_no_vals

        END SUBROUTINE SORT_reorder_by_rows_no_vals

!===============================================================================
!===============================================================================

      SUBROUTINE SORT_reorder_by_cols( nr, nc, nnz, A_row, A_col, la, A_val,   &
                                       A_ptr, lptr, IW, liw, error, warning,   &
                                       inform )

!  Reorder a sparse matrix A from arbitary coordinate order to column order.

!  Programming: Nick Gould, 28 November 2011.

!  ------------------------- dummy arguments --------------------------
!
!  nr     integer, which gives the number of rows in A.
!         nr must be non-negative.
!
!  nc     integer, which gives the number of columns in A.
!         nc must be non-negative.
!
!  nnz    integer, which gives the number of nonzeros in A.
!         nnz must be non-negative.
!
!  A_row  integer array of length la. On entry, A_row(k), k = 1, ..., nnz give
!         the row indices of A. On exit, A_row will have been reordered so
!         that entries in column j appear directly before those in column j+1
!         for j = 1, ..., nc-1.
!
!  A_col  integer array of length la. On entry, A_col(k), k = 1, ..., nnz give
!         the column indices of A. On exit, A_col will have been reordered but
!         A_col(k) will still be the column index corresponding to the
!         entry with row index A_row(k).
!
!  la    integer, which gives the actual dimension of A_row/A_col/A_val.
!        la must be at least nnz.
!
!  A_val  real array of length la. On entry, A_val(k), k = 1, ..., nnz give
!         the values of A. On exit, A_val will have been reordered so that
!         entries in column j appear directly before those in column j+1
!         for j = 1, ..., nc-1 and correspond to those in A_row and A_col.
!
!  A_ptr  integer array of length lptr. On exit, A_ptr(i), j = 1, ..., nc give
!         the starting addresses for the entries in A_row/A_col/A_val in
!         column j, while A_ptr(nc+1) gives the index of the first non-occupied
!         component of A.
!
!  lptr  integer, which gives the actual dimension of A_ptr.
!        lptr must be at least nc + 1.
!
!  IW     workspace integer array of length liw.
!
!  liw   integer, which gives the actual dimension of IW
!        liw  must be at least MAX( nr, nc ) + 1.
!
!  error integer, which gives the output unit number for error messages.
!        Error messages only occur if error > 0.
!
!  warning integer, which gives the output unit number for warning messages.
!        Warning messages only occur if warning > 0.
!
!  inform integer, which gives the exit status of SORT_reorder_by_rows.
!         Possible values are:
!
!          0   A has been successfully re-orderered.
!         -1   A has been successfully re-orderered, but there were duplicate
!              entries which have been summed.
!         -2   A has been successfully re-orderered, but there were row entries
!              out of range which were ignored.
!         -3   A has been successfully re-orderered, but there were column
!              entries out of range which were ignored.
!         -4   A has been successfully re-orderered, but there were no rows
!         -5   A has been successfully re-orderered, but there were no entries
!          1   nr, nc or nnz is too small. The reordering was unsuccessful.
!          2   la < nnz. The reordering was unsuccessful.
!          3   liw < MAX( nr, nc ) + 1. The reordering was unsuccessful.
!          4   lptr < nr + 1. The reordering was unsuccessful.
!          5   All entries were out of order. The reordering was unsuccessful.
!
!  ------------------ end of dummy arguments --------------------------
!
!  Dummy arguments

      INTEGER, INTENT( IN ) :: nr, nc, nnz, la, lptr, liw, error, warning
      INTEGER, INTENT( OUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( la ) :: A_val
      INTEGER, INTENT( INOUT ), DIMENSION( la ) :: A_row, A_col
      INTEGER, INTENT( OUT ), DIMENSION( lptr ) :: A_ptr
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW

!  Local variables

      INTEGER :: i, j, k, k1, k2, l, nzi, ie, iep, je, jep
      INTEGER :: loc, idup, iout, jout, nzout
      REAL ( KIND = wp ) :: ae, aep

! Initialize data

      inform = 0
      nzout = 0 ; iout = 0 ; jout = 0 ; idup = 0

!  Check for faulty input data

      IF ( nr < 0 .OR. nc < 0 .OR. nnz < 0 ) THEN
        inform = 1
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' nr, nc, or nnz is out of range', /,           &
         &           ' nr = ',I0,' nc = ',I0,' nnz = ', I0 )" ) nr, nc, nnz
        END IF
        RETURN
      END IF

      IF ( la < nnz ) THEN
        inform = 2
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' increase la from', I0, ' to at least ', I0 )")&
            la, nnz
        END IF
        RETURN
      END IF

      IF ( liw < MAX( nr, nc ) + 1 ) THEN
        inform = 3
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'liw ', liw, MAX( nr, nc ) + 1
        END IF
        RETURN
      END IF

      IF ( lptr < nc + 1 ) THEN
        inform = 4
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'lptr', lptr, nc + 1
        END IF
        RETURN
      END IF

!  If the matrix has no rows or no entries, exit accordingly

      IF ( nr == 0 ) THEN
        inform = - 4
        A_ptr( 1 ) = 0
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no rows' )" )
        END IF
        RETURN
      END IF

      IF ( nnz == 0 ) THEN
        inform = - 5
        A_ptr( 1 : nr + 1 ) = 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no entries' )" )
        END IF
        RETURN
      END IF

!  Record the number of column and row indices out of order in iout and jout
!  and then remove them from consideration

      DO k = 1,nnz
        i = A_row( k ); j = A_col( k )
        IF ( i > nr .OR. i < 1 ) THEN
          iout = iout + 1
        ELSE IF ( j > nc .OR. j < 1 ) THEN
          jout = jout + 1
        ELSE
          nzout = nzout + 1
          A_row( nzout ) = i ; A_col( nzout ) = j
          A_val( nzout ) = A_val( k )
        END IF
      END DO

!  Inform the user if there has been faulty data

      IF ( iout > 0 ) THEN
        inform = - 2
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_row were out of',     &
         &              ' range and have been ignored by the routine')" ) iout
        END IF
      END IF

      IF ( jout > 0 ) THEN
        inform = - 3
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_col were out of',     &
         &              ' range and have been ignored by the routine')" ) jout
        END IF
      END IF

!  If all the data is faulty, exit

      IF ( iout + jout == nnz ) THEN
        inform = 5
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 4X, ' All entries input in A were out of range ' )" )
        END IF
        nzout = 0
        RETURN
      END IF

!  nzout gives the number of nonzero entries following removals. Now sort the
!  pattern of a sparse matrix from arbitary order to column order. The
!  order within each column is unimportant

!  Record the number of elements in each row in IW

      IW( : nc + 1 ) = 0
      DO k = 1, nzout
        i = A_col( k )
        IW( i ) = IW( i ) + 1
      END DO

!  Record the positions where each row would begin, in a compressed format
!  with the rows in natural order, in A_ptr and IW

      A_ptr( 1 ) = 1
      DO i = 2, nc + 1
        A_ptr( i ) = IW( i - 1 ) + A_ptr( i - 1 )
        IW( i - 1 ) = A_ptr( i - 1 )
      END DO

!  Reorder the elements into column order. Fill in each column from the front,
!  and increase the pointer IW( k ) by 1 as a new entry is placed in column k

      DO l = 1, nc
        DO k = IW( l ), A_ptr( l + 1 ) - 1
          ie = A_row( k ) ; je = A_col( k ) ; ae = A_val( k )
          DO j = 1, nzout
            IF ( je == l ) EXIT
            loc = IW( je )
            iep = A_row( loc ) ; jep = A_col( loc ) ; aep = A_val( loc )
            IW( je ) = loc + 1
            A_row( loc ) = ie ; A_col( loc ) = je ; A_val( loc ) = ae
            ie = iep ; je = jep ; ae = aep
          END DO
          A_row( k ) = ie ; A_col( k ) = je ; A_val( k ) = ae
        END DO
      END DO

!  Check for duplicates

      nzout = 0
      k1 = 1
      nzi = 0
      IW( : nc ) = 0
      DO i = 1, nc
        k2 = A_ptr( i + 1 ) - 1
        A_ptr( i + 1 ) = A_ptr( i )
        DO k = k1, k2
          j = A_row( k )
          IF ( IW( j ) <= nzi ) THEN
            nzout = nzout + 1
            A_row( nzout ) = j
            A_col( nzout ) = A_col( k ) ; A_val( nzout ) = A_val( k )
            A_ptr( i + 1 ) = A_ptr( i + 1 ) + 1
            IW( j ) = nzout

!  There is a duplicate in row i; sum the values

          ELSE
            idup = idup + 1
            A_val( IW( j ) ) = A_val( IW( j ) ) + A_val( k )
          END IF
        END DO
        k1 = k2 + 1
        nzi = nzout
      END DO
      IF ( idup > 0 ) THEN
        inform = - 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 3X, I0,' duplicate input entries summed' )" ) idup
        END IF
      END IF

!  Non-executable statements

 2000 FORMAT( /,' - Error return from SORT_reorder_by_cols - inform = ', I0 )
 2010 FORMAT( /,' - Warning message from SORT_reorder_by_cols - inform = ', I0 )
 2020 FORMAT( 1X, ' increase', A4, ' from', I0,' to at least ', I0 )

!  End of subroutine SORT_reorder_by_cols

        END SUBROUTINE SORT_reorder_by_cols

!===============================================================================

      SUBROUTINE SORT_reorder_by_cols_no_vals( nr, nc, nnz, A_row, A_col, la,  &
                                               A_ptr, lptr, IW, liw, error,    &
                                               warning, inform )

!  Reorder a sparse matrix A from arbitary coordinate order to column order.

!  Programming: Nick Gould, 28 November 2011.

!  ------------------------- dummy arguments --------------------------
!
!  nr     integer, which gives the number of rows in A.
!         nr must be non-negative.
!
!  nc     integer, which gives the number of columns in A.
!         nc must be non-negative.
!
!  nnz    integer, which gives the number of nonzeros in A.
!         nnz must be non-negative.
!
!  A_row  integer array of length la. On entry, A_row(k), k = 1, ..., nnz give
!         the row indices of A. On exit, A_row will have been reordered so
!         that entries in column j appear directly before those in column j+1
!         for j = 1, ..., nc-1.
!
!  A_col  integer array of length la. On entry, A_col(k), k = 1, ..., nnz give
!         the column indices of A. On exit, A_col will have been reordered but
!         A_col(k) will still be the column index corresponding to the
!         entry with row index A_row(k).
!
!  la    integer, which gives the actual dimension of A_row/A_col.
!        la must be at least nnz.
!
!  A_ptr  integer array of length lptr. On exit, A_ptr(i), j = 1, ..., nc give
!         the starting addresses for the entries in A_row/A_col in
!         column j, while A_ptr(nc+1) gives the index of the first non-occupied
!         component of A.
!
!  lptr  integer, which gives the actual dimension of A_ptr.
!        lptr must be at least nc + 1.
!
!  IW     workspace integer array of length liw.
!
!  liw   integer, which gives the actual dimension of IW
!        liw  must be at least MAX( nr, nc ) + 1.
!
!  error integer, which gives the output unit number for error messages.
!        Error messages only occur if error > 0.
!
!  warning integer, which gives the output unit number for warning messages.
!        Warning messages only occur if warning > 0.
!
!  inform integer, which gives the exit status of SORT_reorder_by_rows.
!         Possible values are:
!
!          0   A has been successfully re-orderered.
!         -1   A has been successfully re-orderered, but there were duplicate
!              entries which have been summed.
!         -2   A has been successfully re-orderered, but there were row entries
!              out of range which were ignored.
!         -3   A has been successfully re-orderered, but there were column
!              entries out of range which were ignored.
!         -4   A has been successfully re-orderered, but there were no rows
!         -5   A has been successfully re-orderered, but there were no entries
!          1   nr, nc or nnz is too small. The reordering was unsuccessful.
!          2   la < nnz. The reordering was unsuccessful.
!          3   liw < MAX( nr, nc ) + 1. The reordering was unsuccessful.
!          4   lptr < nr + 1. The reordering was unsuccessful.
!          5   All entries were out of order. The reordering was unsuccessful.
!
!  ------------------ end of dummy arguments --------------------------
!
!  Dummy arguments

      INTEGER, INTENT( IN ) :: nr, nc, nnz, la, lptr, liw, error, warning
      INTEGER, INTENT( OUT ) :: inform
      INTEGER, INTENT( INOUT ), DIMENSION( la ) :: A_row, A_col
      INTEGER, INTENT( OUT ), DIMENSION( lptr ) :: A_ptr
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW

!  Local variables

      INTEGER :: i, j, k, k1, k2, l, nzi, ie, iep, je, jep
      INTEGER :: loc, idup, iout, jout, nzout

! Initialize data

      inform = 0
      nzout = 0 ; iout = 0 ; jout = 0 ; idup = 0

!  Check for faulty input data

      IF ( nr < 0 .OR. nc < 0 .OR. nnz < 0 ) THEN
        inform = 1
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' nr, nc, or nnz is out of range', /,           &
         &           ' nr = ',I0,' nc = ',I0,' nnz = ', I0 )" ) nr, nc, nnz
        END IF
        RETURN
      END IF

      IF ( la < nnz ) THEN
        inform = 2
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 1X, ' increase la from', I0, ' to at least ', I0 )")&
            la, nnz
        END IF
        RETURN
      END IF

      IF ( liw < MAX( nr, nc ) + 1 ) THEN
        inform = 3
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'liw ', liw, MAX( nr, nc ) + 1
        END IF
        RETURN
      END IF

      IF ( lptr < nc + 1 ) THEN
        inform = 4
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, 2020 ) 'lptr', lptr, nc + 1
        END IF
        RETURN
      END IF

!  If the matrix has no rows or no entries, exit accordingly

      IF ( nr == 0 ) THEN
        inform = - 4
        A_ptr( 1 ) = 0
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no rows' )" )
        END IF
        RETURN
      END IF

      IF ( nnz == 0 ) THEN
        inform = - 5
        A_ptr( 1 : nr + 1 ) = 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( '   the matrix has no entries' )" )
        END IF
        RETURN
      END IF

!  Record the number of column and row indices out of order in iout and jout
!  and then remove them from consideration

      DO k = 1,nnz
        i = A_row( k ); j = A_col( k )
        IF ( i > nr .OR. i < 1 ) THEN
          iout = iout + 1
        ELSE IF ( j > nc .OR. j < 1 ) THEN
          jout = jout + 1
        ELSE
          nzout = nzout + 1
          A_row( nzout ) = i ; A_col( nzout ) = j
        END IF
      END DO

!  Inform the user if there has been faulty data

      IF ( iout > 0 ) THEN
        inform = - 2
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_row were out of',     &
         &              ' range and have been ignored by the routine')" ) iout
        END IF
      END IF

      IF ( jout > 0 ) THEN
        inform = - 3
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 1X, I0,' entries input in A_col were out of',     &
         &              ' range and have been ignored by the routine')" ) jout
        END IF
      END IF

!  If all the data is faulty, exit

      IF ( iout + jout == nnz ) THEN
        inform = 5
        IF ( error > 0 ) THEN
          WRITE( error, 2000 ) inform
          WRITE( error, "( 4X, ' All entries input in A were out of range ' )" )
        END IF
        nzout = 0
        RETURN
      END IF

!  nzout gives the number of nonzero entries following removals. Now sort the
!  pattern of a sparse matrix from arbitary order to column order. The
!  order within each column is unimportant

!  Record the number of elements in each row in IW

      IW( : nc + 1 ) = 0
      DO k = 1, nzout
        i = A_col( k )
        IW( i ) = IW( i ) + 1
      END DO

!  Record the positions where each row would begin, in a compressed format
!  with the rows in natural order, in A_ptr and IW

      A_ptr( 1 ) = 1
      DO i = 2, nc + 1
        A_ptr( i ) = IW( i - 1 ) + A_ptr( i - 1 )
        IW( i - 1 ) = A_ptr( i - 1 )
      END DO

!  Reorder the elements into column order. Fill in each column from the front,
!  and increase the pointer IW( k ) by 1 as a new entry is placed in column k

      DO l = 1, nc
        DO k = IW( l ), A_ptr( l + 1 ) - 1
          ie = A_row( k ) ; je = A_col( k )
          DO j = 1, nzout
            IF ( je == l ) EXIT
            loc = IW( je )
            iep = A_row( loc ) ; jep = A_col( loc )
            IW( je ) = loc + 1
            A_row( loc ) = ie ; A_col( loc ) = je
            ie = iep ; je = jep
          END DO
          A_row( k ) = ie ; A_col( k ) = je
        END DO
      END DO

!  Check for duplicates

      nzout = 0
      k1 = 1
      nzi = 0
      IW( : nc ) = 0
      DO i = 1, nc
        k2 = A_ptr( i + 1 ) - 1
        A_ptr( i + 1 ) = A_ptr( i )
        DO k = k1, k2
          j = A_row( k )
          IF ( IW( j ) <= nzi ) THEN
            nzout = nzout + 1
            A_row( nzout ) = j
            A_col( nzout ) = A_col( k )
            A_ptr( i + 1 ) = A_ptr( i + 1 ) + 1
            IW( j ) = nzout

!  There is a duplicate in row i

          ELSE
            idup = idup + 1
          END IF
        END DO
        k1 = k2 + 1
        nzi = nzout
      END DO
      IF ( idup > 0 ) THEN
        inform = - 1
        IF ( warning > 0 ) THEN
          WRITE( warning, 2010 ) inform
          WRITE( warning, "( 3X, I0,' duplicate input entries summed' )" ) idup
        END IF
      END IF

!  Non-executable statements

 2000 FORMAT( /,' - Error return from SORT_reorder_by_cols - inform = ', I0 )
 2010 FORMAT( /,' - Warning message from SORT_reorder_by_cols - inform = ', I0 )
 2020 FORMAT( 1X, ' increase', A4, ' from', I0,' to at least ', I0 )

!  End of subroutine SORT_reorder_by_cols_no_vals

        END SUBROUTINE SORT_reorder_by_cols_no_vals

   END MODULE GALAHAD_SORT_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*     END SORT  M O D U L E   *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
