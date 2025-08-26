! THIS VERSION: GALAHAD 5.3 - 2025-08-25 AT 09:30 GMT.

!#include "galahad_modules.h"
#include "ssids_procedures.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ M O   M O D U L E  *-*-*-*-*-*-*-*-*-*-*-

!      -----------------------------------------------------------------
!     | Matrix ordering package originally spral_match_order from SPRAL |
!      -----------------------------------------------------------------

!  COPYRIGHT (c) 2012-3 Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  authors: Jonathan Hogg and Jennifer Scott
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

      MODULE GALAHAD_MO_precision

!  given a sparse symmetric matrix A, this module provides routines to
!  use a matching algorithm to compute an elimination order that is suitable 
!  for use with a sparse direct solver. It optionally computes scaling factors

!  FIXME: At some stage replace call to mo_match() with call to a higher level 
!  routine from GALAHAD_MS instead (NB: have to cope with fact we are currently
!  expecting a full matrix, even if it means 2x more log operations)

        USE GALAHAD_KINDS_precision
        USE GALAHAD_NODEND_precision, ONLY: NODEND_half_order,                 &
                                            NODEND_control_type,               &
                                            NODEND_inform_type
        USE GALAHAD_MS_precision, ONLY: MS_hungarian_match
        IMPLICIT NONE

        PRIVATE

!  find a matching-based ordering using the Hungarian algorithm 
!  (Harold W. Kuhn, "The Hungarian Method for the assignment problem",
!  Naval Research Logistics Quarterly, 2: 83–97, 1955) for matching 
!  and METIS (George Karypis & Vipin Kumar, 1995) for ordering

        PUBLIC :: MO_match_order_metis 

!  error flags

        INTEGER ( KIND = ip_ ), PARAMETER :: SUCCESS = 0
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_ALLOCATION = - 1
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_A_N_OOR = - 2
        INTEGER ( KIND = ip_ ), PARAMETER :: ERROR_UNKNOWN = - 99

!  warning flags

        INTEGER ( KIND = ip_ ), PARAMETER :: WARNING_SINGULAR = 1

        INTERFACE MO_match_order_metis
          MODULE PROCEDURE match_order_metis_ptr32, match_order_metis_ptr64
        END INTERFACE MO_match_order_metis

      CONTAINS

!-*-*-*-*-*-*-   M O  match_order_metis_ptr32  S U B R O U T I N E  -*-*-*-*-*-

        SUBROUTINE match_order_metis_ptr32( n, ptr, row, val, order, scale,    &
                                            nodend_control, nodend_inform,     &
                                            flag, stat )

!  on input ptr, row , val hold the ** lower AND upper ** triangular parts of 
!  the matrix. this reduces amount of copies of matrix required (so slightly
!  more efficient on memory and does not need to expand supplied matrix )

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = i4_ ), DIMENSION ( : ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( : ), INTENT ( IN ) :: val

!  order(i) holds the position of variable i in the elimination order 
!  (pivot sequence)

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( OUT ) :: order 

!  returns the hungarian_match (mc64) symmetric scaling as proposed by
!   Duff, I. S. and Koster, J. On algorithms for permuting large entries to
!   the diagonal of a sparse matrix. SIAM J Matrix Analysis and Applications
!   22 (4), 973-996, 2001.

        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scale
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag ! return value

! stat value returned on failed allocation

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: stat
        TYPE ( NODEND_control_type ), INTENT ( IN ) :: nodend_control
        TYPE ( NODEND_inform_type ), INTENT ( INOUT ) :: nodend_inform

! used to hold matching

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: cperm

! column pointers for expanded matrix.

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr2

! row indices for expanded matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: row2

! entries of expanded matrix

        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: val2

        INTEGER ( KIND = ip_ ) :: i, j, ne
        INTEGER ( KIND = i8_ ) :: k

        flag = 0 ; stat = 0

!  check n has valid value

        IF ( n < 0 ) THEN
          flag = ERROR_A_N_OOR
          RETURN
        END IF

!  just return with no action if n = 0

        IF ( n == 0 ) RETURN

!  remove any explicit zeroes and take absolute values

        ne = ptr( n + 1 ) - 1
        ALLOCATE ( ptr2( n + 1 ), row2( ne ), val2( ne ), cperm( n ),          &
                   STAT = stat )
        IF ( stat /= 0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

        k = 1
        DO i = 1, n
          ptr2( i ) = k
          DO j = ptr( i ), ptr( i + 1 ) - 1
            IF ( val( j ) == 0.0 ) CYCLE
            row2( k ) = row( j )
            val2( k ) = ABS( val( j ) )
            k = k + 1
          END DO
        END DO
        ptr2( n + 1 ) = k

!  compute matching and scaling

        CALL mo_scale( n, ptr2, row2, val2, scale, flag, stat, perm = cperm )
        DEALLOCATE ( val2, STAT = stat )

        IF ( flag < 0 ) RETURN

!  note: row j is matched with column cperm(j)
!  write ( *,'( a,15i4 )' ) 'cperm',cperm(1:min(15,n))

!  Split matching into 1- and 2-cycles only and then compress matrix and order

        CALL mo_split( n, row2, ptr2, order, cperm, nodend_control,            &
                       nodend_inform, flag, stat )

        scale( 1 : n ) = EXP( SCALE( 1 : n ) )
        RETURN

        END SUBROUTINE match_order_metis_ptr32

!-*-*-*-*-*-*-   M O  match_order_metis_ptr64  S U B R O U T I N E  -*-*-*-*-*-

        SUBROUTINE match_order_metis_ptr64( n, ptr, row, val, order, scale,    &
                                            nodend_control, nodend_inform,     &
                                            flag, stat )

!  on input ptr, row , val hold the ** lower AND upper ** triangular parts of 
!  the matrix. This reduces amount of copies of matrix required (so slightly
!  more efficient on memory and does not need to expand supplied matrix)

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = i8_ ), DIMENSION ( : ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( : ), INTENT ( IN ) :: val

!  order(i)  holds the position of variable i in the elimination order 
!  (pivot sequence)

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( OUT ) :: order

!  returns the hungarian_match symmetric scaling

        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scale
        TYPE ( NODEND_control_type ), INTENT ( IN ) :: nodend_control
        TYPE ( NODEND_inform_type ), INTENT ( INOUT ) :: nodend_inform
        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: flag ! return value

!  stat value returned on failed allocation

        INTEGER ( KIND = ip_ ), INTENT ( OUT ) :: stat 

!  used to hold matching

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: cperm

!  column pointers for expanded matrix

        INTEGER ( KIND = i8_ ), DIMENSION ( : ), ALLOCATABLE :: ptr2

!  row indices for expanded matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: row2

!  entries of expanded matrix

        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: val2

        INTEGER ( KIND = ip_ ) :: i
        INTEGER ( KIND = i8_ ) :: j, k, ne

        flag = 0 ; stat = 0

!  check n has valid value

        IF ( n < 0 ) THEN
          flag = ERROR_A_N_OOR
          RETURN
        END IF

!  just return with no action if n = 0

        IF ( n == 0 ) RETURN

!  remove any explicit zeroes and take absolute values

        ne = ptr( n + 1 ) - 1
        ALLOCATE ( ptr2( n + 1 ), row2( ne ), val2( ne ), cperm( n ),          &
                   STAT = stat )
        IF ( stat /= 0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

        k = 1
        DO i = 1, n
          ptr2( i ) = k
          DO j = ptr( i ), ptr( i + 1 ) - 1
            IF ( val( j ) == 0.0 ) CYCLE
            row2( k ) = row( j )
            val2( k ) = ABS( val( j ) )
            k = k + 1
          END DO
        END DO
        ptr2( n + 1 ) = k

!  compute matching and scaling

        CALL mo_scale( n, ptr2, row2, val2, scale, flag, stat, perm = cperm )
        DEALLOCATE ( val2, STAT = stat )
        IF ( flag < 0 ) RETURN

!  note: row j is matched with column cperm(j)
!  write (*,'(a,15i4)') 'cperm',cperm(1:min(15,n))
!  split matching into 1- and 2-cycles only and then compress matrix and order

        CALL mo_split( n, row2, ptr2, order, cperm, nodend_control,            &
                       nodend_inform, flag, stat )

        scale( 1 : n ) = EXP( SCALE( 1 : n ) )
        RETURN

        END SUBROUTINE match_order_metis_ptr64

!-*-*-*-*-*-*-*-*-*-   M O _ S P L I T   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

        SUBROUTINE mo_split( n, row2, ptr2, order, cperm, nodend_control,      &
                             nodend_inform, flag, stat )

!  split matching into 1- and 2-cycles only and then compress matrix and 
!  order using Metis

!  Input (ptr2, row2 , val2) holds the ** lower and upper triangles ** of
!  the matrix (with explicit zeros removed). Overwritten in the singular case

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( : ), INTENT ( IN ) :: ptr2
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( IN ) :: row2

!  used to hold ordering

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( OUT ) :: order

!  used to hold matching

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( INOUT ) :: cperm
        TYPE ( NODEND_control_type ), INTENT ( IN ) :: nodend_control
        TYPE ( NODEND_inform_type ), INTENT ( INOUT ) :: nodend_inform
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: flag
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: stat

!  workspace array

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: iwork

!  holds mapping between original matrix indices and those in condensed matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ),                               &
                                ALLOCATABLE :: old_to_new, new_to_old

!  column pointers for condensed matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: ptr3

!  row indices for condensed matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: row3

        INTEGER ( KIND = ip_ ) :: csz ! current cycle length
        INTEGER ( KIND = ip_ ) :: i, j, j1, j2, jj, k, krow, metis_flag
        INTEGER ( KIND = long_ ) :: klong
        INTEGER ( KIND = ip_ ) :: max_csz ! maximum cycle length
        INTEGER ( KIND = ip_ ) :: ncomp ! order of compressed matrix

! order of compressed matrix (matched entries only)

        INTEGER ( KIND = ip_ ) :: ncomp_matched
        INTEGER ( KIND = long_ ) :: ne ! number of non zeros
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: invp

! Use iwork to track what has been matched:
!  -2 unmatched
!  -1 matched as singleton
!   0 not yet seen
!  >0 matched with specified node

        ne = ptr2( n + 1 ) - 1
        ALLOCATE ( ptr3( n + 1 ), row3( ne ), old_to_new( n ),                 &
                   new_to_old( n ), iwork( n ), STAT = stat )
        IF ( stat /= 0 ) RETURN

        iwork( 1 : n ) = 0
        max_csz = 0
        DO i = 1, n
          IF ( iwork( i ) /= 0 ) CYCLE
          j = i
          csz = 0
          DO

!  unmatched by hungarian_match

            IF ( cperm( j ) == - 1 ) THEN
              iwork( j ) = - 2
              csz = csz + 1
              EXIT

!  match as singleton, unmatched or finished

            ELSE IF ( cperm( j ) == i ) THEN
              iwork( j ) = - 1
              csz = csz + 1
              EXIT
            END IF

!  match j and cperm( j )

            jj = cperm( j )
            iwork( j ) = jj
            iwork( jj ) = j
            csz = csz + 2

! move onto next start of pair

            j = cperm( jj )
            IF ( j == i ) EXIT
          END DO
          max_csz = MAX( max_csz, csz )
        END DO

!  overwrite cperm with new matching

        cperm( 1 : n ) = iwork( 1 : n )

!  build maps for new numbering schemes

        k = 1
        DO i = 1, n
          j = cperm( i )
          IF ( j < i .AND. j > 0 ) CYCLE
          old_to_new( i ) = k
          new_to_old( k ) = i ! note: new_to_old only maps to first of a pair
          IF ( j > 0 ) old_to_new( j ) = k
          k = k + 1
        END DO
        ncomp_matched = k - 1

!  produce a condensed version of the matrix for ordering. Hold pattern using 
!  ptr3 and row3

        ptr3( 1 ) = 1
        iwork( : ) = 0 ! Use to indicate if entry is in a paired column
        ncomp = 1
        jj = 1
        DO i = 1, n
          j = cperm( i )
          IF ( j < i .AND. j > 0 ) CYCLE ! already seen
          DO klong = ptr2( i ), ptr2( i + 1 ) - 1
            krow = old_to_new( row2( klong ) )
            IF ( iwork( krow ) == i ) CYCLE ! already added to column
            IF ( krow > ncomp_matched ) CYCLE ! unmatched row not participating
            row3( jj ) = krow
            jj = jj + 1
            iwork( krow ) = i
          END DO

!  also check column cperm( i )

          IF ( j > 0 ) THEN
            DO klong = ptr2( j ), ptr2( j + 1 ) - 1
              krow = old_to_new( row2( klong ) )
              IF ( iwork( krow ) == i ) CYCLE ! already added to column
              IF ( krow > ncomp_matched ) CYCLE !unmatched row not participating
              row3( jj ) = krow
              jj = jj + 1
              iwork( krow ) = i
            END DO
          END IF
          ptr3( ncomp + 1 ) = jj
          ncomp = ncomp + 1
        END DO
        ncomp = ncomp - 1

!  store just lower triangular part for input to Metis

        ptr3( 1 ) = 1
        jj = 1
        j1 = 1
        DO i = 1, ncomp
          j2 = ptr3( i + 1 )
          DO k = j1, j2 - 1
            krow = row3( k )
            IF ( krow < i ) CYCLE ! already added to column
            row3( jj ) = krow
            jj = jj + 1
          END DO
          ptr3( i + 1 ) = jj
          j1 = j2
        END DO

        ALLOCATE ( invp( ncomp ), STAT = stat )
        IF ( stat /= 0 ) RETURN

!  reorder the compressed matrix using Metis

        CALL nodend_half_order( ncomp, ptr3, row3, order, nodend_control,      &
                               nodend_inform )
        metis_flag = nodend_inform%status
        stat = nodend_inform%alloc_status

        SELECT CASE ( metis_flag )
        CASE ( 0 ) ! OK, do nothing
        CASE ( - 1 ) ! allocation error
          flag = ERROR_ALLOCATION
          RETURN
        CASE DEFAULT ! unknown error, should never happen
          PRINT *, 'metis_order() returned unknown error ', metis_flag
          flag = ERROR_UNKNOWN
        END SELECT

        DO i = 1, ncomp
          j = order( i )
          iwork( j ) = i
        END DO

!  translate inverse permutation in iwork back to permutation for 
!  original variables

        k = 1
        DO i = 1, ncomp
          j = new_to_old( iwork( i ) )
          order( j ) = k
          k = k + 1
          IF ( cperm( j ) > 0 ) THEN
            j = cperm( j )
            order( j ) = k
            k = k + 1
          END IF
        END DO
        RETURN

        END SUBROUTINE mo_split

!-*-*-*-*-*-*-*-*-*-   M O _ S C A L E    S U B R O U T I N E  -*-*-*-*-*-*-*-*-

        SUBROUTINE mo_scale( n, ptr, row, val, scale, flag, stat, perm )

!  scale the matrix using hungarian_match, accounting for singular matrices 
!  using the approach of Duff and Pralet (SIAM Journal Matrix Analysis and 
!  Applications 27 313-340, 2015)

!  expects a full matrix as input

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n
        INTEGER ( KIND = long_ ), DIMENSION ( : ), INTENT ( IN ) :: ptr
        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( IN ) :: row
        REAL ( KIND = rp_ ), DIMENSION ( : ), INTENT ( IN ) :: val

!  returns the symmetric scaling

        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scale
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: flag
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: stat

!  if present, returns the matching

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( OUT ),               &
                                                 OPTIONAL :: perm

!  column pointers after zeros removed

        INTEGER ( KIND = long_ ), DIMENSION ( : ), ALLOCATABLE :: ptr2

!  row indices after zeros removed

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: row2

!  matrix of absolute values (zeros removed)

        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: val2

!  temporary copy of scaling factors. Only needed if A is rank deficient

        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: cscale

        INTEGER ( KIND = ip_ ) :: i, struct_rank
        INTEGER ( KIND = long_ ) :: j, k, ne

        struct_rank = n

!  remove any explicit zeroes and take absolute values

        ne = ptr( n + 1 ) - 1
        ALLOCATE ( ptr2( n + 1 ), row2( ne ), val2( ne ), STAT = stat )
        IF ( stat /= 0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

        k = 1
        DO i = 1, n
          ptr2( i ) = k
          DO j = ptr( i ), ptr( i + 1 ) - 1
            IF ( val( j ) == 0.0 ) CYCLE
            row2( k ) = row( j )
            val2( k ) = val( j )
            k = k + 1
          END DO
        END DO
        ptr2( n + 1 ) = k

        CALL mo_match( n, row2, ptr2, val2, scale, flag, stat, perm = perm )
        IF ( flag < 0 ) RETURN

!  structurally singular case. At this point, scaling factors
!  for rows in corresponding to rank deficient part are set to
!  zero. The following is to set them according to Duff and Pralet.

        IF ( struct_rank /= n ) THEN
          DEALLOCATE ( ptr2, row2, val2, STAT = stat )
          ALLOCATE ( cscale( n ), STAT = stat )
          IF ( stat /= 0 ) THEN
            flag = ERROR_ALLOCATION
            RETURN
          END IF
          cscale( 1 : n ) = scale( 1 : n )
          DO i = 1, n
            IF ( cscale( i ) /= - HUGE( scale ) ) CYCLE
            DO j = ptr( i ), ptr( i + 1 ) - 1
              k = row( j )
              IF ( cscale( k ) == - HUGE( scale ) ) CYCLE
              scale( i ) = MAX( scale( i ), val( j ) + scale( k ) )
            END DO
            IF ( scale( i ) == - HUGE( scale ) ) THEN
              scale( i ) = 0.0
            ELSE
              scale( i ) = - scale( i )
            END IF
          END DO
        END IF
        RETURN

        END SUBROUTINE mo_scale

!-*-*-*-*-*-*-*-*-*-   M O _ M A T C H   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

        SUBROUTINE mo_match( n, row2, ptr2, val2, scale, flag, stat, perm )

!  input ( ptr2, row2 , val2 ) holds the ** lower and upper triangles ** of 
!  the matrix (with explicit zeros removed). val2 holds absolute values of 
!  matrix entries, overwritten in the singular case

        IMPLICIT NONE
        INTEGER ( KIND = ip_ ), INTENT ( IN ) :: n

!  in singular case, overwritten by column pointers for non singular part 
!  of matrix

        INTEGER ( KIND = long_ ), DIMENSION ( : ), INTENT ( INOUT ) :: ptr2

!  in singular case, overwritten by row indices for non singular part of matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), INTENT ( INOUT ) :: row2

!  in singular case, overwritten by entries for non singular part of matrix

        REAL ( KIND = rp_ ), DIMENSION ( : ), INTENT ( INOUT ) :: val2 

!  returns the symmetric scaling

        REAL ( KIND = rp_ ), DIMENSION ( n ), INTENT ( OUT ) :: scale
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: flag
        INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: stat

!  if present, returns the matching

        INTEGER ( KIND = ip_ ), DIMENSION ( n ), INTENT ( OUT ),               &
                                                 OPTIONAL :: perm

!  used to hold matching

        INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: cperm

!  holds mapping between original matrix indices and those in reduced
!  non singular matrix

        INTEGER ( KIND = ip_ ), DIMENSION ( : ),                               &
                                ALLOCATABLE :: old_to_new, new_to_old

!  (log) column maximum

        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: cmax

!  array used by hungarian_match

        REAL ( KIND = rp_ ), DIMENSION ( : ), ALLOCATABLE :: dw1, dw2
        INTEGER ( KIND = ip_ ) :: i, j, jj, k
        INTEGER ( KIND = long_ ) :: jlong, j1, j2
        INTEGER ( KIND = long_ ) :: ne ! number of non zeros

!  holds number of rows/cols in non singular part of matrix

        INTEGER ( KIND = ip_ ) :: nn

!  only used in singular case. Holds number of non zeros in non-singular part 
!  of matrix

        INTEGER ( KIND = ip_ ) :: nne
        INTEGER ( KIND = ip_ ) :: rank ! returned by 
        REAL ( KIND = rp_ ) :: colmax ! max. entry in col. of expanded matrix

        ALLOCATE ( cperm( n ), dw1( n ), dw2( n ), cmax( n ), STAT = stat )
        IF ( stat /= 0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

!  compute column maximums

        DO i = 1, n
          colmax                                                               &
            = MAX( 0.0_rp_, MAXVAL( val2( ptr2( i ) : ptr2( i + 1 ) - 1 ) ) )
          IF ( colmax /= 0.0_rp_ ) colmax = LOG( colmax )
          cmax( i ) = colmax
        END DO

        DO i = 1, n
          val2( ptr2( i ) : ptr2( i + 1 ) - 1 )                                &
            = cmax( i ) - LOG( val2( ptr2( i ) : ptr2( i + 1 ) - 1 ) )
        END DO

        ne = ptr2( n + 1 ) - 1
        CALL MS_hungarian_match( n, n, ptr2, row2, val2, cperm, rank,          &
                                 dw1, dw2, stat )
        IF ( stat /= 0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

        IF ( rank == n ) THEN
          DO i = 1, n
            scale( i ) = ( dw1( i ) + dw2( i ) - cmax( i ) ) / 2
          END DO
          IF ( PRESENT( perm ) ) perm( 1 : n ) = cperm( 1 : n )
          RETURN
        END IF

!  handle the singular case. Either immediate exit or set warning, squeeze 
!  out the unmatched entries and recall hungarian_match

        flag = WARNING_SINGULAR

        ALLOCATE ( old_to_new( n ), new_to_old( n ), STAT = stat )
        IF ( stat  /=  0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

        k = 0
        DO i = 1, n
          IF ( cperm( i )  ==  0 ) THEN

!  row i and col j are not part of the matching

            old_to_new( i ) = - 1
          ELSE
            k = k + 1

!  old_to_new( i ) holds the new index for variable i after removal of
!  singular part and new_to_old( k ) is the original index for k

            old_to_new( i ) = k
            new_to_old( k ) = i
          END IF
        END DO

!  overwrite ptr2, row2 and val2

        nne = 0
        k = 0
        ptr2( 1 ) = 1
        j2 = 1
        DO i = 1, n
          j1 = j2
          j2 = ptr2( i + 1 )

!  skip over unmatched entries

          IF ( cperm( i ) == 0 ) CYCLE
          k = k + 1
          DO jlong = j1, j2 - 1
            jj = row2( jlong )
            IF ( cperm( jj ) == 0 ) CYCLE
            nne = nne + 1
            row2( nne ) = old_to_new( jj )
            val2( nne ) = val2( jlong )
          END DO
          ptr2( k + 1 ) = nne + 1
        END DO

!  nn is order of non-singular part.

        nn = k
        CALL MS_hungarian_match( nn, nn, ptr2, row2, val2, cperm, rank,        &
                                 dw1, dw2, stat )
        IF ( stat /= 0 ) THEN
          flag = ERROR_ALLOCATION
          RETURN
        END IF

        DO i = 1, n
          j = old_to_new( i )
          IF ( j < 0 ) THEN
            scale( i ) = - HUGE( scale )
          ELSE

!  note: we need to subtract col max using old matrix numbering

            scale( i ) = ( dw1( j ) + dw2( j ) - cmax( i ) ) / 2
          END IF
        END DO

        IF ( PRESENT( perm ) ) THEN
          perm( 1 : n ) = - 1
          DO i = 1, nn
            j = cperm( i )
            perm( new_to_old( i ) ) = new_to_old( j )
          END DO
        END IF
        RETURN

        END SUBROUTINE mo_match

      END MODULE GALAHAD_MO_precision
