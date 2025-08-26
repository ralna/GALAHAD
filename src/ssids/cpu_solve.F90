! THIS VERSION: GALAHAD 5.1 - 2024-11-21 AT 09:40 GMT.

#include "galahad_blas.h"
#include "ssids_procedures.h"

#ifdef GALAHAD_BLAS
#ifdef REAL_32
#ifdef INTEGER_64
#define trsm galahad_strsm_64
#define trsv galahad_strsv_64
#define gemm galahad_sgemm_64
#define gemv galahad_sgemv_64
#else
#define trsm galahad_strsm
#define trsv galahad_strsv
#define gemm galahad_sgemm
#define gemv galahad_sgemv
#endif
#elif REAL_128
#ifdef INTEGER_64
#define trsm galahad_qtrsm_64
#define trsv galahad_qtrsv_64
#define gemm galahad_qgemm_64
#define gemv galahad_qgemv_64
#else
#define trsm galahad_qtrsm
#define trsv galahad_qtrsv
#define gemm galahad_qgemm
#define gemv galahad_qgemv
#endif
#else
#ifdef INTEGER_64
#define trsm galahad_dtrsm_64
#define trsv galahad_dtrsv_64
#define gemm galahad_dgemm_64
#define gemv galahad_dgemv_64
#else
#define trsm galahad_dtrsm
#define trsv galahad_dtrsv
#define gemm galahad_dgemm
#define gemv galahad_dgemv
#endif
#endif
#else
#ifdef REAL_32
#ifdef INTEGER_64
#define trsm strsm_64
#define trsv strsv_64
#define gemm sgemm_64
#define gemv sgemv_64
#else
#define trsm strsm
#define trsv strsv
#define gemm sgemm
#define gemv sgemv
#endif
#elif REAL_128
#ifdef INTEGER_64
#define trsm qtrsm_64
#define trsv qtrsv_64
#define gemm qgemm_64
#define gemv qgemv_64
#else
#define trsm qtrsm
#define trsv qtrsv
#define gemm qgemm
#define gemv qgemv
#endif
#else
#ifdef INTEGER_64
#define trsm dtrsm_64
#define trsv dtrsv_64
#define gemm dgemm_64
#define gemv dgemv_64
#else
#define trsm dtrsm
#define trsv dtrsv
#define gemm dgemm
#define gemv dgemv
#endif
#endif
#endif

! This module provides a way of doing solve on CPU using GPU data structures

  MODULE GALAHAD_SSIDS_gpu_cpu_solve_precision
    USE GALAHAD_KINDS_precision
    USE GALAHAD_SSIDS_types_precision
    IMPLICIT none

    PRIVATE
    PUBLIC :: subtree_bwd_solve, fwd_diag_solve

  CONTAINS

!*************************************************************************
!
! This subroutine performs a backwards solve on the chunk of nodes specified
! by sa:en.
!
    SUBROUTINE subtree_bwd_solve( en, sa, job, pos_def, nnodes, nodes, sptr,  &
                                  rptr, rlist, invp, nrhs, x, ldx, st )
    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: en
    INTEGER( ip_ ), INTENT( IN ) :: sa
    LOGICAL, INTENT( IN ) :: pos_def
    INTEGER( ip_ ), INTENT( IN ) :: job ! controls whether we are doing forward
      ! eliminations, back substitutions etc.
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    type( node_type ), DIMENSION( * ), INTENT( IN ) :: nodes
    INTEGER( ip_ ), DIMENSION( nnodes+1 ), INTENT( IN ) :: sptr
    INTEGER( long_ ), DIMENSION( nnodes+1 ), INTENT( IN ) :: rptr
    INTEGER( ip_ ), DIMENSION( rptr( nnodes+1 )-1 ), INTENT( IN ) :: rlist
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    REAL( rp_ ), DIMENSION( ldx,nrhs ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( OUT ) :: st  ! stat parameter

    INTEGER( ip_ ) :: blkm
    INTEGER( ip_ ) :: blkn
    INTEGER( ip_ ) :: nd
    INTEGER( ip_ ) :: nelim
    INTEGER( ip_ ) :: node
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: xlocal
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: map

    st = 0
    ALLOCATE( xlocal( nrhs * ( sptr( nnodes + 1 ) - 1 ) ),                     &
              map( sptr( nnodes + 1 ) - 1 ), STAT = st )

!  backwards solve DL^Tx = z or L^Tx = z

    DO node = en, sa, - 1
      nelim = nodes( node )%nelim
      IF ( nelim == 0 ) cycle
      nd = nodes( node )%ndelay
      blkn = sptr( node + 1 ) - sptr( node ) + nd
      blkm = int( rptr( node + 1 ) - rptr( node ) ) + nd

      IF ( nrhs == 1 ) THEN
        CALL solve_bwd_one( pos_def, job, rlist( rptr( node ) ), invp, x,      &
                            blkm, blkn, nelim, nd, nodes( node )%lcol,         &
                            nodes( node )%lcol( 1 + blkm * blkn : ),           &
                            nodes( node )%perm, xlocal, map )
      ELSE
        CALL solve_bwd_mult( pos_def, job, rlist( rptr( node ) ), invp,        &
                             nrhs, x, ldx, blkm, blkn, nelim, nd,              &
                             nodes( node )%lcol,                               &
                             nodes( node )%lcol( 1 + blkm * blkn : ),          &
                             nodes( node )%perm, xlocal, map )
      END IF
    END DO
  END SUBROUTINE subtree_bwd_solve

!*************************************************************************
!
! Provides serial versions of Forward ( s/n ) and diagonal solves.
!
    SUBROUTINE fwd_diag_solve( pos_def, job, nnodes, nodes, sptr, rptr, rlist, &
                               invp, nrhs, x, ldx, st )
    IMPLICIT none
    LOGICAL, INTENT( IN ) :: pos_def

!  controls whether we are doing forward eliminations, back substitutions etc.

    INTEGER( ip_ ), INTENT( IN ) :: job
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    type( node_type ), DIMENSION( * ), INTENT( IN ) :: nodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr
    INTEGER( ip_ ), DIMENSION( rptr( nnodes + 1 ) - 1 ), INTENT( IN ) :: rlist
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    REAL( rp_ ), DIMENSION( ldx,nrhs ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( OUT ) :: st  ! stat parameter

    INTEGER( ip_ ) :: blkm, blkn, nd, nelim, node
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: xlocal
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: map

    st = 0
    ALLOCATE( xlocal( nrhs * ( sptr( nnodes + 1 ) - 1 ) ),                     &
              map( sptr( nnodes + 1 ) - 1 ), STAT = st )
    IF ( st /= 0 ) RETURN

!  forwards solve Ly = b

    IF ( job == SSIDS_SOLVE_JOB_ALL .OR. job == SSIDS_SOLVE_JOB_FWD ) THEN
      DO node = 1, nnodes
        nelim = nodes( node )%nelim
        IF ( nelim == 0 ) CYCLE
        nd = nodes( node )%ndelay
        blkn = sptr( node + 1 ) - sptr( node ) + nd
        blkm = INT( rptr( node + 1 ) - rptr( node ) ) + nd

        IF ( nrhs == 1 ) THEN
          CALL solve_fwd_one( pos_def, rlist( rptr( node ) ), invp, x,         &
                              blkm, blkn, nelim, nd, nodes( node )%lcol,       &
                              nodes( node )%perm, xlocal, map )
        ELSE
          CALL solve_fwd_mult( pos_def, rlist( rptr( node ) ), invp,           &
                               nrhs, x, ldx, blkm, blkn, nelim, nd,            &
                               nodes( node )%lcol, nodes( node )%perm,         &
                               xlocal, map )
        END IF
      END DO
    END IF

!  diagonal solve Dx = z

    IF ( job == SSIDS_SOLVE_JOB_DIAG ) THEN
      DO node = nnodes, 1, - 1
        nelim = nodes( node )%nelim
        IF ( nelim == 0 ) CYCLE
        nd = nodes( node )%ndelay
        blkn = sptr( node + 1 ) - sptr( node ) + nd
        blkm = int( rptr( node + 1 ) - rptr( node ) ) + nd

        IF ( nrhs == 1 ) THEN
          CALL solve_diag_one( invp, x, nelim,                                 &
               nodes( node )%rsmptr%rmem( nodes( node )%rsmsa+blkm*blkn ),     &
               nodes( node )%ismptr%imem( nodes( node )%ismsa ) )
        ELSE
          CALL solve_diag_mult( invp, nrhs, x, ldx, nelim,                     &
               nodes( node )%rsmptr%rmem( nodes( node )%rsmsa+blkm*blkn ),     &
               nodes( node )%ismptr%imem( nodes( node )%ismsa ) )
        END IF
      END DO
    END IF
    END SUBROUTINE fwd_diag_solve

    SUBROUTINE solve_fwd_one( pos_def, rlist, invp, x, blkm, blkn, nelim, nd, &
                              lcol, lperm, xlocal, map )

!  forward substitution single rhs

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: pos_def
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: rlist
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: blkm
    INTEGER( ip_ ), INTENT( IN ) :: blkn
    INTEGER( ip_ ), INTENT( IN ) :: nelim
    INTEGER( ip_ ), INTENT( IN ) :: nd
    REAL( rp_ ), DIMENSION( * ), INTENT( IN ) :: lcol
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: lperm
    REAL( rp_ ), DIMENSION( * ), INTENT( OUT ) :: xlocal
    INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ) :: map

    INTEGER( long_ ) :: ip, ip2
    INTEGER( ip_ ) :: i, j, k, rp1
    REAL( rp_ ) :: ri, ri2

    DO i = 1, blkn
      map( i ) = invp( lperm( i ) )
    END DO
    k = 1 + blkn - nd
    DO i = blkn + 1, blkm
      map( i ) = invp( rlist( k ) )
      k = k + 1
    END DO

!  copy eliminated variables into xlocal

    DO i = 1, nelim
      rp1 = map( i )
      xlocal( i ) = x( rp1 )
    END DO

!  perform the solve

    IF ( blkm > 10 .AND. nelim > 4 ) THEN

!  work with xlocal

      IF ( pos_def ) THEN
        CALL trsv( 'L', 'N', 'N', nelim, lcol, blkm, xlocal, 1_ip_ )
      ELSE
        CALL trsv( 'L', 'N', 'U', nelim, lcol, blkm, xlocal, 1_ip_ )
      END IF

      IF ( blkm-nelim > 0 ) THEN
        CALL gemv( 'N', blkm - nelim, nelim, - one, lcol( nelim + 1 ), blkm,   &
                   xlocal, 1_ip_, zero, xlocal( nelim + 1 ), 1_ip_ )

!  add contribution into x. Delays first

        DO i = nelim + 1, blkm
          rp1 = map( i )
          x( rp1 ) = x( rp1 ) + xlocal( i )
        END DO
      END IF
    ELSE
      DO i = 1, nelim - 1, 2
        ip = ( i - 1 ) * blkm
        ip2 = i * blkm
        IF ( pos_def ) xlocal( i ) = xlocal( i ) / lcol( ip + i )
        ri = xlocal( i )
        xlocal( i + 1 ) = xlocal( i + 1 ) - ri * lcol( ip + i + 1 )
        IF ( pos_def ) xlocal( i + 1 ) = xlocal( i + 1 ) / lcol( ip2 + i + 1 )
        ri2 = xlocal( i + 1 )
        DO j = i + 2, nelim
          xlocal( j ) = xlocal( j )                                            &
                          - ri * lcol( ip + j ) - ri2 * lcol( ip2 + j )
        END DO
        DO j = nelim + 1, blkm
          rp1 = map( j )
          x( rp1 ) = x( rp1 ) - ri * lcol( ip + j ) - ri2 * lcol( ip2 + j )
        END DO
      END DO
      IF ( MOD( nelim, 2 ) == 1 ) THEN
        ip = ( i - 1 ) * blkm
        ip2 = i * blkm
        IF ( pos_def ) xlocal( i ) = xlocal( i ) / lcol( ip + i )
        ri = xlocal( i )
        DO j = nelim+1, blkm
          rp1 = map( j )
          x( rp1 ) = x( rp1 ) - ri * lcol( ip + j )
        END DO
      END IF
    END IF

    ! Copy solution back from xlocal
    DO i = 1, nelim
      rp1 = map( i )
      x( rp1 ) = xlocal( i )
    END DO
    END SUBROUTINE solve_fwd_one

    SUBROUTINE solve_fwd_mult( pos_def, rlist, invp, nrhs, x, ldx, blkm, blkn, &
                               nelim, nd, lcol, lperm, xlocal, map )

!  forward substitution multiple rhs

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: pos_def
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: rlist
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    REAL( rp_ ), DIMENSION( ldx, * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: blkm
    INTEGER( ip_ ), INTENT( IN ) :: blkn
    INTEGER( ip_ ), INTENT( IN ) :: nelim
    INTEGER( ip_ ), INTENT( IN ) :: nd
    REAL( rp_ ), DIMENSION( * ), INTENT( IN ) :: lcol
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: lperm
    REAL( rp_ ), DIMENSION( blkm, * ), INTENT( OUT ) :: xlocal
    INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ) :: map

    INTEGER( long_ ) :: ip
    INTEGER( ip_ ) :: i, j, k, r, rp1
    REAL( rp_ ) :: ri

    DO i = 1, blkn
      map( i ) = invp( lperm( i ) )
    END DO
    k = 1 + blkn - nd
    DO i = blkn + 1, blkm
      map( i ) = invp( rlist( k ) )
      k = k + 1
    END DO

!  copy eliminated variables into xlocal

    DO r = 1, nrhs
      DO i = 1, nelim
        rp1 = map( i )
        xlocal( i,r ) = x( rp1, r )
      END DO
    END DO

!  perform the solve

    IF ( blkm > 10 .AND. nelim > 4 ) THEN

!  work with xlocal

      IF ( pos_def ) THEN
         CALL trsm( 'L', 'L', 'N', 'N', nelim, nrhs,                           &
                    one, lcol, blkm, xlocal, blkm )
      ELSE
         CALL trsm( 'L', 'L', 'N', 'U', nelim, nrhs,                           &
                    one, lcol, blkm, xlocal, blkm )
      END IF

      IF ( blkm > nelim ) THEN
        CALL gemm( 'N', 'N', blkm-nelim, nrhs, nelim, -one, &
             lcol( nelim+1 ), blkm, xlocal, blkm, zero, &
             xlocal( nelim+1,1 ), blkm )

!  add contribution into x

        DO r = 1, nrhs
           DO i = nelim + 1, blkn !  delays first
             rp1 = map( i )
             x( rp1, r ) = x( rp1, r ) + xlocal( i, r )
           END DO
           DO j = blkn + 1, blkm !  expected rows
             rp1 = map( j )
             x( rp1, r ) = x( rp1, r ) + xlocal( j, r )
           END DO
        END DO
      END IF
    ELSE
      DO r = 1, nrhs
        DO i = 1, nelim
          ip = ( i - 1 ) * blkm
          IF ( pos_def ) xlocal( i, r ) = xlocal( i, r ) / lcol( ip + i )
          ri = xlocal( i, r )
          DO j = i + 1, nelim
             xlocal( j, r ) = xlocal( j, r ) - ri * lcol( ip + j )
          END DO
          DO j = nelim + 1, blkm
             rp1 = map( j )
             x( rp1, r ) = x( rp1, r ) - ri * lcol( ip + j )
          END DO
        END DO
      END DO
    END IF

!  copy solution back from xlocal

    DO r = 1, nrhs
      DO i = 1, nelim
        rp1 = map( i )
        x( rp1, r ) = xlocal( i, r )
      END DO
    END DO
    END SUBROUTINE solve_fwd_mult

    SUBROUTINE solve_bwd_one( pos_def, job, rlist, invp, x, blkm, blkn, nelim, &
                              nd, lcol, d, lperm, xlocal, map )

!  Back substitution ( with diagonal solve ) single rhs

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: pos_def

!  job is used to indicate whether diagonal solve isrequired
!   job = 3 : backsubs only ( ( PL )^Tx = b )
!   job = 0 or 4 : diag and backsubs ( D( PL )^Tx = b )

    INTEGER( ip_ ), INTENT( IN ) :: job  
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: rlist
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: blkm
    INTEGER( ip_ ), INTENT( IN ) :: blkn
    INTEGER( ip_ ), INTENT( IN ) :: nelim
    INTEGER( ip_ ), INTENT( IN ) :: nd
    REAL( rp_ ), DIMENSION( * ), INTENT( IN ) :: lcol
    REAL( rp_ ), DIMENSION( 2 * nelim ) :: d
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: lperm
    REAL( rp_ ), DIMENSION( * ), INTENT( OUT ) :: xlocal
    INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ) :: map

    INTEGER( long_ ) :: ip
    INTEGER( ip_ ) :: i, j, k
    INTEGER( ip_ ) :: rp1, rp2

    DO i = 1, blkn
      map( i ) = invp( lperm( i ) )
    END DO
    k = 1 + blkn - nd
    DO i = blkn + 1, blkm
      map( i ) = invp( rlist( k ) )
      k = k + 1
    END DO

!  no diagonal solve. Copy eliminated variables into xlocal

    IF ( job == SSIDS_SOLVE_JOB_BWD .OR. pos_def ) THEN
      DO i = 1, nelim
        rp1 = map( i )
        xlocal( i ) = x( rp1 )
      END DO

!  copy eliminated vars into xlocal while performing diagonal solve

    ELSE
      j = 1
      DO WHILE ( j <= nelim )
        IF ( d( 2 * j ) /= 0 ) THEN !  2x2 pivot
          rp1 = map( j )
          rp2 = map( j + 1 )
          xlocal( j )   = d( 2 * j - 1 ) * x( rp1 ) + d( 2 * j )   * x( rp2 )
          xlocal( j + 1 ) = d( 2 * j )   * x( rp1 ) + d( 2 * j + 1 ) * x( rp2 )
          j = j + 2
        ELSE !  1x1 pivot
          IF ( d( 2 * j - 1 ) == 0.0_rp_ ) THEN !  zero pivot column
            xlocal( j ) = 0.0_rp_
          ELSE !  proper pivot
            rp1 = map( j )
            xlocal( j ) = x( rp1 ) * d( 2 * j - 1 )
          END IF
          j = j + 1
        END IF
      END DO
    END IF

!  perform the solve

    IF (  blkm > 10 .AND. nelim > 4 ) THEN
       DO i = nelim + 1, blkn !  delays
         rp1 = map( i )
         xlocal( i ) = x( rp1 )
       END DO
       DO j = blkn + 1, blkm !  expected rows
          rp1 = map( j )
          xlocal( j ) = x( rp1 )
       END DO
       IF ( blkm > nelim )                                                     &
          CALL gemv( 'T', blkm - nelim, nelim, - one, lcol( nelim + 1 ), blkm, &
                     xlocal( nelim + 1 ), 1_ip_, one, xlocal, 1_ip_ )
       END IF

       IF ( pos_def ) THEN
         CALL trsv( 'L', 'T', 'N', nelim, lcol, blkm, xlocal, 1_ip_ )
       ELSE
         CALL trsv( 'L', 'T', 'U', nelim, lcol, blkm, xlocal, 1_ip_ )
       END IF

!  copy solution back from xlocal

       DO i = 1, nelim
         rp1 = map( i )
         x( rp1 ) = xlocal( i )
       END DO

!  do update with indirect addressing

    ELSE
      DO i = 1, nelim
        ip = ( i - 1 ) * blkm
        DO j = nelim + 1, blkm
          rp1 = map( j )
          xlocal( i ) = xlocal( i ) - x( rp1 ) * lcol( ip + j )
        END DO
      END DO

!  solve with direct addressing

      IF ( pos_def ) THEN
        DO i = nelim, 1, - 1
          ip = ( i - 1 ) * blkm
          rp1 = map( i )
          xlocal( i ) = xlocal( i ) -                                          &
            SUM( xlocal( i + 1 : nelim ) * lcol( ip + i + 1 : ip + nelim ) )
          xlocal( i ) = xlocal( i ) / lcol( ip + i )
          x( rp1 ) = xlocal( i )
        END DO
      ELSE
        DO i = nelim, 1, - 1
          ip = ( i - 1 ) * blkm
          rp1 = map( i )
          xlocal( i ) = xlocal( i ) -                                          &
            SUM( xlocal( i + 1 : nelim ) * lcol( ip + i + 1 : ip + nelim ) )
          x( rp1 ) = xlocal( i )
        END DO
      END IF
    END IF
    END SUBROUTINE solve_bwd_one

    SUBROUTINE solve_bwd_mult( pos_def, job, rlist, invp, nrhs, x, ldx, blkm, &
                               blkn, nelim, nd, lcol, d, lperm, xlocal, map )

!  back substitution ( with diagonal solve ) multiple rhs

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: pos_def

!  job is used to indicate whether diagonal solve isrequired
!   job = 3 : backsubs only ( ( PL )^Tx = b )
!   job = 0 or 4 : diag and backsubs ( D( PL )^Tx = b )

    INTEGER( ip_ ), INTENT( IN ) :: job
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: rlist
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    REAL( rp_ ), DIMENSION( ldx, * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: blkm
    INTEGER( ip_ ), INTENT( IN ) :: blkn
    INTEGER( ip_ ), INTENT( IN ) :: nelim
    INTEGER( ip_ ), INTENT( IN ) :: nd
    REAL( rp_ ), DIMENSION( * ), INTENT( IN ) :: lcol
    REAL( rp_ ), DIMENSION( 2 * nelim ) :: d
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: lperm
    REAL( rp_ ), DIMENSION( blkm, * ), INTENT( OUT ) :: xlocal
    INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ) :: map

    INTEGER( long_ ) :: ip
    INTEGER( ip_ ) :: i, j, k, r, rp1, rp2

    DO i = 1, blkn
      map( i ) = invp( lperm( i ) )
    END DO
    k = 1 + blkn - nd
    DO i = blkn + 1, blkm
      map( i ) = invp( rlist( k ) )
      k = k + 1
    END DO

!  no diagonal solve. Copy eliminated variables into xlocal

    IF ( job == SSIDS_SOLVE_JOB_BWD .OR. pos_def ) THEN
      DO r = 1, nrhs
        DO i = 1, nelim
          rp1 = map( i )
          xlocal( i,r ) = x( rp1, r )
        END DO
      END DO

!  copy eliminated vars into xlocal while performing diagonal solve

    ELSE
      DO r = 1, nrhs
        j = 1
        DO WHILE ( j <= nelim )
          IF ( d( 2 * j ) /= 0 ) THEN !  2x2 pivot
            rp1 = map( j )
            rp2 = map( j + 1 )
            xlocal( j, r )                                                     &
              = d( 2 * j - 1 ) * x( rp1,r ) + d( 2 * j )  * x( rp2,r )
            xlocal( j + 1, r )                                                 &
               = d( 2 * j )   * x( rp1,r ) + d( 2 * j + 1 ) * x( rp2,r )
            j = j + 2
          ELSE !  1x1 pivot
            IF ( d( 2 * j - 1 ) == 0.0_rp_ ) THEN !  zero pivot column
               xlocal( j,r ) = 0.0_rp_
            ELSE !  proper pivot
               rp1 = map( j )
               xlocal( j, r ) = x( rp1, r ) * d( 2 * j - 1 )
            END IF
            j = j + 1
          END IF
        END DO
      END DO
    END IF

!  perform the solve

    IF ( blkm > 10 .AND. nelim > 4 ) THEN
      DO r = 1, nrhs
        DO i = nelim + 1, blkn !  delays
          rp1 = map( i )
          xlocal( i, r ) = x( rp1, r )
        END DO
        DO j = blkn + 1, blkm !  expected rows
          rp1 = map( j )
          xlocal( j, r ) = x( rp1, r )
        END DO
      END DO

      IF ( blkm > nelim )                                                      &
        CALL gemm( 'T', 'N', nelim, nrhs, blkm-nelim, - one,                   &
                   lcol( nelim + 1 ), blkm, xlocal( nelim + 1, 1 ),            &
                   blkm, one, xlocal, blkm )

      IF ( pos_def ) THEN
        CALL trsm( 'L', 'L', 'T', 'N', nelim, nrhs, one, lcol,                 &
                    blkm, xlocal, blkm )
      ELSE
        CALL trsm( 'L', 'L', 'T', 'U', nelim, nrhs, one, lcol,                 &
                    blkm, xlocal, blkm )
      END IF
      DO r = 1, nrhs
        DO i = 1, nelim
          rp1 = map( i )
          x( rp1, r ) = xlocal( i, r )
        END DO
      END DO

!  do update with indirect addressing

    ELSE
      DO r = 1, nrhs
        DO i = 1, nelim
          ip = ( i - 1 ) * blkm
          DO j = nelim + 1, blkm
            rp1 = map( j )
            xlocal( i, r ) = xlocal( i, r ) - x( rp1, r ) * lcol( ip + j )
          END DO
        END DO

!  solve with direct addressing

        IF ( pos_def ) THEN
          DO i = nelim, 1, - 1
            ip = ( i - 1 ) * blkm
            rp1 = map( i )
            xlocal( i, r ) = xlocal( i, r ) -                                 & 
             SUM( xlocal( i + 1 : nelim, r ) * lcol( ip + i + 1 : ip + nelim ) )
            xlocal( i, r ) = xlocal( i, r ) / lcol( ip + i )
            x( rp1, r ) = xlocal( i, r )
          END DO
        ELSE
          DO i = nelim, 1, - 1
            ip = ( i - 1 ) * blkm
            rp1 = map( i )
            xlocal( i, r ) = xlocal( i, r ) -                                  &
             SUM( xlocal( i + 1 : nelim, r ) * lcol( ip + i + 1 : ip + nelim ) )
            x( rp1, r ) = xlocal( i, r )
          END DO
        END IF
      END DO
    END IF
    END SUBROUTINE solve_bwd_mult

    SUBROUTINE solve_diag_one( invp, x, nelim, d, lperm )

!  diagonal solve one rhs

    IMPLICIT none
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: nelim
    REAL( rp_ ), DIMENSION( 2 * nelim ) :: d
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: lperm

    INTEGER( ip_ ) :: j, rp1, rp2
    REAL( rp_ ) :: temp

    j = 1
    DO WHILE ( j <= nelim )
      IF ( d( 2 * j ) /= 0 ) THEN ! 2x2 pivot
        rp1 = invp(  lperm( j )  )
        rp2 = invp(  lperm( j + 1 )  )
        temp   = d( 2 * j - 1 ) * x( rp1 ) + d( 2 * j )   * x( rp2 )
        x( rp2 ) = d( 2 * j )   * x( rp1 ) + d( 2 * j + 1 ) * x( rp2 )
        x( rp1 ) = temp
        j = j + 2
      ELSE ! 1x1 pivot
       rp1 = invp(  lperm( j )  )
       x( rp1 ) = x( rp1 ) * d( 2 * j - 1 )
       j = j + 1
      END IF
    END DO
    END SUBROUTINE solve_diag_one

    SUBROUTINE solve_diag_mult( invp, nrhs, x, ldx, nelim, d, lperm )

!  diagonal solve multiple rhs

    IMPLICIT none
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: invp
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    REAL( rp_ ), DIMENSION( ldx, * ), INTENT( INOUT ) :: x
    INTEGER( ip_ ), INTENT( IN ) :: nelim
    REAL( rp_ ), DIMENSION( 2 * nelim ) :: d
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: lperm

    INTEGER( ip_ ) :: j, r, rp1, rp2
    REAL( rp_ ) :: temp

    DO r = 1, nrhs
      j = 1
      DO WHILE ( j <= nelim )
        IF ( d( 2 * j ) /= 0 ) THEN ! 2x2 pivot
          rp1 = invp(  lperm( j )  )
          rp2 = invp(  lperm( j + 1 )  )
          temp = d( 2 * j - 1 ) * x( rp1, r ) + d( 2 * j ) * x( rp2, r )
          x( rp2, r ) = d( 2 * j ) * x( rp1, r ) + d( 2 * j + 1 ) * x( rp2, r )
          x( rp1, r ) = temp
          j = j + 2
        ELSE ! 1x1 pivot
          rp1 = invp(  lperm( j )  )
          x( rp1, r ) = x( rp1, r ) * d( 2 * j - 1 )
          j = j + 1
        END IF
      END DO
    END DO
    END SUBROUTINE solve_diag_mult

  END MODULE GALAHAD_SSIDS_gpu_cpu_solve_precision
