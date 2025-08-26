! THIS VERSION: GALAHAD 5.3 - 2025-08-25 AT 15:20 GMT

#include "ssids_procedures.h"

!  COPYRIGHT (c) 2010-2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

  MODULE GALAHAD_SSIDS_analyse_precision
    USE, INTRINSIC :: iso_c_binding
!$ USE :: omp_lib
    USE GALAHAD_KINDS_precision
    USE GALAHAD_HW, ONLY: HW_numa_region
    USE GALAHAD_SSIDS_akeep_precision, ONLY : SSIDS_akeep_type
    USE GALAHAD_SSIDS_cpu_subtree_precision, ONLY :                            &
      construct_cpu_symbolic_subtree
    USE GALAHAD_SSIDS_gpu_subtree_precision, ONLY :                            &
      construct_gpu_symbolic_subtree
    USE GALAHAD_SSIDS_types_precision
    USE GALAHAD_SSIDS_inform_precision, ONLY : SSIDS_inform_type
    IMPLICIT none

    PRIVATE
    PUBLIC :: analyse_phase,   & !  Calls core analyse & builds data strucutres
              check_order,     & !  Check order is a valid permutation
              expand_pattern,  & !  Specialised half->full matrix conversion
              expand_matrix      !  Specialised half->full matrix conversion

    INTERFACE print_atree
      MODULE PROCEDURE print_atree, print_atree_part
    END INTERFACE print_atree

!  extracted from SPRAL_CORE_ANALYSE
   
    INTEGER( ip_ ), parameter :: minsz_ms = 16 ! minimum size to use merge sort

  CONTAINS

    SUBROUTINE analyse_phase( n, ptr, row, ptr2, row2, order, invp,            &
                              akeep, control, inform  )

!   this routine requires the LOWER and UPPER triangular parts of A
!   to be held in CSC format using ptr2 and row2
!   AND lower triangular part held using ptr and row.
!
!   On exit from this routine, order is set to order
!   input to factorization.

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n !  order of system

!  col pointers (lower triangle)

    INTEGER( long_ ), INTENT( IN ) :: ptr( n + 1 ) 

!  row indices (lower triangle)

    INTEGER( ip_ ), INTENT( IN ) :: row( ptr( n + 1 ) - 1 )

!  col pointers (whole matrix)

    INTEGER( long_ ), INTENT( IN ) :: ptr2( n + 1 ) 

!  row indices (whole matrix)

    INTEGER( ip_ ), INTENT( IN ) :: row2( ptr2( n + 1 ) - 1 ) 

!   On exit, holds the pivot order to be used by factorization.

    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: order

!  Work array. Used to hold inverse of order but
!  is NOT set to inverse for the final order that is returned.

    INTEGER( ip_ ), DIMENSION( n ), INTENT( OUT ) :: invp
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform

    character( 50 )  :: context !  Procedure name ( used when printing ).
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_dest, exec_loc, level

    INTEGER( ip_ ) :: to_launch
    INTEGER( ip_ ) :: numa_region, device, thread_num
    INTEGER( ip_ ) :: nemin, flag
    INTEGER( ip_ ) :: blkm, blkn
    INTEGER( ip_ ) :: i, j
    INTEGER( ip_ ) :: nout, nout1 !  streams for errors and warnings
    INTEGER( long_ ) :: nz !  ptr( n + 1 ) - 1
    INTEGER( ip_ ) :: st

    context = 'ssids_analyse'
    nout = control%unit_error
    IF ( control%print_level < 0 ) nout = - 1
    nout1 = control%unit_warning
    IF ( control%print_level < 0 ) nout1 = - 1
    st = 0

!  check nemin and set to default if out of range

    nemin = control%nemin
    IF ( nemin < 1 ) nemin = nemin_default

!  perform basic analysis so we can decide upon subtrees we want to construct

    CALL basic_analyse( n, ptr2, row2, order, akeep%nnodes, akeep%sptr,        &
                        akeep%sparent, akeep%rptr,akeep%rlist, nemin, flag,    &
                        inform%stat, inform%num_factor, inform%num_flops )
    SELECT CASE( flag )
    CASE( 0 ) !  do nothing
    CASE( SSIDS_ERROR_ALLOCATION ) !  allocation error
      inform%flag = SSIDS_ERROR_ALLOCATION
      RETURN
    CASE( SSIDS_WARNING_ANALYSIS_SINGULAR ) !  zero row/column
      inform%flag = SSIDS_WARNING_ANALYSIS_SINGULAR
    CASE default !  should never reach here
      inform%flag = SSIDS_ERROR_UNKNOWN
    END SELECT

!  set invp to hold inverse of order

    DO i = 1, n
      invp( order( i ) ) = i
    END DO

!  any unused variables are at the end and so can set order for them

    DO j = akeep%sptr( akeep%nnodes + 1 ), n
      i = invp( j )
      order( i ) = 0
    END DO

!  build map from A to L in nptr, nlist

    nz = ptr( n + 1 ) - 1
    ALLOCATE( akeep%nptr( n + 1 ), akeep%nlist( 2, nz ), STAT = st )
    IF ( st /= 0 ) GO TO 100
    CALL build_map( n, ptr, row, order, invp, akeep%nnodes, akeep%sptr,        &
                    akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist, st )
    IF ( st /= 0 ) GO TO 100

!  sort out subtrees

    IF ( control%print_level >= 1 .AND. control%unit_diagnostics >= 0 ) THEN
      WRITE ( control%unit_diagnostics, *  ) "Input topology"
      DO i = 1, SIZE( akeep%topology )
        WRITE ( control%unit_diagnostics, *  )                                 &
             "Region ", i, " with ", akeep%topology( i )%nproc, " cores"
        IF ( SIZE( akeep%topology( i )%gpus )>0 )                              &
          WRITE ( control%unit_diagnostics, *  )                               &
             "---> gpus ", akeep%topology( i )%gpus
      END DO
    END IF
    CALL find_subtree_partition( akeep%nnodes, akeep%sptr, akeep%sparent,      &
                                 akeep%rptr, control, akeep%topology,          &
                                 akeep%nparts, akeep%part,                     &
                                 exec_loc, akeep%contrib_ptr,                  &
                                 akeep%contrib_idx, contrib_dest, inform, st )
    IF ( st /= 0 ) GO TO 100

!print * , "invp = ", akeep%invp
!print * , "sptr = ", akeep%sptr( 1 : akeep%nnodes + 1 )
!print * , "sparent = ", akeep%sparent
!print * , "Partition suggests ", akeep%nparts, " parts"
!print * , "akeep%part = ", akeep%part( 1 : akeep%nparts + 1 )
!print * , "exec_loc   = ", exec_loc( 1 : akeep%nparts )
!print * , "parents = ", akeep%sparent( akeep%part( 2 : akeep%nparts + 1 ) - 1 )
!print * , "contrib_ptr = ", akeep%contrib_ptr( 1:akeep%nparts + 1 )
!print * , "contrib_idx = ", akeep%contrib_idx( 1:akeep%nparts )
!print * , "contrib_dest = ", &
!    contrib_dest( 1 : akeep%contrib_ptr( akeep%nparts + 1 ) - 1 )

!  generate dot file for assembly tree

!   call print_atree( akeep%nnodes, akeep%sptr, akeep%sparent, akeep%rptr )
    IF ( .FALSE. ) & !  change .TRUE. to debug
      CALL print_atree( akeep%nnodes, akeep%sptr, akeep%sparent, akeep%rptr,   &
                        akeep%topology, akeep%nparts, akeep%part, exec_loc )

!  construct symbolic subtrees

    ALLOCATE( akeep%subtree( akeep%nparts ) )

!  split into NUMA regions for setup ( assume mem is first touch )

    to_launch = SIZE( akeep%topology )

!$omp parallel proc_bind( spread ) num_threads( to_launch ) default( shared ) &
!$omp    private( i, numa_region, device, thread_num )
    thread_num = 0
!$  thread_num = omp_get_thread_num( )
    numa_region = thread_num + 1
    DO i = 1, akeep%nparts

!  only initialize subtree if this is the correct region: note that
!  an "all region" subtree with location -1 is initialised by region 0

      IF ( exec_loc( i ) == - 1 ) THEN
        IF ( numa_region /= 1 ) CYCLE
        device = 0
      ELSE IF ( ( MOD( ( exec_loc( i ) - 1 ),                                  &
                  SIZE( akeep%topology ) ) + 1 ) /= numa_region ) THEN
        CYCLE
      ELSE
        device = ( exec_loc( i ) - 1 ) / SIZE( akeep%topology )
      END IF
      akeep%subtree( i )%exec_loc = exec_loc( i )

!  CPU

      IF ( device == 0 ) THEN

!print  * , numa_region, "init cpu subtree ", i, akeep%part( i ), &
!    akeep%part( i + 1 ) - 1

        akeep%subtree( i )%ptr => construct_cpu_symbolic_subtree( akeep%n,     &
          akeep%part( i ), akeep%part( i + 1 ), akeep%sptr, akeep%sparent,     &
          akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist,                    &
          contrib_dest( akeep%contrib_ptr( i ) :                               &
                        akeep%contrib_ptr( i + 1 ) - 1 ), control )
!  GPU

      ELSE
          device = akeep%topology( numa_region )%gpus( device )

!print  * , numa_region, "init gpu subtree ", i, akeep%part( i ), &
!    akeep%part( i + 1 ) - 1, "device", device

        akeep%subtree( i )%ptr => construct_gpu_symbolic_subtree( device,      &
          akeep%n, akeep%part( i ), akeep%part( i + 1 ), akeep%sptr,           &
          akeep%sparent, akeep%rptr, akeep%rlist, akeep%nptr,                  &
          akeep%nlist, control )
      END IF
    END DO
!$omp end parallel

!  info

    ALLOCATE( level( akeep%nnodes + 1 ), STAT = st )
    IF ( st /= 0 ) GO TO 100
    level( akeep%nnodes + 1 ) = 0
    inform%maxfront = 0
    inform%maxdepth = 0
    DO i = akeep%nnodes, 1, - 1
      blkn = akeep%sptr( i + 1 ) - akeep%sptr( i )
      blkm = int( akeep%rptr( i + 1 ) - akeep%rptr( i ) )
      level( i ) = level( akeep%sparent( i ) ) + 1
      inform%maxfront = max( inform%maxfront, blkm )
      inform%maxsupernode = max( inform%maxsupernode, blkn )
      inform%maxdepth = max( inform%maxdepth, level( i ) )
    END DO
    DEALLOCATE( level, STAT = st )
    inform%matrix_rank = akeep%sptr( akeep%nnodes + 1 ) - 1
    inform%num_sup = akeep%nnodes

!  store copy of inform data in akeep

    akeep%inform = inform
    RETURN

100 CONTINUE
    inform%stat = st
    IF ( inform%stat /= 0 ) THEN
      inform%flag = SSIDS_ERROR_ALLOCATION
    END IF
    RETURN
    END SUBROUTINE analyse_phase

    SUBROUTINE check_order( n, order, invp, control, inform )

!   this routine requires the LOWER triangular part of A to be held in CSC 
!   format. The user has supplied a pivot order and this routine checks it 
!   is OK and returns an error if not. Also sets perm, invp.

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n !  order of system
    INTEGER( ip_ ), INTENT( INOUT ) :: order( : )

!  if i is used to index a variable, |order( i )| must
!  hold its position in the pivot sequence. If 1x1 pivot i required,
!  the user must set order( i )>0. If a 2x2 pivot involving variables
!  i and j is required, the user must set
!  order( i )<0, order( j )<0 and |order( j )| = |order( i )| + 1.
!  If i is not used to index a variable, order( i ) must be set to zero.
!  !!!!  In this version, signs are reset to positive value

    INTEGER( ip_ ), INTENT( OUT ) :: invp( n )

!  Used to check order and then holds inverse of perm.

    TYPE( ssids_control_type ), INTENT( IN ) :: control
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

    character( 50 )  :: context !  Procedure name ( used when printing ).

    INTEGER( ip_ ) :: i, j
    INTEGER( ip_ ) :: nout  !  stream for error messages

    context = 'ssids_analyse'
    nout = control%unit_error
    IF ( control%print_level < 0 ) nout = - 1

!  order is too short

    IF ( SIZE( order ) < n ) THEN
      inform%flag = SSIDS_ERROR_ORDER
      RETURN
    END IF

!  initialise

    invp( : ) = 0
    DO i = 1, n
      order( i ) = ABS( order( i ) )
    END DO

!  check user-supplied order and copy the absolute values to invp.
!  Also add up number of variables that are not used ( null rows )

    DO i = 1, n
      j = order( i )
      IF ( j <= 0 .OR. j > n ) EXIT !  Out of range entry
      IF ( invp( j ) /= 0 ) EXIT !  Duplicate found
      invp( j ) = i
    END DO
    IF ( i - 1 /= n ) THEN
      inform%flag = SSIDS_ERROR_ORDER
      RETURN
    END IF
    END SUBROUTINE check_order

    SUBROUTINE expand_pattern( n, nz, ptr, row, aptr, arow )

!  Given lower triangular part of A held in row and ptr, expand to
!  upper and lower triangular parts (pattern only). No checks.
!
!  Note: we do not use half_to_full here to expand A since, if we did, we would
!  need an extra copy of the lower triangle into the full structure before
!  calling half_to_full
!
    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n !  order of system
    INTEGER( long_ ), INTENT( IN ) :: nz
    INTEGER( long_ ), INTENT( IN ) :: ptr( n + 1 )
    INTEGER( ip_ ), INTENT( IN ) :: row( nz )
    INTEGER( long_ ), INTENT( OUT ) :: aptr( n + 1 )
    INTEGER( ip_ ), INTENT( OUT ) :: arow( 2 * nz )

    INTEGER( ip_ ) :: i,j
    INTEGER( long_ ) :: kk

!  set aptr( j ) to hold no. nonzeros in column j

    aptr( : ) = 0
    DO j = 1, n
      DO kk = ptr( j ), ptr( j + 1 ) - 1
        i = row( kk )
        aptr( i ) = aptr( i ) + 1
        IF ( j == i ) CYCLE
        aptr( j ) = aptr( j ) + 1
      END DO
    END DO

!  set aptr( j ) to point to where row indices will end in arow

    DO j = 2, n
      aptr( j ) = aptr( j - 1 ) + aptr( j )
    END DO
    aptr( n + 1 ) = aptr( n ) + 1

!  fill arow and aptr

    DO j = 1, n
      DO kk = ptr( j ), ptr( j + 1 ) - 1
        i = row( kk )
        arow( aptr( i ) ) = j
        aptr( i ) = aptr( i ) - 1
        IF ( j == i ) CYCLE
        arow( aptr( j ) ) = i
        aptr( j ) = aptr( j ) - 1
      END DO
    END DO
    DO j = 1,n
      aptr( j ) = aptr( j ) + 1
    END DO
    END SUBROUTINE expand_pattern

    SUBROUTINE expand_matrix( n, nz, ptr, row, val, aptr, arow, aval )

!  given lower triangular part of A held in row, val and ptr, expand to
!  upper and lower triangular parts

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN )   :: n !  order of system
    INTEGER( long_ ), INTENT( IN )   :: nz
    INTEGER( long_ ), INTENT( IN )   :: ptr( n + 1 )
    INTEGER( ip_ ), INTENT( IN )   :: row( nz )
    REAL( rp_ ), INTENT( IN )  :: val( nz )
    INTEGER( long_ ), INTENT( OUT )  :: aptr( n + 1 )
    INTEGER( ip_ ), INTENT( OUT )  :: arow( 2 * nz )
    REAL( rp_ ), INTENT( OUT ) :: aval( 2 * nz )

    INTEGER( ip_ ) :: i, j
    INTEGER( long_ ) :: kk, ipos, jpos
    REAL( rp_ ) :: atemp

!  set aptr( j ) to hold no. nonzeros in column j

    aptr( : ) = 0
    DO j = 1, n
      DO kk = ptr( j ), ptr( j + 1 ) - 1
        i = row( kk )
        aptr( i ) = aptr( i ) + 1
        IF ( j == i ) CYCLE
        aptr( j ) = aptr( j ) + 1
      END DO
    END DO

!  set aptr( j ) to point to where row indices will end in arow

    DO j = 2, n
      aptr( j ) = aptr( j - 1 ) + aptr( j )
    END DO
    aptr( n + 1 ) = aptr( n ) + 1

!  fill arow, aval and aptr

    DO j = 1, n
      DO kk = ptr( j ), ptr( j + 1 ) - 1
        i = row( kk )
        atemp = val( kk )
        ipos = aptr( i )
        arow( ipos ) = j
        aval( ipos ) = atemp
        aptr( i ) = ipos - 1
        IF ( j == i ) CYCLE
        jpos = aptr( j )
        arow( jpos ) = i
        aval( jpos ) = atemp
        aptr( j ) = jpos - 1
      END DO
    END DO
    DO j = 1,n
      aptr( j ) = aptr( j ) + 1
    END DO
    END SUBROUTINE expand_matrix

    FUNCTION compute_flops( nnodes, sptr, rptr, node )

!   compute flops for processing a node
!    akeep Information generated in analysis phase by SSIDS
!    node Node

    IMPLICIT none

    INTEGER( long_ ) :: compute_flops !  return value
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr
    INTEGER( ip_ ), INTENT( IN ) :: node !  node index

    INTEGER( ip_ ) :: n, m !  node sizes
    INTEGER( long_ ) :: jj

    compute_flops = 0

    m = INT( rptr( node + 1 ) - rptr( node ) )
    n = sptr( node + 1 ) - sptr( node )
    DO jj = m - n + 1, m
      compute_flops = compute_flops + jj ** 2
    END DO

    END FUNCTION compute_flops

    SUBROUTINE find_subtree_partition( nnodes, sptr, sparent, rptr, control,   &
                                       topology, nparts, part, exec_loc,       &
                                       contrib_ptr, contrib_idx,               &
                                       contrib_dest, inform, st  )

!   partition an elimination tree for execution on different NUMA regions
!    and GPUs.
!
!   Start with a single tree, and proceed top DOwn splitting the largest subtree
!   (in terms of total flops) until we have a sufficient number of independent
!   subtrees. A sufficient number is such that subtrees can be assigned to NUMA
!   regions and GPUs with a load balance no worse than max_load_inbalance.
!   Load balance is calculated as the maximum value over all regions/GPUs of:
!   \f[ \frac{ n x_i / \alpha_i } { \sum_j ( x_j/\alpha_j ) } \f]
!   Where \f$ \alpha_i \f$ is the performance coefficient of region/GPU i,
!   \f$ x_i \f$ is the number of flops assigned to region/GPU i and \f$ n \f$ is
!   the total number of regions. \f$ \alpha_i \f$ should be proportional to the
!   speed of the region/GPU ( i.e. if GPU is twice as fast as CPU, set alpha for
!   CPU to 1.0 and alpha for GPU to 2.0 ).
!
!   If the original number of flops is greater than min_gpu_work and the
!   performance coefficient of a GPU is greater than the combined coefficients
!   of the CPU, then subtrees will not be split to become smaller than
!   min_gpu_work until all GPUs are filled.
!
!   If the balance criterion cannot be satisfied after we have split into
!   2 * ( total regions/GPUs ), we just use the best obtained value.
!
!   GPUs may only handle leaf subtrees, so the top nodes are assigned to the
!   full set of CPUs.
!
!   Parts are returned as contigous ranges of nodes. Part i consists of nodes
!   part( i ):part( i + 1 )-1
!
!    nnodes Total number of nodes
!    sptr Supernode pointers. Supernode i consists of nodes
!    sptr( i ):sptr( i + 1 )-1.
!    sparent Supernode parent array. Supernode i has parent sparent( i ).
!    rptr Row pointers. Supernode i has rows rlist( rptr( i ):rptr( i + 1 )-1 ).
!    topology Machine topology to partition for.
!    min_gpu_work Minimum flops for a GPU execution to be worthwhile.
!    max_load_inbalance Number greater than 1.0 representing maximum
!    permissible load inbalance.
!    gpu_perf_coeff The value of \f$ \alpha_i \f$ used for all GPUs,
!    assuming that used for all NUMA region CPUs is 1.0.
!    nparts Number of parts found.
!    parts List of part ranges. Part i consists of supernodes
!    part( i ):part( i + 1 )-1.
!    exec_loc Execution location. Part i should be run on partition
!    mod( ( exec_loc( i ) - 1 ), size( topology ) ) + 1.
!    It should be run on the CPUs if
!    exec_loc( i ) <= size( topology ),
!    otherwise it should be run on GPU number
!    ( exec_loc( i ) - 1 )/size( topology ).
!    contrib_ptr Contribution pointer. Part i has contribution from
!    subtrees contrib_idx( contrib_ptr( i ):contrib_ptr( i + 1 )-1 ).
!    contrib_idx List of contributing subtrees, see contrib_ptr.
!    contrib_dest Node to which each subtree listed in contrib_idx( : )
!    contributes.
!    st Allocation status parameter. If non-zero an allocation error
!          occurred.

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( ip_ ), DIMENSION( nnodes ), INTENT( IN ) :: sparent
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr
    TYPE( ssids_control_type ), INTENT( IN ) :: control
    TYPE( HW_numa_region ), DIMENSION( : ), INTENT( IN ) :: topology
    INTEGER( ip_ ), INTENT( OUT ) :: nparts
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( INOUT ) :: part
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: exec_loc
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( INOUT ) :: contrib_ptr
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( INOUT ) :: contrib_idx
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: contrib_dest
    TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: i, j, k
    INTEGER( ip_ ) :: node
    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: flops
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: size_order
    LOGICAL, DIMENSION( : ), ALLOCATABLE :: is_child
    REAL :: load_balance, best_load_balance
    INTEGER( ip_ ) :: nregion, ngpu
    LOGICAL :: has_parent

!  count flops below each node

    ALLOCATE( flops( nnodes + 1 ), STAT = st )
    IF ( st /= 0 ) RETURN
    flops( : ) = 0
    DO node = 1, nnodes
      flops( node ) = flops( node ) + compute_flops( nnodes, sptr, rptr, node )
      j = sparent( node )
      flops( j ) = flops( j ) + flops( node )
    END DO

!  initialize partition to be all children of virtual root

    ALLOCATE( part( nnodes + 1 ), size_order( nnodes ), exec_loc( nnodes ),    &
              is_child( nnodes ), STAT = st )
    IF ( st /= 0 ) RETURN
    nparts = 0
    part( 1 ) = 1
    DO i = 1, nnodes
      IF ( sparent( i ) > nnodes ) THEN
        nparts = nparts + 1
        part( nparts + 1 ) = i + 1
        is_child( nparts ) = .TRUE. !  all subtrees are intially child subtrees
      END IF
    END DO
    CALL create_size_order( nparts, part, flops, size_order )

!  calculate number of regions/gpus

    nregion = SIZE( topology )
    ngpu = 0
    DO i = 1, SIZE( topology )
      ngpu = ngpu + SIZE( topology( i )%gpus )
    END DO

!  keep splitting until we meet balance criterion

    best_load_balance = HUGE( best_load_balance )

!  check load balance criterion

    DO i = 1, 2 * ( nregion + ngpu )
      load_balance = calc_exec_alloc( nparts, part, size_order, is_child,      &
                                      flops, topology, control%min_gpu_work,   &
                                      control%gpu_perf_coeff, exec_loc, st )
      IF ( st /= 0 ) RETURN
      best_load_balance = min( load_balance, best_load_balance )
      IF ( load_balance < control%max_load_inbalance ) EXIT !  allocation is ok

!  split tree further

      CALL split_tree( nparts, part, size_order, is_child, sparent, flops,     &
                       ngpu, control%min_gpu_work, st )
      IF ( st /= 0 ) RETURN
    END DO

    IF ( control%print_level >= 1 .AND. control%unit_diagnostics >= 0 )        &
      WRITE ( control%unit_diagnostics, *  )                                   &
        "[find_subtree_partition] load_balance = ", best_load_balance

!  consolidate adjacent non-children nodes into same part and regn exec_alloc
!print  * 
!print  * , "pre merge", part( 1 : nparts + 1 )
!print  * , "exec_loc ", exec_loc( 1 : nparts )

    j = 1
    DO i = 2, nparts
      part( j + 1 ) = part( i )

!  we cannot merge j and i

      IF ( is_child( i ) .OR. is_child( j ) ) THEN
        j = j + 1
        is_child( j ) = is_child( i )
      END IF
    END DO
    part( j + 1 ) = part( nparts + 1 )
    nparts = j

!print  * , "post merge", part( 1 : nparts + 1 )

    CALL create_size_order( nparts, part, flops, size_order )
    load_balance = calc_exec_alloc( nparts, part, size_order, is_child,        &
                                    flops, topology, control%min_gpu_work,     &
                                    control%gpu_perf_coeff, exec_loc, st )
    IF ( st /= 0 ) RETURN
!print  * , "exec_loc ", exec_loc( 1 : nparts )

!  merge adjacent subtrees that are executing on the same node so long as
!  there is no more than one contribution to a parent subtree

    j = 1
    k = sparent( part( j + 1 ) - 1 )
    has_parent = ( k <= nnodes )
    DO i = 2, nparts
      part( j + 1 ) = part( i )
      exec_loc( j + 1 ) = exec_loc( i )
      k = sparent( part( i + 1 ) - 1 )

!  we cannot merge j and i

      IF ( exec_loc( i ) /= exec_loc( j ) .OR.                                &
          ( has_parent .AND. k <= nnodes ) ) THEN
        j = j + 1
        has_parent = .FALSE.
      END IF
      has_parent = has_parent.OR.( k<=nnodes )
    END DO
    part( j + 1 ) = part( nparts + 1 )
    nparts = j

!  discover contribution blocks that are input to each part

    ALLOCATE( contrib_ptr( nparts + 3 ), contrib_idx( nparts ),                &
              contrib_dest( nparts ), STAT = st )
    IF ( st /= 0 ) RETURN

!  count contributions at offset + 2

    contrib_ptr( 3 : nparts + 3 ) = 0
    DO i = 1, nparts - 1 !  by definition, last part has no parent
      j = sparent( part( i + 1 ) - 1 ) !  node index of parent
      IF ( j > nnodes ) CYCLE !  part is a root
      k = i + 1 !  part index of j
      DO WHILE( j >= part( k + 1 ) )
        k = k + 1
      END DO
      contrib_ptr( k + 2 ) = contrib_ptr( k + 2 ) + 1
    END DO

!  discover if contrib_ptr starts at offset  + 1

    contrib_ptr( 1 : 2 ) = 1
    DO i = 1, nparts
       contrib_ptr( i + 2 ) = contrib_ptr( i + 1 ) + contrib_ptr( i + 2 )
    END DO

!  drop sources into list

    DO i = 1, nparts - 1 !  by defn, last part has no parent
      j = sparent( part( i + 1 ) - 1 ) !  node index of parent

!  part is a root

      IF ( j > nnodes ) THEN
        contrib_idx( i ) = nparts + 1
        CYCLE
      END IF
      k = i + 1 !  part index of j
      DO while ( j >= part( k + 1 ) )
        k = k + 1
      END DO
      contrib_idx( i ) = contrib_ptr( k + 1 )
      contrib_dest( contrib_idx( i ) ) = j
      contrib_ptr( k + 1 ) = contrib_ptr( k + 1 ) + 1
    END DO
    contrib_idx( nparts ) = nparts + 1 !  last part must be a root

!  fill out inform

    inform%nparts = nparts
    inform%gpu_flops = 0
    DO i = 1, nparts
       IF ( exec_loc( i ) > SIZE( topology ) )                                 &
         inform%gpu_flops = inform%gpu_flops + flops( part( i + 1 ) - 1 )
    END DO
    inform%cpu_flops = flops( nnodes + 1 ) - inform%gpu_flops
    END SUBROUTINE find_subtree_partition

    REAL FUNCTION calc_exec_alloc( nparts, part, size_order, is_child, flops,  &
                                   topology, min_gpu_work, gpu_perf_coeff,     &
                                   exec_loc, st )

!   allocate execution of subtrees to resources and calculate load balance
!
!   Given the partition supplied, uses a greedy algorithm to assign subtrees to
!   resources specified by topology and then returns the resulting load balance
!   as
!   \f[ \frac{\max_i(  n x_i / \alpha_i  )} { \sum_j ( x_j/\alpha_j ) } \f]
!   Where \f$ \alpha_i \f$ is the performance coefficient of region/GPU i,
!   \f$ x_i \f$ is the number of flops assigned to region/GPU i and \f$ n \f$ is
!   the total number of regions. \f$ \alpha_i \f$ should be proportional to the
!   speed of the region/GPU ( i.e. if GPU is twice as fast as CPU, set alpha for
!   CPU to 1.0 and alpha for GPU to 2.0 ).
!
!   Work is only assigned to GPUs if the subtree has at least min_gpu_work flops
!
!   None-child subtrees are ignored ( they will be executed using all available
!   resources ). They are recorded with exec_loc -1.
!
!    nparts Number of parts.
!    parts List of part ranges. Part i consists of supernodes
!    part( i ):part( i + 1 )-1.
!    size_order Lists parts in decreasing order of flops.
!    i.e. size_order( 1 ) is the largest part.
!    is_child True if subtree is a child subtree ( has no contributions
!    from other subtrees ).
!    flops Number of floating points in subtree rooted at each node.
!    topology Machine topology to allocate execution for.
!    min_gpu_work Minimum work before allocation to GPU is useful.
!    gpu_perf_coeff The value of \f$ \alpha_i \f$ used for all GPUs,
!    assuming that used for all NUMA region CPUs is 1.0.
!    exec_loc Execution location. Part i should be run on partition
!    mod( ( exec_loc( i ) - 1 ), size( topology ) ) + 1.
!    It should be run on the CPUs if
!    exec_loc( i ) <= size( topology ),
!    otherwise it should be run on GPU number
!    ( exec_loc( i ) - 1 )/size( topology ).
!    st Allocation status parameter. If non-zero an allocation error
!          occurred.
!    Load balance value as detailed in subroutine description.
!    see also find_subtree_partition( )
!   FIXME: Consider case when gpu_perf_coeff > 2.0 ???
!         ( Round robin may not be correct thing )

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nparts
    INTEGER( ip_ ), DIMENSION( nparts + 1 ), INTENT( IN ) :: part
    INTEGER( ip_ ), DIMENSION( nparts ), INTENT( IN ) :: size_order
    LOGICAL, DIMENSION( nparts ), INTENT( IN ) :: is_child
    INTEGER( long_ ), DIMENSION( * ), INTENT( IN ) :: flops
    TYPE( HW_numa_region ), DIMENSION( : ), INTENT( IN ) :: topology
    INTEGER( long_ ), INTENT( IN ) :: min_gpu_work
    REAL, INTENT( IN ) :: gpu_perf_coeff
    INTEGER( ip_ ), DIMENSION( nparts ), INTENT( OUT ) :: exec_loc
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: i, p, nregion, ngpu, max_gpu, next
    INTEGER( long_ ) :: pflops

!  list resources in order of decreasing power

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: map 
    REAL, DIMENSION( : ), ALLOCATABLE :: load_balance
    REAL :: total_balance

!  initialise in case of an error return

    calc_exec_alloc = huge( calc_exec_alloc )

!  create resource map

    nregion = SIZE( topology )
    ngpu = 0
    max_gpu = 0
    DO i = 1, SIZE( topology )
      ngpu = ngpu + SIZE( topology( i )%gpus )
      max_gpu = max( max_gpu, size( topology( i )%gpus ) )
    END DO
    ALLOCATE( map( nregion + ngpu ), STAT = st )
    IF ( st /= 0 ) RETURN
 
!  GPUs are more powerful than CPUs

    IF ( gpu_perf_coeff > 1.0 ) THEN
      next = 1
      DO i = 1, SIZE( topology )
        DO p = 1, SIZE( topology( i )%gpus )
           map( next ) = p * nregion + i
           next = next + 1
        END DO
      END DO
      DO i = 1, SIZE( topology )
        map( next ) = i
        next = next + 1
      END DO

!  CPUs are more powerful than GPUs

    ELSE
      next = 1
      DO i = 1, SIZE( topology )
        map( next ) = i
        next = next + 1
      END DO
      DO i = 1, SIZE( topology )
        DO p = 1, SIZE( topology( i )%gpus )
          map( next ) = p * nregion + i
          next = next + 1
        END DO
      END DO
    END IF

!  simple round robin allocation in decreasing size order.

    next = 1
    DO i = 1, nparts
      p = SIZE_order( i )

!  not a child subtree

      IF ( .NOT. is_child( p ) ) THEN
        exec_loc( p ) = - 1
        CYCLE
      END IF

!  avoid GPUs

      pflops = flops( part( p + 1 ) - 1 )
      IF ( pflops < min_gpu_work ) THEN
        DO while ( map( next ) > nregion )
          next = next + 1
          IF ( next > SIZE( map ) ) next = 1
        END DO
      END IF
      exec_loc( p ) = map( next )
      next = next + 1
      IF ( next > SIZE( map ) ) next = 1
    END DO

!  calculate load inbalance

    ALLOCATE( load_balance( nregion * ( 1 + max_gpu ) ), STAT = st )
    IF ( st /= 0 ) RETURN
    load_balance( : ) = 0.0
    total_balance = 0.0

!  sum total

    DO p = 1, nparts
      IF ( exec_loc( p ) ==  - 1 ) CYCLE !  not a child subtree
      pflops = flops( part( p + 1 ) - 1 )

!  GPU

      IF ( exec_loc( p ) > nregion ) THEN
        load_balance( exec_loc( p ) )                                          &
           = load_balance( exec_loc( p ) ) +  REAL( pflops ) / gpu_perf_coeff
        total_balance = total_balance + REAL( pflops ) / gpu_perf_coeff

  !  CPU

      ELSE
        load_balance( exec_loc( p ) )                                          &
           = load_balance( exec_loc( p ) ) + REAL( pflops )
        total_balance = total_balance + REAL( pflops )
      END IF
    END DO

!  calculate n * max( x_i / a_i ) / sum( x_j / a_j )

    calc_exec_alloc                                                            &
      = REAL( nregion + ngpu ) * maxval( load_balance( : ) ) / total_balance
    END FUNCTION calc_exec_alloc

    SUBROUTINE split_tree( nparts, part, size_order, is_child, sparent, flops, &
                           ngpu, min_gpu_work, st )

!   split tree into an additional part as required by find_subtree_partition( ).
!
!   split largest partition into two parts, unless DOing so would reduce the
!   number of subtrees with at least min_gpu_work below ngpu.
!
!   Note: We require all input parts to have a single root.
!
!    nparts Number of parts: normally increased by one on return.
!    part Part i consists of nodes part( i ):part( i + 1 ).
!    size_order Lists parts in decreasing order of flops.
!    i.e. size_order( 1 ) is the largest part.
!    is_child True if subtree is a child subtree ( has no contributions
!    from other subtrees ).
!    sparent Supernode parent array. Supernode i has parent sparent( i ).
!    flops Number of floating points in subtree rooted at each node.
!    ngpu Number of gpus.
!    min_gpu_work Minimum worthwhile work to give to GPU.
!    st Allocation status parameter. If non-zero an allocation error
!          occurred.
!    see also find_subtree_partition( )

    IMPLICIT none
    INTEGER( ip_ ), INTENT( INOUT ) :: nparts
    INTEGER( ip_ ), DIMENSION( * ), INTENT( INOUT ) :: part
    INTEGER( ip_ ), DIMENSION( * ), INTENT( INOUT ) :: size_order
    LOGICAL, DIMENSION( * ), INTENT( INOUT ) :: is_child
    INTEGER( ip_ ), DIMENSION( * ), INTENT( IN ) :: sparent
    INTEGER( long_ ), DIMENSION( * ), INTENT( IN ) :: flops
    INTEGER( ip_ ), INTENT( IN ) :: ngpu
    INTEGER( long_ ), INTENT( IN ) :: min_gpu_work
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: i, p, nchild, nbig, root, to_split, old_nparts
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: children, temp

!  look for all children of root in biggest child part

    nchild = 0
    ALLOCATE( children( 10 ), STAT = st ) !  we will resize if necessary
    IF ( st /= 0 ) RETURN

!  find biggest child subtree

    to_split = 1
    DO WHILE( .NOT. is_child( size_order( to_split ) ) )
      to_split = to_split + 1
    END DO
    to_split = size_order( to_split )

!  find all children of root

    root = part( to_split + 1 ) - 1
    DO i = part( to_split ), root - 1
      IF ( sparent( i ) == root ) THEN
        nchild = nchild + 1

!  increase size of children(:)

        IF ( nchild > SIZE( children ) ) THEN
          ALLOCATE( temp( 2 * SIZE( children ) ), STAT = st )
          IF ( st /= 0 ) RETURN
          temp( 1 : SIZE( children ) ) = children( : )
          DEALLOCATE( children )
          CALL move_alloc( temp, children )
        END IF
        children( nchild ) = i
      END IF
    END DO

!  check we can split safely

    IF ( nchild == 0 ) RETURN !  singleton node, can't split
    nbig = 0 !  number of new parts > min_gpu_work
    DO i = to_split + 1, nparts
      p = size_order( i )
      IF ( .NOT. is_child( p ) ) CYCLE !  non-children can't go on GPUs
      root = part( p + 1 ) - 1
      IF ( flops( root ) < min_gpu_work ) EXIT
      nbig = nbig + 1
    END DO

!  original partition met min_gpu_work criterion

    IF ( ( nbig + 1 ) >= ngpu ) THEN
      DO i = 1, nchild
        IF ( flops( children( i ) ) >= min_gpu_work ) nbig = nbig + 1
      END DO
      IF ( nbig < ngpu ) RETURN !  new partition fails min_gpu_work criterion
    END IF

!  Can safely split, so DO so. As part to_split was contigous, when
!  split the new parts fall into the same region. Thus, we first push any
!  later regions back to make room, then add the new parts.

    part( to_split + nchild + 1 : nparts + nchild + 1 )                        &
      = part( to_split + 1 : nparts + 1 )
    is_child( to_split + nchild + 1 : nparts + nchild )                        &
      = is_child( to_split + 1 : nparts )

!  new part corresponding to child i * ends * at part( to_split + i ) - 1

    DO i = 1, nchild
      part( to_split + i ) = children( i ) + 1
    END DO
    is_child( to_split:to_split + nchild - 1 ) = .TRUE.
    is_child( to_split + nchild ) = .FALSE. !  Newly created non-parent subtree
    old_nparts = nparts
    nparts = old_nparts + nchild

!  finally, recreate size_order array

    CALL create_size_order( nparts, part, flops, size_order )
    END SUBROUTINE split_tree

    SUBROUTINE create_size_order( nparts, part, flops, size_order )

!  determine order of subtrees based on size
!
!  note sorting algorithm could be improved if this becomes a bottleneck
!
!   nparts - number of parts: normally increased by one on return
!   part - part i consists of nodes part( i ):part( i + 1 )
!   flops - number of floating points in subtree rooted at each node
!   size_order - lists parts in decreasing order of flops, i.e. 
!                size_order( 1 ) is the largest part

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nparts
    INTEGER( ip_ ), DIMENSION( nparts + 1 ), INTENT( IN ) :: part
    INTEGER( long_ ), DIMENSION( * ), INTENT( IN ) :: flops
    INTEGER( ip_ ), DIMENSION( nparts ), INTENT( OUT ) :: size_order

    INTEGER( ip_ ) :: i, j
    INTEGER( long_ ) :: iflops

    DO i = 1, nparts

!  we assume parts 1:i-1 are in order and aim to insert part i

      iflops = flops( part( i + 1 ) - 1 )
      DO j = 1, i - 1

! exit if node i belongs in position j

        IF ( iflops > flops( part( j + 1 ) - 1 ) ) EXIT 
      END DO
      size_order( j + 1 : i ) = size_order( j : i - 1 )
      size_order( j ) = i
    END DO
    END SUBROUTINE create_size_order

    SUBROUTINE print_atree( nnodes, sptr, sparent, rptr )

!   prints assembly tree

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( ip_ ), DIMENSION( nnodes ), INTENT( IN ) :: sparent
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr

    INTEGER( ip_ ) :: node
    INTEGER( ip_ ) :: n, m !  node sizes
    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: flops
    INTEGER( ip_ ) :: j
    REAL :: tot_weight, weight

!  count flops below each node

    ALLOCATE( flops( nnodes + 1 ) )
    flops( : ) = 0
    DO node = 1, nnodes
      flops( node ) = flops( node ) + compute_flops( nnodes, sptr, rptr, node )
      j = sparent( node )
      IF( j > 0 ) flops( j ) = flops( j ) + flops( node )
!   print  * , "Node ", node, "parent", j, " flops ", flops( node )
    END DO
    tot_weight = REAL( flops( nnodes ) )

    OPEN( 2, file = "atree.dot" )
    WRITE( 2, '( "graph atree {" )' )
    WRITE( 2, '( "node [" )' )
    WRITE( 2, '( "style=filled" )' )
    WRITE( 2, '( "]" )' )

    DO node = 1, nnodes

      weight = REAL( flops( node ) ) / tot_weight

      IF ( weight < 0.01 ) CYCLE !  Prune smallest nodes

      n = sptr( node + 1 ) - sptr( node )
      m = int( rptr( node + 1 ) - rptr( node ) )

!  node idx

      WRITE( 2, '( i10 )', ADVANCE = 'no' ) node
      WRITE( 2, '( " " )', ADVANCE = 'no' )
      WRITE( 2, '( "[" )', ADVANCE = 'no' )

!  node label

      WRITE( 2, '( "label=""" )', ADVANCE = 'no' )
      WRITE( 2, '( "node:", i5,"\n" )', ADVANCE = 'no' ) node
      WRITE( 2, '( "m:", i5,"\n" )', ADVANCE = 'no' ) m
      WRITE( 2, '( "n:", i5,"\n" )', ADVANCE = 'no' ) n
      WRITE( 2, '( "w:", f6.2,"\n" )', ADVANCE = 'no' ) 100 * weight
      WRITE( 2, '( """" )', ADVANCE = 'no' )

!  node color

      WRITE( 2, '( " fillcolor=white" )', ADVANCE = 'no' )

      WRITE( 2, '( "]" )', ADVANCE = 'no' )
      WRITE( 2, '( " " )' )

!  parent node

      IF ( sparent( node ) /= - 1 )                                            &
        WRITE( 2, '( i10, "--", i10 )' )sparent( node ), node
    END DO

    WRITE( 2, '( "}" )' )
    CLOSE( 2 )

    END SUBROUTINE print_atree

    SUBROUTINE print_atree_part( nnodes, sptr, sparent, rptr, topology,        &
                                 nparts, part, exec_loc )

!   prints assembly tree with partitions

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( ip_ ), DIMENSION( nnodes ), INTENT( IN ) :: sparent
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr
    TYPE( HW_numa_region ), DIMENSION( : ), INTENT( IN ) :: topology
    INTEGER( ip_ ), INTENT( IN ) :: nparts
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( IN ) :: part
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( IN ) :: exec_loc

    INTEGER( ip_ ) :: node
    INTEGER( ip_ ) :: n, m !  Node DIMENSIONs
    INTEGER( ip_ ) :: region !  Where to execute node
    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: flops
    REAL :: tot_weight, weight
    INTEGER( ip_ ) :: i, j
    CHARACTER(  LEN = 5 ) :: part_str
    REAL :: small

    small = 0.001

!  count flops below each node

    ALLOCATE( flops( nnodes + 1 ) )
    flops( : ) = 0
    DO node = 1, nnodes
      flops( node ) = flops( node ) + compute_flops( nnodes, sptr, rptr, node )
      j = sparent( node )
      IF( j > 0 ) flops( j ) = flops( j ) + flops( node )
!     print  * , "Node ", node, "parent", j, " flops ", flops( node )
    END DO
    tot_weight = REAL( flops( nnodes ) )

    OPEN( 2, file="atree_part.dot" )
    WRITE( 2, '( "graph atree {" )' )
    WRITE( 2, '( "node [" )' )
    WRITE( 2, '( "style=filled" )' )
    WRITE( 2, '( "]" )' )

    DO i = 1, nparts
      region = mod( ( exec_loc( i ) - 1 ), size( topology ) ) + 1
!     print  * , "part = ", i, ", exec_loc = ", exec_loc( i ),                 &
!                ", region = ", region

      WRITE( part_str, '( i5 )' )part( i )
      WRITE( 2, * )"subgraph cluster"// adjustl( trim( part_str ) ) // " {"
      IF (  exec_loc( i ) > size( topology ) ) THEN !  GPU subtree
        WRITE( 2, * )"color=red"
      ELSE
        WRITE( 2, * )"color=black"
      END IF
      WRITE( 2, '( "label=""" )', ADVANCE = 'no' )
      WRITE( 2, '( "part:", i5,"\n" )', ADVANCE = 'no' ) i
      WRITE( 2, '( "region:", i5,"\n" )', ADVANCE = 'no' ) region
      WRITE( 2, '( "exec_loc:", i5,"\n" )', ADVANCE = 'no' ) exec_loc( i )
      WRITE( 2, '( """" )', ADVANCE = 'no' )

      DO node = part( i ), part( i + 1 ) - 1

        weight = REAL( flops( node ) ) / tot_weight
        IF ( weight < small ) CYCLE !  Prune smallest nodes

        n = sptr( node + 1 ) - sptr( node )
        m = int( rptr( node + 1 ) - rptr( node ) )

!  node idx

        WRITE( 2, '( i10 )', ADVANCE = 'no' ) node
        WRITE( 2, '( " " )', ADVANCE = 'no' )
        WRITE( 2, '( "[" )', ADVANCE = 'no' )

!  node label

        WRITE( 2, '( "label=""" )', ADVANCE = 'no' )
        WRITE( 2, '( "node:", i5,"\n" )', ADVANCE = 'no' ) node
        WRITE( 2, '( "m:", i5,"\n" )', ADVANCE = 'no' ) m
        WRITE( 2, '( "n:", i5,"\n" )', ADVANCE = 'no' ) n
        WRITE( 2, '( "w:", f6.2,"\n" )', ADVANCE = 'no' ) 100 * weight
        WRITE( 2, '( """" )', ADVANCE = 'no' )

!  node color

        WRITE( 2, '( " fillcolor=white" )', ADVANCE = 'no' )

        WRITE( 2, '( "]" )', ADVANCE = 'no' )
        WRITE( 2, '( " " )' )
      END DO

      WRITE( 2, '( "}" )' ) !  Subgraph

      DO node = part( i ), part( i + 1 ) - 1
        weight = REAL( flops( node ) ) / tot_weight
        IF ( weight < small ) CYCLE !  Prune smallest nodes
        IF ( sparent( node ) /=  - 1 )                                         &
          WRITE( 2, '( i10, "--", i10 )' ) sparent( node ), node
      END DO
    END DO

    WRITE( 2, '( "}" )' ) !  Graph
    CLOSE( 2 )

    END SUBROUTINE print_atree_part

    SUBROUTINE build_map( n, ptr, row, perm, invp, nnodes, sptr, rptr, rlist,  &
                          nptr, nlist, st )

!   build a map from A to nodes
!    lcol(nlist( 2,i )) = val(nlist( 1,i ))
!    nptr defines start of each node in nlist

    IMPLICIT none

!  original matrix A

    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr
    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row

!  permutation and its inverse ( some entries of perm may be negative to
!  act as flags for 2x2 pivots, so need to use abs( perm ) )

    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: perm
    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: invp

!  supernode partition of L

    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr

!  row indices of L

    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr
    INTEGER( ip_ ), DIMENSION( rptr( nnodes + 1 ) - 1 ), INTENT( IN ) :: rlist

!  output mapping

    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( OUT ) :: nptr
    INTEGER( long_ ), DIMENSION( 2, ptr( n + 1 ) - 1 ), INTENT( OUT ) :: nlist

!  error check paramter

    INTEGER( ip_ ), INTENT( OUT ) :: st
    INTEGER( ip_ ) :: i, j, k, blkm, col, node
    INTEGER( long_ ) :: ii, jj, pp
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: ptr2, row2
    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: origin
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: map

    ALLOCATE( map( n ), ptr2( n + 3 ), row2( ptr( n + 1 ) - 1 ),               &
              origin( ptr( n + 1 ) - 1 ), STAT = st )
    IF ( st /= 0 ) RETURN

!  build transpose of A in ptr2, row2. Store original posn of entries in
!  origin array.

!  Count number of entries in row i in ptr2( i + 2 ). Don't include diagonals.

    ptr2( : ) = 0
    DO i = 1, n
      DO jj = ptr( i ), ptr( i + 1 ) - 1
        k = row( jj )
        IF ( k == i ) CYCLE
        ptr2( k + 2 ) = ptr2( k + 2 ) + 1
      END DO
    END DO

!  compute row starts such that row i starts in posn ptr2( i + 1 )

    ptr2( 1 : 2 ) = 1
    DO i = 1, n
      ptr2( i + 2 ) = ptr2( i + 2 ) + ptr2( i + 1 )
    END DO

!  drop entries into place

    DO i = 1, n
      DO jj = ptr( i ), ptr( i + 1 ) - 1
        k = row( jj )
        IF ( k == i ) CYCLE
        row2( ptr2( k + 1 ) ) = i
        origin( ptr2( k + 1 ) ) = jj
        ptr2( k + 1 ) = ptr2( k + 1 ) + 1
      END DO
    END DO

!  build nptr, nlist map

    pp = 1
    DO node = 1, nnodes
      blkm = int( rptr( node + 1 ) - rptr( node ) )
      nptr( node ) = pp

!  build map for node indices

      DO jj = rptr( node ), rptr( node + 1 ) - 1
        map( rlist( jj ) ) = int( jj-rptr( node ) + 1 )
      END DO

!  build nlist from A-lower transposed

      DO j = sptr( node ), sptr( node + 1 ) - 1
        col = invp( j )
        DO i = ptr2( col ), ptr2( col + 1 ) - 1
          k = ABS( perm( row2( i ) ) ) !  row of L
          IF ( k < j ) CYCLE
          nlist( 2,pp ) = ( j-sptr( node ) ) * blkm + map( k )
          nlist( 1,pp ) = origin( i )
          pp = pp + 1
        END DO
      END DO

 !  build nlist from A-lower

      DO j = sptr( node ), sptr( node + 1 ) - 1
        col = invp( j )
        DO ii = ptr( col ), ptr( col + 1 ) - 1
          k = ABS( perm( row( ii ) ) ) !  row of L
          IF ( k < j ) CYCLE
          nlist( 2,pp ) = ( j-sptr( node ) ) * blkm + map( k )
          nlist( 1,pp ) = ii
          pp = pp + 1
        END DO
      END DO
    END DO
    nptr( nnodes + 1 ) = pp
    END SUBROUTINE build_map

!   ============================================================================
!   ================ extracted from SPRAL_CORE_ANALYSE module ==================
!   ============================================================================

    SUBROUTINE basic_analyse( n, ptr, row, perm, nnodes, sptr, sparent, rptr,  &
                              rlist, nemin, info, stat, nfact, nflops )

!  Outline analysis routine
!
!  for assembled matrix input, this subroutine performs a full analysis.
!  This is essentially a wrapper around the rest of the package.
!
!  Performance might be improved by:
!  * Improving the sort algorithm used in find_row_idx
!
    IMPLICIT none

 !  dimension of system

    INTEGER( ip_ ), INTENT( IN ) :: n

!  column pointers

    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr

!  row indices

    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row

!  perm( i ) must hold position of i in the pivot sequence.
!  On exit, holds the pivot order to be used by factorization.

    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: perm

!  number of supernodes found

    INTEGER( ip_ ), INTENT( OUT ) :: nnodes

!  supernode pointers

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: sptr 

!  assembly tree

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: sparent 

!  pointers to rlist

    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: rptr

!  row lists

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: rlist

!  node amalgamation parameter

    INTEGER( ip_ ), INTENT( IN ) :: nemin
    INTEGER( ip_ ), INTENT( OUT ) :: info
    INTEGER( ip_ ), INTENT( OUT ) :: stat
    INTEGER( long_ ), INTENT( OUT ) :: nfact
    INTEGER( long_ ), INTENT( OUT ) :: nflops

    INTEGER( ip_ ) :: i

!  inverse permutation of perm

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: invp
    INTEGER( ip_ ) :: j
    INTEGER( ip_ ) :: realn !  number of variables with an actual entry present
    INTEGER( ip_ ) :: st !  stat argument in allocate calls

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: scc

!  number of entries in each column

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: cc

!  parent of each node in etree

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: parent

!  temporary permutation  vector

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: tperm

!  quick exit for n < 0.
!  ERROR_ALLOCATION will be signalled, but since allocation status cannot be
!  negative, in this way info about the invalid argument ( n < 0 ) is conveyed

    IF ( n < 0 ) THEN
      st = n
    ELSE
      st = 0
    END IF
    IF ( st /= 0 ) GO TO 490

!  initialize return code to success

    info = 0

!  ensure ALLOCATABLE output arguments are deallocated

    DEALLOCATE( sptr, STAT = st )
    DEALLOCATE( sparent, STAT = st )
    DEALLOCATE( rptr, STAT = st )
    DEALLOCATE( rlist, STAT = st )

!  initialise inverse permutation and check for duplicates

    ALLOCATE( invp( n ), STAT = st )
    IF ( st /= 0 ) GO TO 490
    DO i = 1, n
      j = perm( i )
      invp( j ) = i
    END DO

    realn = n !  Assume full rank

!  build elimination tree

    ALLOCATE( parent( n ), STAT = st )
    IF ( st /= 0 ) GO TO 490
    CALL find_etree( n, ptr, row, perm, invp, parent, st )
    IF ( st /= 0 ) GO TO 490

!  postorder tree (modifies perm!)

    CALL find_postorder( n, realn, ptr, perm, invp, parent, st )
    IF ( st /= 0 ) GO TO 490

    IF ( n /= realn ) info = SSIDS_WARNING_ANALYSIS_SINGULAR
                             
!  determine column counts

    ALLOCATE( cc( n + 1 ), STAT = st )
    IF ( st /= 0 ) GO TO 490
    CALL find_col_counts( n, ptr, row, perm, invp, parent, cc, st )
    IF ( st /= 0 ) GO TO 490

!  identify supernodes

    ALLOCATE( tperm( n ), sptr( n + 1 ), sparent( n ), scc( n ), STAT = st )
    IF ( st /= 0 ) GO TO 490
    CALL find_supernodes( n, realn, parent, cc, tperm, nnodes, sptr, sparent,  &
                          scc, nemin, info, st )
    IF ( info < 0 ) RETURN

!  apply permutation to obtain final elimination order

    CALL apply_perm( n, tperm, perm, invp, cc )

!  determine column patterns - keep%nodes( : )%index

    ALLOCATE( rptr( nnodes + 1 ), rlist( sum( scc( 1 : nnodes ) ) ), STAT = st )
    IF ( st /= 0 ) GO TO 490
    CALL find_row_lists( n, ptr, row, perm, invp, nnodes, sptr,                &
                         sparent, scc, rptr, rlist, info, st )
    IF ( st /= 0 ) GO TO 490

!  calculate info%num_factor and info%num_flops

    CALL calc_stats( nnodes, sptr, scc, nfact=nfact, nflops=nflops )

!  sort entries of row lists

    CALL dbl_tr_sort( n, nnodes, rptr, rlist, st )
    IF ( st /= 0 ) GO TO 490

    RETURN

!  error handlers

490 CONTINUE
    info = SSIDS_ERROR_ALLOCATION
    stat = st
    RETURN
    END SUBROUTINE basic_analyse

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Elimination tree routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    SUBROUTINE find_etree( n, ptr, row, perm, invp, parent, st )

!  this subroutine determines the elimination tree of a PAP^T where A is a
!  sparse symmetric matrix stored in compressed sparse column form with
!  entries both above and below the diagonal present in the argument matrix.
!  P is a permutation stored in order such that order(i) gives the pivot
!  position of column i. i.e. order(3) = 5 means that the fifth pivot is A_33
!
!  The elimination tree is returned in the array parent. parent( i ) gives the
!  parent in the elimination tree of pivot i
!
!  The algorithm used is that of Liu [1]
!
!  [1] Liu, J. W. 1986. A compact row storage scheme for Cholesky factors using
!      elimination trees. ACM TOMS 12, 2, 127--148

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n !  DIMENSION of system

!  column pointers of A

    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr

!  row indices of A

    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row

!  perm(i) is the pivot position of column i

    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: perm
    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: invp !  inverse of perm

!  parent(i) is the parent of pivot i in the elimination tree

    INTEGER( ip_ ), DIMENSION( n ), INTENT( OUT ) :: parent
    INTEGER( ip_ ), INTENT( OUT ) :: st !  stat parmeter for allocate calls

    INTEGER( long_ ) :: i !  next index into row
    INTEGER( ip_ ) :: j !  current entry in row
    INTEGER( ip_ ) :: k !  current ancestor
    INTEGER( ip_ ) :: l !  next ancestor
    INTEGER( ip_ ) :: piv !  current pivot
    INTEGER( ip_ ) :: rowidx !  current column of A = invp( piv )

!  virtual forest, used for path compression ( shortcuts to top of each tree )

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: vforest

!  allocate virtual forest and initialise it

    ALLOCATE( vforest( n ), STAT = st )
    IF ( st /= 0 ) RETURN
    vforest( : ) = n + 1

!  loop over rows of A in pivot order

    piv = 1
    DO WHILE ( piv <= n )
!     print  * , "row ", piv
      rowidx = invp( piv )

!  loop over entries in row in lower triangle of PAP^T

      DO i = ptr( rowidx ), ptr( rowidx + 1 ) - 1
        j = perm( row( i ) )
        IF ( j >= piv ) CYCLE !  not in lower triangle
!       print  * , "  entry ", j
        k = j
        DO WHILE ( vforest( k ) < piv )
          l = vforest( k )
          vforest( k ) = piv
          k = l
        END DO

!  check if we have already done this pivot

        IF ( vforest( k ) == piv ) CYCLE
        parent( k ) = piv
        vforest( k ) = piv
      END DO
      parent( piv ) = n + 1 !  set to be a root if not overwritten
      piv = piv + 1 !  move on to next pivot
    END DO
    END SUBROUTINE find_etree

    SUBROUTINE find_postorder( n, realn, ptr, perm, invp, parent, st )

!  this subroutine will postorder the elimination tree. That is to say it will
!  reorder the nodes of the tree such that they are in depth-first search order
!
!  this is done by performing a depth-first search to identify mapping from the
!  original pivot order to the new one. This map is then applied to order, invp
!  and parent to enact the relabelling.

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), INTENT( OUT ) :: realn
    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr

!  perm(i) is the pivot  position of column i

    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: perm
    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: invp !  inverse of perm

!  parent(i) is the parent of pivot i in the elimination tree

    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: parent
    INTEGER( ip_ ), INTENT( OUT ) :: st !  stat parmeter for allocate calls

!  chead(i) is first child of i

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: chead

!  cnext(i) is next child of i

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: cnext
    INTEGER( ip_ ) :: i
    INTEGER( ip_ ) :: id
    INTEGER( ip_ ) :: j

!  mapping from original pivot order to new one

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: map
    INTEGER( ip_ ) :: node
    INTEGER( ip_ ) :: shead !  pointer to top of stack

!  stack for depth first search

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: stack

    realn = n

!  build linked lists of children for each node

    ALLOCATE( chead( n + 1 ), cnext( n + 1 ), STAT = st )
    IF ( st /= 0 ) RETURN
    chead( : ) =  - 1 !  no parent if necessary
    DO i = n, 1,  - 1 !  do in reverse order so they come off in original order
      j = parent( i )
      cnext( i ) = chead( j )
      chead( j ) = i
    END DO

!  perform depth first search to build map

    ALLOCATE( map( n + 1 ), stack( n ), STAT = st )
    IF ( st /= 0 ) RETURN

!  place virtual root on top of stack

    shead = 1
    stack( shead ) = n + 1
    id = n + 1 !  next node id

    DO WHILE ( shead /= 0 )

!  get node from top of stack

      node = stack( shead )
      shead = shead - 1

!  number it

      map( node ) = id
      id = id - 1

!  place all its children on the stack such that the last child is
!  at the top of the stack and first child closest to the bottom

      IF ( node == n + 1 ) THEN

!  virtual root node, detect children with no entries at same time
!  placing those that are empty at the top of the stack
!  first do those which are proper roots

        i = chead( node )
        DO WHILE ( i /= - 1 )
          IF ( ptr( invp( i ) + 1 ) == ptr( invp( i ) ) ) THEN
            i = cnext( i )
            CYCLE
          END IF
          shead = shead + 1
          stack( shead ) = i
          i = cnext( i )
        END DO

!  second do those which are null roots

        i = chead( node )
        DO while ( i /=  - 1 )
          IF ( ptr( invp( i ) + 1 ) /= - ptr( invp( i ) ) ) THEN
            i = cnext( i )
            CYCLE
          END IF
          realn = realn - 1
          shead = shead + 1
          stack( shead ) = i
          i = cnext( i )
         END DO
       ELSE !  A normal node
         i = chead( node )
         DO while ( i /=  - 1 )
           shead = shead + 1
           stack( shead ) = i
           i = cnext( i )
         END DO
       END IF
    END DO

!  Apply map to perm, invp and parent

!  invp is straight forward, use stack as a temporary

    stack( 1 : n ) = invp( 1 : n )
    DO i = 1, n
      j = map( i )
      invp( j ) = stack( i )
    END DO

!  perm can be easily done as the inverse of invp

    DO i = 1, n
      perm( invp( i ) ) = i
    END DO

!  parent is done in two stages. The first copies it to stack and permutes
!  parent( i ), but not the locations. i.e. if 1 is a parent of 3, and
!  map( 1 )=2 and map( 3 )=4, then the first stage sets stack( 1 ) = 4.
!  The second stage then permutes the entries of map back into parent

    DO i = 1, n
      stack( i ) = map( parent( i ) )
    END DO
    DO i = 1, n
      parent( map( i ) ) = stack( i )
    END DO
    END SUBROUTINE find_postorder

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Column count routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    SUBROUTINE find_col_counts( n, ptr, row, perm, invp, parent, cc, st )

!  this subroutine determines column counts given the elimination tree and
!  pattern of the matrix PAP^T
!
!  the algorithm is a specialisation of that given by Gilbert, Ng and Peyton 
!  [1] to only determine column counts. It is also described in Section 4.4 
!  "Row counts" of [2]
!
!  The essential technique is to determine the net number of entries introduced
!  at a node ( the "weight" in [1] ). This is composed over the following terms:
!   wt[i] = [ - #children of node i
!             - #common indices between children
!             + #additional "new" row indices from column of A ]
!
!  The clever part of this algorithm is how to determine the number of common
!  indices between the children. This is accomplished by storing the last column
!  at which an index was encountered, and a partial elimination tree. This
!  partial elimination tree consists of all nodes processed so far, plus their
!  parents. As we have a postorder on the tree, the current top of the tree
!  containing node i is the least common ancestor of node i and the current node
!  We then observe that the first time an index will be double counted is at the
!  least common ancestor of the current node and the last node where it was
!  encountered.
!
!  [1] Gilbert, Ng, Peyton, "An efficient algorithm to compute row and column
!      counts for sparse Cholesky factorization", SIMAX 15( 4 ) 1994
!
!  [2] Tim Davis's book "Direct Methods for Sparse Linear Systems", SIAM 2006

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n !  dimension of system

!  column pointers of A

    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr

!  row indices of A

    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row

!  perm(i) is the pivot position of column i

    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: perm
    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: invp !  inverse of perm

!  parent(i ) is the parent of pivot i in the elimination tree

    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: parent

!  on exit, cc(i) is the number of entries in the lower triangular part of
!  L (includes diagonal) for the column containing pivot i. For most of the 
!  routine however, it is used as a work space to track the net number of 
!  entries appearing for the first time at node i of the elimination tree 
!  (this may be negative)

    INTEGER( ip_ ), DIMENSION( n + 1 ), INTENT( OUT ) :: cc

    INTEGER( ip_ ), INTENT( OUT ) :: st !  stat parmeter for allocate calls

    INTEGER( ip_ ) :: col !  column of matrix associated with piv
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: first !  first descendants
    INTEGER( ip_ ) :: i
    INTEGER( long_ ) :: ii

!  previous neighbour

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: last_nbr
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: last_p !  previous p?
    INTEGER( ip_ ) :: par !  parent node of piv
    INTEGER( ip_ ) :: piv !  current pivot
    INTEGER( ip_ ) :: pp !  last pivot where u was encountered
    INTEGER( ip_ ) :: lca !  least common ancestor of piv and pp
    INTEGER( ip_ ) :: u !  current entry in column col
    INTEGER( ip_ ) :: uwt !  weight of u
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: vforest !  virtual forest

!  determine first descendants, and set cc = 1 for leaves and cc = 0 for
!  non-leaves

    ALLOCATE( first( n + 1 ), STAT = st )
    IF ( st /= 0 ) RETURN
    DO i = 1, n + 1
      first( i ) = i
    END DO
    DO i = 1, n
      par = parent( i )
      first( par ) = min( first( i ), first( par ) ) !  first descendant
      IF ( first( i ) == i ) THEN !  is it a leaf or not?
        cc( i ) = 1
      ELSE
         cc( i ) = 0
      END IF
    END DO
    cc( n + 1 ) = n + 1 !  Set to non-physical value

!  we store the partial elimination trees in a virtual forest. It is
!  initialised such that each node is in its own tree to begin with

    ALLOCATE( vforest( n + 1 ), STAT = st )
    IF ( st /= 0 ) RETURN
    vforest( : ) = 0

!  initialise previous pivot and neightbour arrays to indicate no previous
!  pivot or neightbour

    ALLOCATE( last_p( n + 1 ), last_nbr( n + 1 ), STAT = st )
    IF ( st /= 0 ) RETURN
    last_p( : ) = 0
    last_nbr( : ) = 0

!  determine cc( i ), the number of net new entries to pass up tree from
!  node i. Loop over entries in column below the diagonal

    DO piv = 1, n
      col = invp( piv )
      DO ii = ptr( col ), ptr( col + 1 ) - 1
        u = perm( row( ii ) )
        IF ( u <= piv ) CYCLE !  not in lower triangular part

!  check if entry has been seen by a descendant of this pivot, if
!  so we skip the tests that would first add one to the current
!  pivot's weight before then subtracting it again.

        IF ( first( piv ) > last_nbr( u ) ) THEN

 !  count new entry in current column

          uwt = 1
          cc( piv ) = cc( piv ) + uwt

 !  Determine least common ancestor of piv and the node at which
 !  u was last encountred

          pp = last_p( u )
          IF ( pp /= 0 ) THEN

!  u has been seen before, find top of partial elimination
!  tree for node pp

            lca = FIND( vforest, pp )

!  prevent double counting of u at node lca

            cc( lca ) = cc( lca ) - uwt
          END IF

!  update last as u has now been seen at piv.

          last_p( u ) = piv
        END IF

!  record last neighbour of u so we can determine if it has been
!  seen in this subtree before

        last_nbr( u ) = piv
      END DO

!  pass uneliminated variables up to parent

      par = parent( piv )
      cc( par ) = cc( par ) + cc( piv ) - 1

!  place the parent of piv into the same partial elimination tree as piv

      vforest( piv ) = par !  operation "UNION" from [1]
    END DO
    END SUBROUTINE find_col_counts

    INTEGER( ip_ ) function FIND( vforest, u )

!  return top most element of tree containing u.
!  Implements path compression to speed up subsequent searches.

    IMPLICIT none
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: vforest
    INTEGER( ip_ ), INTENT( IN ) :: u

    INTEGER( ip_ ) :: current, prev

    prev =  - 1
    current = u
    DO WHILE ( vforest( current ) /= 0 )
       prev = current
       current = vforest( current )
       IF ( vforest( current ) /= 0 ) vforest( prev ) = vforest( current )
    END DO

    FIND = current
    END FUNCTION FIND

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Supernode amalgamation routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    SUBROUTINE find_supernodes( n, realn, parent, cc, sperm, nnodes, sptr,     &
                                sparent, scc, nemin, info, st )

!  this subroutine identifies ( relaxed ) supernodes from the elimination tree
!  and column counts

!  A node, u, and its parent, v, are merged if:
!  ( a ) No new fill-in is introduced i.e. cc( v ) = cc( u )-1
!  ( b ) The number of columns in both u and v is less than nemin

!  Note: assembly tree must be POSTORDERED on output

    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), INTENT( IN ) :: realn

!  parent( i ) is the parent of supernode i in the elimination/assembly tree

    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: parent

!  cc(i) is the column count of supernode i, including elements eliminated 
!  at supernode i

    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: cc

!  on exit contains a  permutation from pivot order to a new pivot order 
!  with contigous  supernodes

    INTEGER( ip_ ), DIMENSION( n ), INTENT( OUT ) :: sperm
    INTEGER( ip_ ), INTENT( OUT ) :: nnodes !  number of supernodes
    INTEGER( ip_ ), DIMENSION( n + 1 ), INTENT( OUT ) :: sptr
    INTEGER( ip_ ), DIMENSION( n ), INTENT( OUT ) :: sparent
    INTEGER( ip_ ), DIMENSION( n ), INTENT( OUT ) :: scc
    INTEGER( ip_ ), INTENT( IN ) :: nemin
    INTEGER( ip_ ), INTENT( INOUT ) :: info
    INTEGER( ip_ ), INTENT( OUT ) :: st !  stat paremter from allocate calls

    INTEGER( ip_ ) :: i, j, k

!  used to track height  of tree

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: height

!  flag array for nodes to  finalise

    LOGICAL, DIMENSION( : ), ALLOCATABLE :: mark

!  map vertex idx -> supernode idx

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: map

!  number of eliminated  variables

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: nelim

!  number of eliminated supervariables

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: nvert
    INTEGER( ip_ ) :: node

!  temporary array of snode pars

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: npar

!  parent of current node

    INTEGER( ip_ ) :: par

!  current head of stack

    INTEGER( ip_ ) :: shead

!  used to navigate tree

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: stack
    INTEGER( ip_ ) :: v

!  heads of vertex linked  lists

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: vhead

!  next element in linked  lists

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: vnext

!  number of explicit  zeros

    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: ezero

!  chead( i ) is first child  of i

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: chead

!  cnext( i ) is next child  of i

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: cnext
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: child
    INTEGER( ip_ ) :: nchild
    INTEGER( ip_ ) :: start !  First pivot in block pivot
    INTEGER( ip_ ) :: totalwt !  sum of weights

!  initialise supernode representation

    ALLOCATE( nelim( n + 1 ), nvert( n + 1 ), vhead( n + 1 ),                  &
              vnext( n + 1 ), stack( n ), height( n + 1 ), mark( n ), STAT = st)
    IF ( st /= 0 ) GO TO 490
    vnext( : ) =  - 1
    vhead( : ) =  - 1
    height( : ) = 1

!  initialise number of variables in each node

    nelim( 1 : n + 1 ) = 1
    totalwt = n
    nvert( 1 : n + 1 ) = 1 !  Note: Explicit bounds used to silence warning

    ALLOCATE( map( n + 1 ), npar( n + 1 ), ezero( n + 1 ), STAT = st )
    IF ( st /= 0 ) GO TO 490

    ezero( : ) = 0 !  Initially no explicit zeros
    ezero( n + 1 ) = huge( ezero ) !  ensure root is not merged

!  ensure virtual root never gets amlgamated

    nelim( n + 1 ) = totalwt + 1 + nemin

!  build child linked lists for nodes; merge block pivots if needed

    ALLOCATE( chead( n + 1 ), cnext( n + 1 ), child( n ), STAT = st )
    IF ( st /= 0 ) GO TO 490
    chead( : ) =  - 1 !  no parent if necessary
    DO i = realn, 1,  - 1 !  do in reverse order so come off in original order
      j = parent( i )
      cnext( i ) = chead( j )
      chead( j ) = i
    END DO

!  merge supernodes

    v = 1
    nnodes = 0
    start=n + 2
    DO par = 1, n + 1
      nchild = 0
      node = chead( par )
      DO while ( node /=  - 1 )
        nchild = nchild + 1
        child( nchild ) = node
        node = cnext( node )
      END DO
      CALL sort_by_val( nchild, child, cc, st )
      IF ( st /= 0 ) GO TO 490

      DO j = 1, nchild
        node = child( j )
        IF ( do_merge( node, par, nelim, cc, ezero, nemin ) ) THEN

!  merge contents of node into par. Delete node

          CALL merge_nodes( node, par, nelim, nvert, vhead, vnext, height,     &
                            ezero, cc )
          mark( node ) = .FALSE.
        ELSE
          mark( node ) = .TRUE.
        END IF
      END DO
    END DO

    DO node = 1, realn

!  node not merged, now a complete supernode

      IF ( .NOT. mark( node ) ) CYCLE

!  record start of supernode

      nnodes = nnodes + 1
      sptr( nnodes ) = v
      npar( nnodes ) = parent( node )
      scc( nnodes ) = cc( node ) + nelim( node ) - 1

!  record height in tree of parent vertices

      height( parent( node ) )                                                 &
        = MAX( height( parent( node ) ), height( node ) + 1 )

!  determine last vertex of node so we can number backwards

      v = v + nvert( node )
      k = v

!  loop over member vertices of node and number them

      shead = 1
      stack( shead ) = node
      DO while ( shead > 0 )
        i = stack( shead )
        shead = shead - 1

!  order current vertex

        k = k - 1
        sperm( i ) = k
        map( i ) = nnodes

!  stack successor, if any

        IF ( vnext( i ) /=  - 1 ) THEN
          shead = shead + 1
          stack( shead ) = vnext( i )
        END IF

!  descend into tree rooted at i

        IF ( vhead( i ) /=  - 1 ) THEN
          shead = shead + 1
          stack( shead ) = vhead( i )
        END IF
      END DO
    END DO
    sptr( nnodes + 1 ) = v !  Record end of final supernode
    map( n + 1 ) = nnodes + 1 !  virtual root vertex maps to virtual root sn
    npar( nnodes + 1 ) = n + 1

!  handle permutation of empty columns

    DO i = realn + 1, n
      sperm( i ) = i
    END DO

!  allocate arrays for return and copy data into them correctly

    DO node = 1, nnodes
      par = npar( node ) !  parent /vertex/ of supernode
      par = map( par )   !  parent /node/   of supernode
      sparent( node ) = par !  store parent
    END DO
    RETURN

490 CONTINUE
    info = SSIDS_ERROR_ALLOCATION
    RETURN
    END SUBROUTINE find_supernodes

    RECURSIVE SUBROUTINE sort_by_val( n, idx, val, st )

!  sort n items labelled by idx into decreasing order of val( idx( i ) )

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: idx
    INTEGER( ip_ ), DIMENSION( : ), INTENT( IN ) :: val
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: ice_idx, ice_val, ik_idx, ik_val
    INTEGER( ip_ ) :: klo,kor,k,kdummy

    st = 0

    IF ( n >= minsz_ms ) THEN
      CALL sort_by_val_ms( n, idx, val, st )
    ELSE
      klo = 2
      kor = n

!  items kor, kor + 1, .... ,nchild are in order

     DO kdummy = klo, n
        ice_idx = idx( kor - 1 )
        ice_val = val( ice_idx )
        DO k = kor, n
          ik_idx = idx( k )
          ik_val = val( ik_idx )
          IF ( ice_val >= ik_val ) EXIT
          idx( k - 1 ) = ik_idx
        END DO
        idx( k - 1 ) = ice_idx
        kor = kor - 1
      END DO
    END IF
    END SUBROUTINE sort_by_val

    RECURSIVE SUBROUTINE sort_by_val_ms( n, idx, val, st )

!  sort n items labelled by idx into decreasing order of val(idx(i))

!  merge sort version, dramatically improves performance for nodes with large
!  numbers of children (passes to simple sort for small numbers of entries)

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: idx
    INTEGER( ip_ ), DIMENSION( : ), INTENT( IN ) :: val
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: i, j, jj, jj2, k, kk, kk2
    INTEGER( ip_ ) :: mid
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: work

    IF ( n <= 1 ) RETURN
    IF ( n < minsz_ms ) THEN
      CALL sort_by_val( n, idx, val, st )
      RETURN
    END IF
    mid = ( n - 1 ) / 2 + 1

!  Recurse to order half lists

    CALL sort_by_val_ms( mid, idx( 1 : mid ), val, st )
    IF ( st /= 0 ) RETURN
    CALL sort_by_val_ms( n - mid, idx( mid + 1 : n ), val, st )
    IF ( st /= 0 ) RETURN

!  merge two half lists, taking a copy of the first half list so we don't 
!  overwrite it

    ALLOCATE( work( mid ), STAT = st )
    IF ( st /= 0 ) RETURN
    work( : ) = idx( 1 : mid )
    j = 1
    k = mid + 1
    jj = work( j )
    jj2 = val( jj )
    kk = idx( k )
    kk2 = val( kk )
    DO i = 1, n
      IF ( jj2 >= kk2 ) THEN
        idx( i ) = jj
        j = j + 1
        IF( j > mid ) EXIT
        jj = work( j )
        jj2 = val( jj )
      ELSE
        idx( i ) = kk
        k = k + 1
        IF ( k > n ) EXIT
        kk = idx( k )
        kk2 = val( kk )
      END IF
    END DO
    IF ( j <= mid ) idx( i + 1 : n ) = work( j : mid )
    END SUBROUTINE sort_by_val_ms

    LOGICAL FUNCTION do_merge( node, par, nelim, cc, ezero, nemin )

!  return .TRUE. if we should merge node and par, .FALSE. if we should not

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: node !  node to merge and delete
    INTEGER( ip_ ), INTENT( IN ) :: par !  parent to merge into
    INTEGER( ip_ ), DIMENSION( : ), INTENT( IN ) :: nelim
    INTEGER( ip_ ), DIMENSION( : ), INTENT( IN ) :: cc
    INTEGER( long_ ), DIMENSION( : ), INTENT( IN ) :: ezero
    INTEGER( ip_ ), INTENT( IN ) :: nemin

    IF ( ezero( par ) == HUGE( ezero ) ) THEN
      do_merge = .FALSE.
      RETURN
    END IF

    do_merge = ( cc( par ) == cc( node ) - 1 .AND. nelim( par ) == 1 ) .OR.    &
               ( nelim( par ) < nemin .AND. nelim( node ) < nemin )
    END FUNCTION do_merge

    SUBROUTINE merge_nodes( node, par, nelim, nvert, vhead, vnext, height,     &
                            ezero, cc )

!  this subroutine merges node with its parent, deleting node in the process.

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: node !  node to merge and delete
    INTEGER( ip_ ), INTENT( IN ) :: par !  parent to merge into
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: nelim
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: nvert
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: vhead
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: vnext
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: height
    INTEGER( long_ ), DIMENSION( : ), INTENT( INOUT ) :: ezero
    INTEGER( ip_ ), DIMENSION( : ), INTENT( IN ) :: cc

!  add node to list of children merged into par

    vnext( node ) = vhead( par )
    vhead( par ) = node

!  compute number of explicit zeros in new node

    ezero( par ) = ezero( par ) + ezero( node ) +                              &
      ( cc( par ) - 1 + nelim( par ) - cc( node ) + 1_long_ ) * nelim( par )

!  add together eliminated variables

    nelim( par ) = nelim( par ) + nelim( node )
    nvert( par ) = nvert( par ) + nvert( node )

!  nodes have same height

    height( par ) = max( height( par ), height( node ) )
    END SUBROUTINE merge_nodes

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Statistics routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    SUBROUTINE calc_stats( nnodes, sptr, scc, nfact, nflops )

!  This subroutine merely calculates interesting statistics

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( ip_ ), DIMENSION( nnodes ), INTENT( IN ) :: scc
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nfact
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nflops

    INTEGER( ip_ ) :: j
    INTEGER( ip_ ) :: m !  number of entries in retangular part of ndoe
    INTEGER( ip_ ) :: nelim !  width of node
    INTEGER( ip_ ) :: node !  current node of assembly tree
    INTEGER( long_ ) :: r_nfact, r_nflops

    IF ( .NOT. PRESENT( nfact ) .AND.                                         &
         .NOT. PRESENT( nflops ) ) RETURN !  nothing to do

    r_nfact = 0
    r_nflops = 0
    DO node = 1, nnodes
      nelim = sptr( node + 1 ) - sptr( node )
      m = scc( node ) - nelim

!  number of entries

      r_nfact = r_nfact + ( nelim * ( nelim + 1 ) ) / 2 !  triangular block
      r_nfact = r_nfact + nelim * m !  below triangular block

!  flops

      DO j = 1, nelim
        r_nflops = r_nflops + ( m + j ) ** 2
      END DO
    END DO

    IF ( PRESENT( nfact ) ) nfact = r_nfact
    IF ( PRESENT( nflops ) ) nflops = r_nflops

!print  * , "n = ", n
!print  * , "nnodes = ", nnodes
!print  * , "nfact = ", nfact
!print  * , "sum cc=", sum( cc( 1 : n ) )
!print  * , "nflops = ", nflops
    END SUBROUTINE calc_stats

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Row list routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    SUBROUTINE find_row_lists( n, ptr, row, perm, invp, nnodes,                &
                               sptr, sparent, scc, rptr, rlist, info, st )

!  this subroutine determines the row indices for each supernode

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr
    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row
    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: perm
    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: invp
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( ip_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: sptr
    INTEGER( ip_ ), DIMENSION( nnodes ), INTENT( IN ) :: sparent
    INTEGER( ip_ ), DIMENSION( nnodes ), INTENT( IN ) :: scc
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( OUT ) :: rptr
    INTEGER( ip_ ), DIMENSION( SUM( scc( 1 : nnodes ) ) ),                     &
                    INTENT( OUT ) :: rlist
    INTEGER( ip_ ), INTENT( INOUT ) :: info
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: child !  current child of node
    INTEGER( ip_ ) :: col !  current column of matrix corresponding to piv
    INTEGER( long_ ) :: i
    INTEGER( long_ ) :: idx !  current insert position into nodes( node )%index
    INTEGER( ip_ ) :: j
    INTEGER( ip_ ) :: node !  current node of assembly tree
    INTEGER( ip_ ) :: piv !  current pivot position
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: seen  ! tag last time index
                                                         ! was seen
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: chead ! head of child linked
                                                         ! lists
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: cnext ! pointer to next child

!  allocate and initialise memory

    ALLOCATE( seen( n ), chead( nnodes + 1 ), cnext( nnodes + 1 ), STAT = st )
    IF ( st /= 0 ) THEN
      info = SSIDS_ERROR_ALLOCATION
      RETURN
    END IF
    seen( : ) = 0
    chead( : ) = - 1

!  build child linked lists (backwards so pop off in good order)

    DO node = nnodes, 1,  - 1
      i = sparent( node )
      cnext( node ) = chead( i )
      chead( i ) = node
    END DO

!  loop over nodes from bottom up building row lists

    rptr( 1 ) = 1
    DO node = 1, nnodes

!  allocate space for row indices

      rptr( node + 1 ) = rptr( node ) + scc( node )
      idx = rptr( node ) !  insert position

!  add entries eliminated at this node

      DO piv = sptr( node ), sptr( node + 1 ) - 1
        seen( piv ) = node
        rlist( idx ) = piv
        idx = idx + 1
      END DO

!  find indices inherited from children

      child = chead( node )
      DO while ( child /= - 1 )
        DO i = rptr( child ), rptr( child + 1 ) - 1
          j = rlist( i )
          IF ( j < sptr( node ) ) CYCLE !  eliminated
          IF ( seen( j ) == node ) CYCLE !  already seen
          seen( j ) = node
          rlist( idx ) = j
          idx = idx + 1
        END DO
        child = cnext( child )
      END DO

!  find new indices from A

      DO piv = sptr( node ), sptr( node + 1 ) - 1
        col = invp( piv )
        DO i = ptr( col ), ptr( col + 1 ) - 1
          j = perm( row( i ) )
          IF ( j < piv ) CYCLE !  in upper triangle
          IF ( seen( j ) == node ) CYCLE !  already seen in this snode

!  otherwise, this is a new entry

          seen( j ) = node
          rlist( idx ) = j
          idx = idx + 1
        END DO
      END DO
    END DO
    END SUBROUTINE find_row_lists

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Assorted auxilary routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    SUBROUTINE dbl_tr_sort( n, nnodes, rptr, rlist, st )

!  This subroutine performs a double transpose sort on the row indices of sn

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), INTENT( IN ) :: nnodes
    INTEGER( long_ ), DIMENSION( nnodes + 1 ), INTENT( IN ) :: rptr
    INTEGER( ip_ ), DIMENSION( rptr( nnodes + 1 ) - 1 ),                       &
                    INTENT( INOUT ) :: rlist
    INTEGER( ip_ ), INTENT( OUT ) :: st

    INTEGER( ip_ ) :: node
    INTEGER( ip_ ) :: i, j
    INTEGER( long_ ) :: ii, jj
    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: ptr
    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: nptr
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: col

    ALLOCATE( ptr( n + 2 ), STAT = st )
    IF ( st /= 0 ) RETURN
    ptr( : ) = 0

!  Count number of entries in each row. ptr( i + 2 ) = #entries in row i

    DO node = 1, nnodes
      DO ii = rptr( node ), rptr( node + 1 ) - 1
        j = rlist( ii ) !  row entry
        ptr( j + 2 ) = ptr( j + 2 ) + 1
      END DO
    END DO

!  determine row starts. ptr( i + 1 ) = start of row i

    ptr( 1 : 2 ) = 1
    DO i = 1, n
      ptr( i + 2 ) = ptr( i + 1 ) + ptr( i + 2 )
    END DO

    jj = ptr( n + 2 ) - 1 !  total number of entries
    ALLOCATE( col( jj ), STAT = st )
    IF ( st /= 0 ) RETURN

!  now fill in col array

    DO node = 1, nnodes
      DO ii = rptr( node ), rptr( node + 1 ) - 1
        j = rlist( ii ) !  row entry
        col(  ptr( j + 1 )  ) = node
        ptr( j + 1 ) = ptr( j + 1 ) + 1
      END DO
    END DO

!  finally transpose back into nodes

    ALLOCATE( nptr( nnodes ), STAT = st )
    IF ( st /= 0 ) RETURN
    nptr( : ) = rptr( 1 : nnodes )
    DO i = 1, n
      DO jj = ptr( i ), ptr( i + 1 ) - 1
        node = col( jj )
        rlist( nptr( node ) ) = i
        nptr( node ) = nptr( node ) + 1
      END DO
    END DO
    END SUBROUTINE dbl_tr_sort

    SUBROUTINE apply_perm( n, perm, order, invp, cc )

!  This subroutine applies the permutation perm to order, invp and cc

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( ip_ ), DIMENSION( n ), INTENT( IN ) :: perm
    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: order
    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: invp
    INTEGER( ip_ ), DIMENSION( n ), INTENT( INOUT ) :: cc

    INTEGER( ip_ ) :: i, j

!  use order as a temporary variable to permute cc. Don't care about cc(n+1)

    order( 1 : n ) = cc( 1 : n )
    DO i = 1, n
      j = perm( i )
      cc( j ) = order( i )
    END DO

!  use order as a temporary variable to permute invp.

    order( 1 : n ) = invp( 1 : n )
    DO i = 1, n
      j = perm( i )
      invp( j ) = order( i )
    END DO

!  recover order as inverse of invp

    DO i = 1, n
      order( invp( i ) ) = i
    END DO
    END SUBROUTINE apply_perm

!!$!   =========================================================================
!!$!   ==================== extracted from SPRAL_PGM module ====================
!!$!   =========================================================================
!!$
!!$    SUBROUTINE writePPM( funit, bitmap, color, scale )
!!$
!!$!   write out a Portable Pixel Map (.ppm) file
!!$!   values of the array bitmap(:,:) specify an index of the array color(:,:)
!!$!   color(:,:n) should have size colours(3,ncolor) where ncolor is the 
!!$!   maximum number of colors. For a given color i:
!!$!   color( 1,i ) gives the red   component with value between 0 and 255
!!$!   color( 2,i ) gives the green component with value between 0 and 255
!!$!   color( 3,i ) gives the blue  component with value between 0 and 255
!!$
!!$    INTEGER( ip_ ), INTENT( IN ) :: funit
!!$    INTEGER( ip_ ), DIMENSION( : , : ), INTENT( IN ) :: bitmap
!!$    INTEGER( ip_ ), DIMENSION( : , : ), INTENT( IN ) :: color
!!$
!!$!  how many pixels  point occupies
!!$
!!$    INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: scale 
!!$    INTEGER( ip_ ) :: m, n, nlvl, i, j, c, s1, s2, scale2
!!$
!!$    scale2 = 1
!!$    IF ( PRESENT( scale ) ) scale2 = scale
!!$
!!$    m = SIZE( bitmap, 1 )
!!$    n = SIZE( bitmap, 2 )
!!$    nlvl = MAXVAL( bitmap( : , : ) )
!!$
!!$    WRITE( funit, "( A )" ) "P3"
!!$    WRITE( funit, "( 3I5 )" ) n * scale, m * scale, 255
!!$    DO i = 1, m !  loop over rows
!!$      DO s1 = 1, scale2
!!$        DO j = 1, n
!!$          c = bitmap( i, j )
!!$          DO s2 = 1, scale2
!!$            WRITE( funit, "( 3I5 )" ) color( : , c )
!!$          END DO
!!$        END DO
!!$      END DO
!!$    END DO
!!$    END SUBROUTINE writePPM

  END MODULE GALAHAD_SSIDS_analyse_precision
