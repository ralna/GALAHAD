! THIS VERSION: GALAHAD 5.3 - 2025-08-29 AT 14:00 GMT
! (consistent with SPRAL up to issue #250)

#include "ssids_procedures.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S S I D S   M O D U L E  *-*-*-*-*-*-*-*-*-

!  COPYRIGHT (c) 2011 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  authors: Jonathan Hogg and Jennifer Scott
!  Forked from SPRAL and extended for GALAHAD, Nick Gould, version 3.1, 2016
!  Absorbed SSIDS_analyse, SSIDS_fkeep, SPRAL_CORE_ANALYSE
!  & SPRAL_pgm modules, version 5.3, 2025

  MODULE GALAHAD_SSIDS_precision

!     ------------------------------------------------------------
!    |                                                            |
!    | SSIDS, a sparse, symmetric, idefinite direct linear solver |
!    |                                                            |
!     ------------------------------------------------------------

    USE, INTRINSIC :: iso_c_binding
!$  USE omp_lib
    USE GALAHAD_KINDS_precision
    USE GALAHAD_HW, ONLY: HW_guess_topology, HW_numa_region
    USE GALAHAD_MU_precision, ONLY: SSIDS_MATRIX_REAL_SYM_INDEF,               &
                                    SSIDS_MATRIX_REAL_SYM_PSDEF,               &
                                    MU_convert_coord_to_cscl,                  &
                                    MU_clean_cscl_oop,                         &
                                    MU_apply_conversion_map
    USE GALAHAD_MS_precision, ONLY: MS_auction_scale_sym,                      &
                                    MS_equilib_scale_sym,                      &
                                    MS_hungarian_scale_sym,                    &
                                    MS_equilib_control_type,                   &
                                    MS_equilib_inform_type,                    &
                                    MS_hungarian_control_type,                 &
                                    MS_hungarian_inform_type
    USE GALAHAD_MO_precision, ONLY: MO_match_order_metis
    USE GALAHAD_RB_precision, ONLY: RB_write_options, RB_write
    USE GALAHAD_NODEND_precision, ONLY: NODEND_half_order,                     &
                                        NODEND_control_type,                   &
                                        NODEND_inform_type
    USE GALAHAD_SSIDS_types_precision
    USE GALAHAD_SSIDS_subtree_precision, ONLY: numeric_subtree_base,           &
                                               symbolic_subtree_base
    USE GALAHAD_SSIDS_cpu_subtree_precision, ONLY: cpu_numeric_subtree,        &
                                               construct_cpu_symbolic_subtree
!   USE GALAHAD_SSIDS_gpu_subtree_precision, ONLY:                             &
!     construct_gpu_symbolic_subtree
#ifdef PROFILE
     USE GALAHAD_SSIDS_profile_precision, ONLY : profile_begin, profile_end,   &
                                                 profile_add_event
#endif

    IMPLICIT NONE

    PRIVATE
    PUBLIC :: ssids_analyse, ssids_analyse_coord, ssids_factor, ssids_solve,   &
              ssids_free, ssids_enquire_posdef, ssids_enquire_indef,           &
              ssids_alter, SSIDS_akeep_type, SSIDS_fkeep_type,                 &
              SSIDS_control_type, SSIDS_inform_type
!----------------------
!   P a r a m e t e r s
!----------------------

    LOGICAL, PUBLIC, PROTECTED :: ssids_available = .TRUE.
    INTEGER( ip_ ), PARAMETER, PUBLIC :: DEBUG_PRINT_LEVEL = 9999

!  extracted from SPRAL_CORE_ANALYSE
   
    INTEGER( ip_ ), parameter :: minsz_ms = 16 ! minimum size to use merge sort

!----------------------
!   I n t e r f a c e s
!----------------------

!  analyse phase, CSC-lower input

    INTERFACE ssids_analyse
      MODULE PROCEDURE analyse_precision, analyse_precision_ptr32
    END INTERFACE ssids_analyse

!  analyse phase, coordinate input

    INTERFACE ssids_analyse_coord
      MODULE PROCEDURE ssids_analyse_coord_precision
    END INTERFACE ssids_analyse_coord

!  factorize phase

    INTERFACE ssids_factor
      MODULE PROCEDURE ssids_factor_ptr32_precision
      MODULE PROCEDURE ssids_factor_ptr64_precision
    END INTERFACE ssids_factor

!  solve phase

    INTERFACE ssids_solve
      MODULE PROCEDURE ssids_solve_one_precision
      MODULE PROCEDURE ssids_solve_mult_precision
    END INTERFACE ssids_solve

!   free akeep and/or fkeep

    INTERFACE ssids_free
      MODULE PROCEDURE free_akeep_precision
      MODULE PROCEDURE free_fkeep_precision
      MODULE PROCEDURE free_both_precision
    END INTERFACE ssids_free

!  pivot information in positive definite case

    INTERFACE ssids_enquire_posdef
      MODULE PROCEDURE ssids_enquire_posdef_precision
    END INTERFACE ssids_enquire_posdef

!  pivot information in indefinite case

    INTERFACE ssids_enquire_indef
      MODULE PROCEDURE ssids_enquire_indef_precision
    END INTERFACE ssids_enquire_indef

!  alter diagonal

    INTERFACE ssids_alter
      MODULE PROCEDURE ssids_alter_precision
    END INTERFACE ssids_alter

!  extracted from ssids_analyse

    INTERFACE print_atree
      MODULE PROCEDURE print_atree, print_atree_part
    END INTERFACE print_atree

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE :: omp_settings ! CACHES user OpenMP ICV values for later restoration
      LOGICAL :: dynamic
      INTEGER( KIND = ip_ ) :: max_active_levels
    END TYPE omp_settings

!  -------------------------
!  extracts from SSIDS_akeep
!  -------------------------

    TYPE symbolic_subtree_ptr
      INTEGER( ip_ ) :: exec_loc
      CLASS( symbolic_subtree_base ), POINTER :: ptr => null( )
    END TYPE symbolic_subtree_ptr

!  type for information generated in analyse phase

    TYPE, PUBLIC :: SSIDS_akeep_type

!  copy of check as input to analyse phase

      LOGICAL :: check

!  dimension of matrix

      INTEGER( ip_ ) :: n

!  number of nodes in assembly tree

      INTEGER( ip_ ) :: nnodes = - 1
      
!  subtree partition

      INTEGER( ip_ ) :: nparts
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: part
      TYPE( symbolic_subtree_ptr ), DIMENSION( : ), ALLOCATABLE :: subtree
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_ptr
      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: contrib_idx

!  inverse of pivot order that is passed to factorize phase

      INTEGER( ipc_ ), DIMENSION( : ), ALLOCATABLE :: invp

!  map from A to factors. For nodes i, the entries 
!  nlist(1:2,nptr(i):nptr(i+1)-1) define a relationship:
!  nodes( node )%lcol(nlist( 2,j )) = val(nlist(1,j))

      INTEGER( long_ ), DIMENSION( :,: ), ALLOCATABLE :: nlist

!  entries into nlist for nodes of the assembly tree. Has length nnodes+1

      INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: nptr

!  rlist(rptr(i):rptr(i+1)-1) contains the row indices for node i of the 
!  assembly tree. At each node, the list is in elimination order.
!  Allocated within basic_analyse in ssids

      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: rlist 

!  pointers into rlist for nodes of assembly tree. Has length nnodes+1.
!  Allocated within basic_analyse in ssids

      INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: rptr

!  sparent(i) is the parent of node i in assembly tree. sparent(i)=nnodes+1 if 
!  i is a  root. The parent is always numbered higher than each of its children.
!  Allocated within basic_analyse in ssids

      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: sparent 

!  (super)node pointers. Supernode i consists of sptr(i) through sptr(i+1)-1.
!  Allocated within basic_analyse in ssids

      INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: sptr
      
!  the following components are for cleaned up matrix data. LOWER triangle 
!  only. We have to retain these for factorize phase as used if the user wants
!  to do scaling. These components are NOT used if check is set to .false.
!  on call to ssids_analyse

      INTEGER( long_ ), ALLOCATABLE :: ptr( : ) ! column pointers
      INTEGER( ip_ ), ALLOCATABLE :: row( : ) ! row indices
      INTEGER( long_ ) :: lmap ! length of map
      INTEGER( long_ ), ALLOCATABLE :: map( : ) ! map from old A to cleaned A
      
!  scaling from matching-based ordering

      REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: scaling
      
!  machine topology
   
     TYPE( HW_numa_region ), DIMENSION( : ), ALLOCATABLE :: topology
      
!  inform at end of analyse phase

      TYPE( ssids_inform_type ) :: inform

    CONTAINS

      PROCEDURE, PASS( akeep ) :: free => free_akeep

    END TYPE SSIDS_akeep_type

!  -------------------------
!  extracts from SSIDS_fkeep
!  -------------------------

     TYPE numeric_subtree_ptr
       CLASS( numeric_subtree_base ), POINTER :: ptr
     END TYPE numeric_subtree_ptr

!  type for data generated in factorise phase

     TYPE, PUBLIC :: ssids_fkeep_type

!  stores scaling for each entry ( in original matrix order )

       REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: scaling
       LOGICAL :: pos_def ! set to true if user indicates matrix pos. definite

!  factored subtrees

       TYPE( numeric_subtree_ptr ), DIMENSION( : ), ALLOCATABLE :: subtree

!  copy of inform on exit from factorize

       TYPE( ssids_inform_type ) :: inform

     CONTAINS

!  do actual factorization

       PROCEDURE, PASS( fkeep ) :: inner_factor => inner_factor_cpu 

!  do actual solve

       PROCEDURE, PASS( fkeep ) :: inner_solve => inner_solve_cpu 
       PROCEDURE, PASS( fkeep ) :: enquire_posdef => enquire_posdef_cpu
       PROCEDURE, PASS( fkeep ) :: enquire_indef => enquire_indef_cpu
       PROCEDURE, PASS( fkeep ) :: alter => alter_cpu ! Alter D values
       PROCEDURE, PASS( fkeep ) :: free => free_fkeep ! Frees memory
     END TYPE ssids_fkeep_type

  CONTAINS

!-*-*-  G A L A H A D - S S I D S _ analyse_ptr32  S U B R O U T I N E  -*-*-

    SUBROUTINE analyse_precision_ptr32( check, n, ptr, row, akeep, control,    &
                                        inform, order, val, topology )

!  provides a wrapper around analyse_precision( ) that copies the
!  32-bit ptr to a 64-bit array before calling the 64-bit version

    IMPLICIT NONE
    LOGICAL, INTENT( IN ) :: check
    INTEGER( KIND = ip_ ), INTENT( IN ) :: n
    INTEGER( KIND = i4_ ), INTENT( IN ) :: ptr( : )
    INTEGER( KIND = ip_ ), INTENT( IN ) :: row( : )
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    INTEGER( KIND = ip_ ), OPTIONAL, INTENT( INOUT ) :: order( : )
    REAL( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: val( : )
    TYPE( HW_numa_region ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: topology

!  local variables

    INTEGER( KIND = i8_ ), DIMENSION( : ), ALLOCATABLE :: ptr64

!  copy 32-bit ptr to 64-bit version

    ALLOCATE( ptr64( n + 1 ), stat=inform%stat )
    IF ( inform%stat /= 0 ) THEN
      inform%flag = SSIDS_ERROR_ALLOCATION
      akeep%inform = inform
      CALL inform%print_flag( control, 'ssids_analyse' )
      RETURN
    END IF
    ptr64( 1 : n + 1 ) = ptr( 1 : n + 1 )

!  call 64-bit version of routine

    CALL analyse_precision( check, n, ptr64, row, akeep, control, inform,      &
                            order = order, val = val, topology = topology )
    RETURN

    END SUBROUTINE analyse_precision_ptr32

!-*-*-  G A L A H A D - S S I D S _ a n a l y s e   S U B R O U T I N E  -*-*-

    SUBROUTINE analyse_precision( check, n, ptr, row, akeep, control, inform,  &
                                  order, val, topology )

!  analyse phase: matrix entered in CSC format (lower triangle).
!  The user optionally inputs the pivot order. If not, metis called.
!  Structure is then expanded, supervariables are computed and then
!  the assembly tree is constructed and the data structures required
!  by the factorization are set up.

!  There is no checking of the user's data if check = .false.
!  Otherwise, matrix_util routines are used to clean data.

!  see user documentation for full detail on parameters.

    IMPLICIT NONE

!  clean matrix data if check=true (cleaned version stored in akeep)

    LOGICAL, INTENT( IN ) :: check

!  order of A

    INTEGER( KIND = ip_ ), INTENT( IN ) :: n

!  column pointers of A

    INTEGER( KIND = i8_ ), INTENT( IN ) :: ptr( : )

!  row indices of A

    INTEGER( KIND = ip_ ), INTENT( IN ) :: row( : )

!  symbolic factorization output

    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep

!  user-supplied options

    TYPE( SSIDS_control_type ), INTENT( IN ) :: control

!  stats/information returned to the user

    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  return ordering to user/allow user to supply order

    INTEGER( KIND = ip_ ), OPTIONAL, INTENT( INOUT ) :: order( : )

!  values of A - only required if matching-based ordering requested

    REAL( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: val( : )

!  optionally Specify machine topology to work with

    TYPE( HW_numa_region ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: topology

!  local variables

    CHARACTER( LEN = 50 ) :: context !  Procedure name ( used when printing )
    INTEGER( KIND = ip_ ) :: mu_flag !  error flag for matrix_util routines
    INTEGER( KIND = long_ ) :: nz    !  entries in expanded matrix
    INTEGER( KIND = ip_ ) :: st      !  stat parameter
    INTEGER( KIND = ip_ ) :: flag    !  error flag for metis

    INTEGER( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: order2

!  col ptrs and row indices for the expanded matrix

    INTEGER( KIND = long_ ), DIMENSION( : ), ALLOCATABLE :: ptr2
    INTEGER( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: row2  

!  The following are only used for matching-based orderings
!  cleaned values if val is present and checking is required

    REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: val_clean

!  expanded matrix if val is present

    REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: val2

    INTEGER( KIND = ip_ ) :: mo_flag
    INTEGER( KIND = ip_ ) :: free_flag
    TYPE( ssids_inform_type ) :: inform_default

!  initialise

    context = 'ssids_analyse'
    inform = inform_default
    CALL ssids_free( akeep, free_flag )
!   IF ( free_flag /= 0 ) THEN
!      inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
!      inform%cuda_error = free_flag
!      akeep%inform = inform
!      CALL inform%print_flag( control, context )
!      RETURN
!   END IF

!  print status on entry

    CALL control%print_summary_analyse( context )
    IF ( control%print_level >= 1 .AND.                                        &
         control%unit_diagnostics >= 0 ) THEN
      WRITE( control%unit_diagnostics, '( A, I15 )' )                          &
            ' n                         =  ', n
    END IF

    akeep%check = check
    akeep%n = n

!  checking of matrix data

    IF ( n < 0 ) THEN
      inform%flag = SSIDS_ERROR_A_N_OOR
      akeep%inform = inform
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  check if analyse has been run

    IF ( n == 0 ) THEN
      akeep%nnodes = 0
      ALLOCATE( akeep%sptr( 0 ), STAT = st )
      IF ( st /= 0 ) GO TO 490
      akeep%inform = inform
      RETURN
    END IF

!  check control%ordering has a valid value

    IF ( control%ordering < 0 .OR. control%ordering > 2 ) THEN
      inform%flag = SSIDS_ERROR_ORDER
      akeep%inform = inform
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  check val present when expected

    IF ( control%ordering == 2 ) THEN
      IF ( .NOT. PRESENT( val ) ) THEN
        inform%flag = SSIDS_ERROR_VAL
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END IF
    END IF

    st = 0
    IF ( check ) THEN
      ALLOCATE ( akeep%ptr( n + 1 ),STAT = st )
      IF ( st /= 0 ) GO TO 490

      IF ( PRESENT( val ) ) THEN
        CALL MU_clean_cscl_oop( SSIDS_MATRIX_REAL_SYM_INDEF, n, n, ptr, row,   &
                             akeep%ptr, akeep%row, mu_flag, val_in = val,      &
                             val_out = val_clean, lmap = akeep%lmap,           &
                             map = akeep%map, noor = inform%matrix_outrange,   &
                             ndup = inform%matrix_dup )
      ELSE
        CALL MU_clean_cscl_oop( SSIDS_MATRIX_REAL_SYM_INDEF, n, n, ptr, row,   &
                             akeep%ptr, akeep%row, mu_flag, lmap = akeep%lmap, &
                             map = akeep%map, noor = inform%matrix_outrange,   &
                             ndup = inform%matrix_dup )
      END IF

!  check for errors

      IF ( mu_flag < 0 ) THEN
        IF ( mu_flag == - 1 ) inform%flag = SSIDS_ERROR_ALLOCATION
        IF ( mu_flag == - 5 ) inform%flag = SSIDS_ERROR_A_PTR
        IF ( mu_flag == - 6 ) inform%flag = SSIDS_ERROR_A_PTR
        IF ( mu_flag == - 10 ) inform%flag = SSIDS_ERROR_A_ALL_OOR
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END IF

!  check whether warning needs to be raised. Note: same numbering of 
!  positive flags as in matrix_util

      IF ( mu_flag > 0 ) THEN
        inform%flag = mu_flag
        CALL inform%print_flag( control, context )
      END IF
      nz = akeep%ptr( n + 1 ) - 1
    ELSE
      nz = ptr( n + 1 ) - 1
    END IF

!  if the pivot order is not supplied, we need to compute an order.
!  Otherwise, we check the supplied order

    ALLOCATE( akeep%invp( n ), order2( n ), ptr2( n + 1 ), row2( 2 * nz ),     &
              STAT = st )
    IF ( st /= 0 ) GO TO 490
    IF ( control%ordering == 2 ) THEN
      ALLOCATE( akeep%scaling( n ), val2( 2 * nz ), STAT = st )
      IF ( st /= 0 ) GO TO 490
    END IF

    SELECT CASE( control%ordering )

!  user-provided ordering

    CASE( 0 )

!  raise an error if the user claims but fails to supply the order

      IF ( .NOT. PRESENT( order ) ) THEN
        inform%flag = SSIDS_ERROR_ORDER
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END IF
      CALL check_order( n,order,akeep%invp,control,inform )
      IF ( inform%flag < 0 ) GO TO 490
      order2( 1 : n ) = order( 1 : n )
      IF ( check ) THEN
        CALL expand_pattern( n, nz, akeep%ptr, akeep%row, ptr2, row2 )
      ELSE
        CALL expand_pattern( n, nz, ptr, row, ptr2, row2 )
      END IF
    CASE( 1 )

!  METIS ordering

      IF ( check ) THEN
        CALL NODEND_half_order( n, akeep%ptr, akeep%row, order2,               &
                                control%nodend_control, inform%nodend_inform )
        inform%flag = inform%nodend_inform%status
        st = inform%nodend_inform%alloc_status
        IF ( inform%flag < 0 ) GO TO 490
        CALL expand_pattern( n, nz, akeep%ptr, akeep%row, ptr2, row2 )
      ELSE
        CALL NODEND_half_order( n, ptr, row, order2,                           &
                                control%nodend_control, inform%nodend_inform )
        inform%flag = inform%nodend_inform%status
        st = inform%nodend_inform%alloc_status
        IF ( inform%flag < 0 ) GO TO 490
        CALL expand_pattern( n, nz, ptr, row, ptr2, row2 )
      END IF
      flag = inform%flag

!  matching-based ordering required; expand the matrix as more efficient to do
!  it and then call MO_match_order_metis( ) with full matrix supplied

    CASE( 2 )
      IF ( check ) THEN
        CALL expand_matrix( n, nz, akeep%ptr, akeep%row, val_clean, ptr2,      &
                            row2, val2 )
        DEALLOCATE ( val_clean, STAT = st )
      ELSE
        CALL expand_matrix( n, nz, ptr, row, val, ptr2, row2, val2 )
      END IF
      CALL MO_match_order_metis( n, ptr2, row2, val2, order2, akeep%scaling,   &
                                 control%nodend_control, inform%nodend_inform, &
                                 mo_flag, inform%stat )
      SELECT CASE( mo_flag )

!  success: do nothing

      CASE( 0 )

!  singularity warning required

      CASE( 1 )
        inform%flag = SSIDS_WARNING_ANALYSIS_SINGULAR
      CASE( - 1 )
        inform%flag = SSIDS_ERROR_ALLOCATION
        akeep%inform = inform
        CALL inform%print_flag( control, context )
         RETURN
      CASE default
        inform%flag = SSIDS_ERROR_UNKNOWN
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END SELECT
      DEALLOCATE( val2, STAT = st )
    END SELECT

!  discover the topology

!  user supplied

    IF ( PRESENT( topology ) ) THEN
      ALLOCATE( akeep%topology( SIZE( topology ) ), STAT = st )
      IF ( st /= 0 ) GO TO 490
      akeep%topology( : ) = topology( : )

!  guess it

    ELSE
      CALL HW_guess_topology( akeep%topology, st )
      IF ( st /= 0 ) GO TO 490
    END IF
    CALL squash_topology( akeep%topology, control, st )
    IF ( st /= 0 ) GO TO 490

!  perform the rest of analyse

    IF ( check ) THEN
      CALL analyse_phase( n, akeep%ptr, akeep%row, ptr2, row2, order2,         &
                          akeep%invp, akeep, control, inform )
    ELSE
      CALL analyse_phase( n, ptr, row, ptr2, row2, order2, akeep%invp,         &
                          akeep, control, inform )
    END IF

    IF ( PRESENT( order ) ) order( 1 : n ) = ABS( order2( 1 : n ) )
    IF ( control%print_level > DEBUG_PRINT_LEVEL )                             &
         PRINT *, "order = ", order2( 1 : n )

490 CONTINUE
    inform%stat = st
    IF ( inform%stat /= 0 ) inform%flag = SSIDS_ERROR_ALLOCATION
    akeep%inform = inform
    CALL inform%print_flag( control, context )
    RETURN

    END SUBROUTINE analyse_precision

!-*-*-  G A L A H A D - S S I D S _  squash_topology  S U B R O U T I N E  -*-*-

    SUBROUTINE squash_topology( topology, control, st )

!  given an initial topology, modify it to squash any resources control
!  parameters tell us to ignore.

    IMPLICIT NONE
    TYPE( HW_numa_region ), DIMENSION( : ), ALLOCATABLE,                       &
                                            INTENT( INOUT ) :: topology
    TYPE( ssids_control_type ), INTENT( IN ) :: control
    INTEGER( KIND = ip_ ), INTENT( OUT ) :: st

!  local variables

    LOGICAL :: no_omp
    INTEGER( KIND = ip_ ) :: i, j, ngpu
    TYPE( HW_numa_region ), DIMENSION( : ), ALLOCATABLE :: new_topology

    st = 0

    no_omp = .TRUE.
!$  no_omp = .FALSE.

!  get rid of GPUs if we're not using them

    IF ( .NOT. control%use_gpu ) THEN
      DO i = 1, SIZE( topology )
        IF ( SIZE( topology( i )%gpus ) /= 0 ) THEN
          DEALLOCATE( topology( i )%gpus )
          ALLOCATE( topology( i )%gpus( 0 ), STAT = st )
          IF ( st /= 0 ) RETURN
        END IF
      END DO
    END IF

!  FIXME: One can envisage a sensible coexistence of both
!  no_omp=.true. AND control%ignore_numa=.false. ( e.g., choose the
!  "best" NUMA node, with the least utilised CPUs and/or GPUs... ).

    IF ( no_omp ) THEN
      ALLOCATE( new_topology( 1 ), STAT = st )
      IF ( st /= 0 ) RETURN
      new_topology( 1 )%nproc = 1

!  count resources to REALlocate

      ngpu = 0
      DO i = 1, SIZE( topology )
        ngpu = ngpu + SIZE( topology( i )%gpus )
      END DO

!  FIXME: if no_omp=.true. AND control%ignore_numa=.true.,
!  then take the "first" GPU ( whichever it might be ), only.
!  A combination not meant for production, only for testing!

      IF ( control%ignore_numa ) ngpu = min( ngpu, 1 )

!  store list of GPUs

      ALLOCATE( new_topology( 1 )%gpus( ngpu ), STAT = st )
      IF ( st /= 0 ) RETURN
      IF ( ngpu > 0 ) THEN
        IF ( control%ignore_numa ) THEN
          new_topology( 1 )%gpus( 1 ) = huge( new_topology( 1 )%gpus( 1 ) )
          DO i = 1, SIZE( topology )
            new_topology( 1 )%gpus( 1 ) =                                      &
               MIN( new_topology( 1 )%gpus( 1 ), MINVAL( topology( i )%gpus ) )
          END DO
        ELSE
          ngpu = 0
          DO i = 1, SIZE( topology )
            DO j = 1, SIZE( topology( i )%gpus )
              new_topology( 1 )%gpus( ngpu + j ) = topology( i )%gpus( j )
            END DO
            ngpu = ngpu + SIZE( topology( i )%gpus )
          END DO
        END IF
      END IF

!  move new_topology into place, deallocating old one

      DEALLOCATE( topology )
      CALL move_alloc( new_topology, topology )

!  squash everything to single NUMA region if we're ignoring numa

    ELSE IF ( SIZE( topology ) > 1 .AND. control%ignore_numa ) THEN
      ALLOCATE( new_topology( 1 ), STAT = st )
      IF ( st /= 0 ) RETURN

!  count resources to REALlocate

      new_topology( 1 )%nproc = 0
      ngpu = 0
      DO i = 1, SIZE( topology )
        new_topology( 1 )%nproc = new_topology( 1 )%nproc + topology( i )%nproc
        ngpu = ngpu + SIZE( topology( i )%gpus )
      END DO

!  store list of GPUs

      ALLOCATE( new_topology( 1 )%gpus( ngpu ), STAT = st )
      IF ( st /= 0 ) RETURN
      IF ( ngpu > 0 ) THEN
        ngpu = 0
        DO i = 1, SIZE( topology )
          DO j = 1, SIZE( topology( i )%gpus )
            new_topology( 1 )%gpus( ngpu + j ) = topology( i )%gpus( j )
          END DO
          ngpu = ngpu + SIZE( topology( i )%gpus )
        END DO
      END IF

!  move new_topology into place, deallocating old one

      DEALLOCATE( topology )
      CALL move_alloc( new_topology, topology )
    END IF
    RETURN

    END SUBROUTINE squash_topology

!-*-*-  G A L A H A D - S S I D S _  analyse_coord  S U B R O U T I N E  -*-*-

    SUBROUTINE ssids_analyse_coord_precision( n, ne, row, col, akeep, control, &
                                              inform, order, val, topology )

!  analyse phase: matrix entered in coordinate format.
!  matrix_util routine is used to convert the data to CSC format.
!  The user OPTIONALly inputs the pivot order. If not, metis called.
!  Structure is then expanded, supervariables are computed and then
!  the assembly tree is constructed and the data structures required
!  by the factorization are set up.

    IMPLICIT NONE
    INTEGER( KIND = ip_ ), INTENT( IN ) :: n !  order of A
    INTEGER( KIND = long_ ), INTENT( IN ) :: ne !  entries to be input by user
    INTEGER( KIND = ip_ ), INTENT( IN ) :: row( : ) !  row indices
    INTEGER( KIND = ip_ ), INTENT( IN ) :: col( : ) !  col indices

!  see derived-type declaration

    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep

!  see derived-type declaration

    TYPE( SSIDS_control_type ), INTENT( IN ) :: control

!  see derived-type declaration

    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  Must be present and set on entry if control%ordering = 0. If  i is used to 
!  index a variable, order(i) must hold its position in the pivot sequence.
!  If i is not used to index a variable, order(i) must be set to zero.
!  On exit, holds the pivot order to be used by factorization

    INTEGER( KIND = ip_ ), INTENT( INOUT ), OPTIONAL  :: order( : )

!  must be present if a matching-based elimination ordering is required
!  (control%ordering = 2). If present, val(k) must hold value of entry in 
!  row(k) and col(k)

    REAL( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: val( : )

!  user specified topology

    TYPE( HW_numa_region ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: topology

!  local variables

!  col ptrs and row indices for the expanded matrix

    INTEGER( KIND = long_ ), DIMENSION( : ), ALLOCATABLE :: ptr2
    INTEGER( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: row2

!  pivot order

    INTEGER( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: order2

    INTEGER( KIND = ip_ ) :: mo_flag

!  cleaned values if  val is present

    REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: val_clean

!  expanded matrix (if val is present)

    REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: val2
    CHARACTER( LEN = 50 )  :: context!  procedure name ( used when printing ).
    INTEGER( KIND = ip_ ) :: mu_flag  !  error flag for matrix_util routines
    INTEGER( KIND = long_ ) :: nz  !  entries in expanded matrix
!   INTEGER( KIND = ip_ ) :: flag  !  error flag for metis
    INTEGER( KIND = ip_ ) :: st    !  stat parameter
    INTEGER( KIND = ip_ ) :: free_flag
    TYPE( ssids_inform_type ) :: inform_default

!  initialise

    context = 'ssids_analyse_coord'
    inform = inform_default
    CALL ssids_free( akeep, free_flag )
!   IF ( free_flag /= 0 ) THEN
!      inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
!      inform%cuda_error = free_flag
!      akeep%inform = inform
!      CALL inform%print_flag( control, context )
!      RETURN
!   END IF

!  output status on entry

    CALL control%print_summary_analyse( context )
    IF ( control%print_level >= 1 .AND. control%unit_diagnostics >= 0 ) THEN
      WRITE ( control%unit_diagnostics,'( A, I15 )' )                          &
            ' n                         =  ', n
       WRITE ( control%unit_diagnostics,'( A, I15 )' )                         &
            ' ne                        =  ', ne
    END IF

    akeep%check = .true.
    akeep%n = n

!  checking of matrix data

    IF ( n < 0 .OR. ne < 0 ) THEN
      inform%flag = SSIDS_ERROR_A_N_OOR
      akeep%inform = inform
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    IF ( n == 0 ) THEN
      akeep%nnodes = 0

!  used to check if analyse has been run

      ALLOCATE( akeep%sptr( 0 ), STAT = st )
      IF ( st /= 0 ) GO TO 490
      akeep%inform = inform
      RETURN
    END IF

!  check control%ordering has a valid value

    IF ( control%ordering < 0 .OR. control%ordering > 2 ) THEN
      inform%flag = SSIDS_ERROR_ORDER
      akeep%inform = inform
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  check val present when expected

    IF ( control%ordering == 2 ) THEN
      IF ( .NOT. PRESENT( val ) ) THEN
        inform%flag = SSIDS_ERROR_VAL
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END IF
    END IF

    st = 0
    ALLOCATE( akeep%ptr( n + 1 ),STAT = st )
    IF ( st /= 0 ) GO TO 490

    IF ( PRESENT( val ) ) THEN
      CALL MU_convert_coord_to_cscl( SSIDS_MATRIX_REAL_SYM_INDEF, n, n, ne,    &
                                     row, col, akeep%ptr, akeep%row, mu_flag,  &
                                     val_in = val, val_out = val_clean,        &
                                     lmap = akeep%lmap, map = akeep%map,       &
                                     noor = inform%matrix_outrange,            &
                                     ndup = inform%matrix_dup )
    ELSE
      CALL MU_convert_coord_to_cscl( SSIDS_MATRIX_REAL_SYM_INDEF, n, n, ne,    &
                                     row, col, akeep%ptr, akeep%row, mu_flag,  &
                                     lmap = akeep%lmap, map = akeep%map,       &
                                     noor = inform%matrix_outrange,            &
                                     ndup = inform%matrix_dup )
    END IF

!  check for errors

    IF ( mu_flag < 0 ) THEN
      IF ( mu_flag == - 1 )  inform%flag = SSIDS_ERROR_ALLOCATION
      IF ( mu_flag == - 10 ) inform%flag = SSIDS_ERROR_A_ALL_OOR
      akeep%inform = inform
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  check whether warning needs to be raised

    IF ( mu_flag > 0 ) inform%flag = mu_flag
    nz = akeep%ptr( n + 1 ) - 1

!  if the pivot order is not supplied, we need to compute an order here, before
!   we expand the matrix structure. Otherwise, we must check the supplied order

    ALLOCATE( akeep%invp( n ), order2( n ), ptr2( n + 1 ), row2( 2 * nz ),     &
              STAT = st )
    IF ( st /= 0 ) GO TO 490
    IF ( control%ordering == 2 ) THEN
      ALLOCATE( val2( 2 * nz ), akeep%scaling( n ),STAT = st )
      IF ( st /= 0 ) GO TO 490
    END IF

    SELECT CASE( control%ordering )

!  user-supplied ordering

    CASE( 0 )

!  raise an error if the user claims but fails to supply the order

      IF ( .NOT. PRESENT( order ) ) THEN
        inform%flag = SSIDS_ERROR_ORDER
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END IF
      CALL check_order( n, order, akeep%invp, control, inform )
      IF ( inform%flag < 0 ) GO TO 490
      order2( 1 : n ) = order( 1 : n )
      CALL expand_pattern( n, nz, akeep%ptr, akeep%row, ptr2, row2 )

!  METIS ordering

    CASE( 1 )
      CALL NODEND_half_order( n, akeep%ptr, akeep%row, order2,                 &
                              control%nodend_control, inform%nodend_inform )
      inform%flag = inform%nodend_inform%status
      st = inform%nodend_inform%alloc_status
      IF ( inform%flag < 0 ) GO TO 490
      CALL expand_pattern( n, nz, akeep%ptr, akeep%row, ptr2, row2 )

!  matching-based ordering required

    CASE( 2 )
      CALL expand_matrix( n, nz, akeep%ptr, akeep%row, val_clean, ptr2, row2,  &
                          val2 )
      DEALLOCATE( val_clean,STAT = st )

      CALL MO_match_order_metis( n, ptr2, row2, val2, order2, akeep%scaling,   &
                                 control%nodend_control, inform%nodend_inform, &
                                 mo_flag, inform%stat )

      SELECT CASE( mo_flag )
      CASE( 0 ) ! success; do nothing
      CASE( 1 ) ! singularity warning required
        inform%flag = SSIDS_WARNING_ANALYSIS_SINGULAR
      CASE( - 1 )
        inform%flag = SSIDS_ERROR_ALLOCATION
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      CASE DEFAULT
        inform%flag = SSIDS_ERROR_UNKNOWN
        akeep%inform = inform
        CALL inform%print_flag( control, context )
        RETURN
      END SELECT

      DEALLOCATE( val2, STAT = st )
    END SELECT

!  assess topology: user supplied

    IF ( PRESENT( topology ) ) THEN
      ALLOCATE( akeep%topology( SIZE( topology ) ), STAT = st )
      IF ( st /= 0 ) GO TO 490
      akeep%topology( : ) = topology( : )

!  guess it

    ELSE
      CALL HW_guess_topology( akeep%topology, st )
      IF ( st /= 0 ) GO TO 490
    END IF

!  we now have the expanded structure held using ptr2, row2, proceed to the 
!  analyse phase

    CALL analyse_phase( n, akeep%ptr, akeep%row, ptr2, row2, order2,           &
                        akeep%invp, akeep, control, inform )
    IF ( inform%flag < 0 ) GO TO 490

    IF ( PRESENT( order ) ) order( 1 : n ) = ABS( order2( 1 : n ) )
    IF ( control%print_level > DEBUG_PRINT_LEVEL )                             &
         PRINT *, "order = ", order2( 1 : n )

490 CONTINUE
    inform%stat = st
    IF ( inform%stat /= 0 ) inform%flag = SSIDS_ERROR_ALLOCATION
    akeep%inform = inform
    CALL inform%print_flag( control, context )
    RETURN

    END SUBROUTINE ssids_analyse_coord_precision

!-*-*-  G A L A H A D - S S I D S _ factor_ptr32   S U B R O U T I N E  -*-*-

    SUBROUTINE ssids_factor_ptr32_precision( posdef, val, akeep, fkeep,        &
                                             control, inform, scale, ptr, row )

!  factorize phase: 32-bit wrapper around 64-bit version, NB ptr is non-OPTIONAL

    IMPLICIT NONE
    LOGICAL, INTENT( IN ) :: posdef
    REAL( KIND = rp_ ), DIMENSION( * ), target, INTENT( IN ) :: val
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    REAL( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: scale
    INTEGER( KIND = i4_ ), DIMENSION( akeep%n + 1 ), INTENT( IN ) :: ptr
    INTEGER( KIND = ip_ ), DIMENSION( * ), OPTIONAL, INTENT( IN ) :: row

!  local variables

    INTEGER( KIND = i8_ ), DIMENSION( : ), ALLOCATABLE :: ptr64

!  copy from 32-bit to 64-bit ptr

    ALLOCATE( ptr64( akeep%n + 1 ), stat=inform%stat )
    IF ( inform%stat /= 0 ) THEN
      inform%flag = SSIDS_ERROR_ALLOCATION
      CALL inform%print_flag( control, 'ssids_factor' )
      fkeep%inform = inform
      RETURN
    END IF
    ptr64( 1 : akeep%n + 1 ) = ptr( 1 : akeep%n + 1 )

!  call 64-bit routine

    CALL ssids_factor_ptr64_precision( posdef, val, akeep, fkeep, control,     &
                                       inform, scale = scale, ptr = ptr64,     &
                                       row = row )
    RETURN

    END SUBROUTINE ssids_factor_ptr32_precision

!-*-*-  G A L A H A D - S S I D S _ factor_ptr64  S U B R O U T I N E  -*-*-

    SUBROUTINE ssids_factor_ptr64_precision( posdef, val, akeep, fkeep,        &
                                             control, inform, scale, ptr, row )

!  factorize phase (64-bit pointers)

    IMPLICIT NONE
    LOGICAL, INTENT( IN ) :: posdef

 !  A values (lower triangle)

    REAL( KIND = rp_ ), DIMENSION( * ), target, INTENT( IN ) :: val
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  used to hold row and column scaling factors. Must be set on entry if
!  control%scaling <= 0.  Note: has to be assumed shape, not assumed size 
!  or fixed size to work around funny compiler bug

    REAL( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: scale

!  must be  present if, on call to analyse phase, check = .false., and must be
!  unchanged since that call

    INTEGER( KIND = i8_ ), DIMENSION( akeep%n + 1 ), OPTIONAL,                 &
                                                     INTENT( IN ) :: ptr

!  must be present if, on call to analyse phase, check = .false., and must be 
!  unchanged  since that call

    INTEGER( KIND = ip_ ), DIMENSION( * ), OPTIONAL, INTENT( IN ) :: row

!  local variables

    REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: val2
    CHARACTER( LEN = 50 ) :: context
    INTEGER( KIND = ip_ ) :: i, n, st
    INTEGER( KIND = i8_ ) :: nz

!  solve parameters. Tree is broken up into multiple chunks. Parent-child
!  relations between chunks are stored in fwd_ptr and fwd (see solve routine
!  comments)

    INTEGER( KIND = ip_ ) :: matrix_type
    REAL( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: scaling

!  types related to scaling routines

    TYPE( MS_hungarian_control_type ) :: hscontrol
    TYPE( MS_hungarian_inform_type ) :: hsinform
    TYPE( MS_equilib_control_type ) :: escontrol
    TYPE( MS_equilib_inform_type ) :: esinform

    TYPE( omp_settings ) :: user_omp_settings
    TYPE( RB_write_options ) :: rb_control
    INTEGER( KIND = ip_ ) :: flag

!  setup for any printing we may require

    context = 'ssids_factor'

!  print summary of input control (depending on print level etc)

    CALL control%print_summary_factor( posdef, context )

!  check for error in call sequence, specfically if analyse cannot has been run

    IF ( .NOT. ALLOCATED( akeep%sptr ) .OR. akeep%inform%flag < 0 ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      fkeep%inform = inform
      RETURN
    END IF

!  initialize

    inform = akeep%inform
    inform%num_factor = 0
    inform%num_flops = 0
    st = 0
    n = akeep%n

!  ensure OpenMP setup is as required

    CALL push_omp_settings( user_omp_settings, inform%flag )
    IF ( inform%flag < 0 ) THEN
       fkeep%inform = inform
       CALL inform%print_flag( control, context )
       RETURN
    END IF

!  immediate return if analyse detected singularity and control%action=false

    IF ( .NOT. control%action .AND. akeep%n /= akeep%inform%matrix_rank ) THEN
      inform%flag = SSIDS_ERROR_SINGULAR
      GO TO 100
    END IF

!  immediate return for trivial matrix

    IF ( akeep%nnodes == 0 ) THEN
      inform%flag = SSIDS_SUCCESS
      inform%matrix_rank = 0
      GO TO 100
    END IF

    fkeep%pos_def = posdef
    IF ( posdef ) THEN
      matrix_type = SSIDS_MATRIX_REAL_SYM_PSDEF
    ELSE
      matrix_type = SSIDS_MATRIX_REAL_SYM_INDEF
    END IF

!  if matrix has been checked, produce a clean version of val in val2

    IF ( akeep%check ) THEN
      nz = akeep%ptr( n + 1 ) - 1
      ALLOCATE( val2( nz ),STAT = st )
      IF ( st /= 0 ) GO TO 10
      CALL MU_apply_conversion_map( matrix_type, akeep%lmap, akeep%map, val,   &
                                    nz, val2 )
    ELSE

!  analyse run with no checking so must have ptr and row present

      IF ( .NOT. PRESENT( ptr ) ) inform%flag = SSIDS_ERROR_PTR_ROW
      IF ( .NOT. PRESENT( row ) ) inform%flag = SSIDS_ERROR_PTR_ROW
      IF ( inform%flag < 0 ) THEN
        fkeep%inform = inform
        GO TO 100
      END IF
    END IF

!  at this point, either (ptr, row, val) or ( akeep%ptr, akeep%row, val2)
!  hold the lower triangular part of A. Dump the matrix if required

    IF ( ALLOCATED( control%rb_dump ) ) THEN
      WRITE( control%unit_warning, * )                                         &
        "Dumping matrix to '", control%rb_dump, "'"
      IF ( akeep%check ) THEN
        CALL RB_write( control%rb_dump, SSIDS_MATRIX_REAL_SYM_INDEF, n, n,     &
                       akeep%ptr, akeep%row, rb_control, flag, val = val2 )
      ELSE
        CALL RB_write( control%rb_dump, SSIDS_MATRIX_REAL_SYM_INDEF, n, n,     &
                       ptr, row, rb_control, flag, val = val )
      END IF
      IF ( flag /= 0 ) THEN
        inform%flag = SSIDS_ERROR_UNKNOWN
        GO TO 100
      END IF
    END IF

!  perform scaling if required

    IF ( control%scaling > 0 .OR. PRESENT( scale ) ) THEN
      IF ( ALLOCATED( fkeep%scaling ) ) THEN
        IF ( SIZE( fkeep%scaling ) < n ) THEN
          DEALLOCATE( fkeep%scaling, STAT = st )
          ALLOCATE( fkeep%scaling( n ), STAT = st )
        END IF
      ELSE
        ALLOCATE( fkeep%scaling( n ), STAT = st )
      END IF
      IF ( st /= 0 ) GO TO 10
    ELSE
      DEALLOCATE( fkeep%scaling, STAT = st )
    END IF

    IF ( ALLOCATED( akeep%scaling ) .AND. control%scaling /= 3 ) THEN
      inform%flag = SSIDS_WARNING_MATCH_ORD_NO_SCALE
      CALL inform%print_flag( control, context )
    END IF

!  find the scaling

    SELECT CASE ( control%scaling )
    CASE( : 0 ) ! user supplied or NONE
      IF ( PRESENT( scale ) ) THEN
        DO i = 1, n
          fkeep%scaling( i ) = scale( akeep%invp( i ) )
        END DO
      END IF

    CASE( 1 ) ! matching-based scaling using the Hungarian algorithm

!  allocate space for scaling

      ALLOCATE( scaling( n ), STAT = st )
      IF ( st /= 0 ) GO TO 10

!  run the Hungarian algorithm

      hscontrol%scale_if_singular = control%action
      IF ( akeep%check ) THEN
        CALL MS_hungarian_scale_sym( n, akeep%ptr, akeep%row, val2, scaling,   &
                                     hscontrol, hsinform )
      ELSE
        CALL MS_hungarian_scale_sym( n, ptr, row, val, scaling,                &
                                     hscontrol, hsinform )
      END IF
      SELECT CASE( hsinform%flag )
      CASE( - 1 ) !  Allocation error
        st = hsinform%stat
        GO TO 10
      CASE( - 2 ) !  Structually singular matrix and control%action=.false.
        inform%flag = SSIDS_ERROR_SINGULAR
        GO TO 100
      END SELECT

!  permute scaling to correct order

      DO i = 1, n
        fkeep%scaling( i ) = scaling( akeep%invp( i ) )
      END DO

!  copy scaling(:) to user array scale(:) if present

      IF ( PRESENT( scale ) ) THEN
        scale( 1 : n ) = scaling( 1 : n )
      END IF

!  cleanup memory

      DEALLOCATE( scaling, STAT = st )

    CASE( 2 ) !  matching-based scaling by Auction Algorithm

!  allocate space for scaling

      ALLOCATE( scaling( n ), STAT = st )
      IF ( st /= 0 ) GO TO 10

!  run the auction algorithm

      IF ( akeep%check ) THEN
        CALL MS_auction_scale_sym( n, akeep%ptr, akeep%row, val2, scaling,     &
                                   control%auction, inform%auction )
      ELSE
        CALL MS_auction_scale_sym( n, ptr, row, val, scaling,                  &
                                   control%auction, inform%auction )
      END IF
      IF ( inform%auction%flag /= 0 ) THEN ! the allocation failed
        st = inform%auction%stat
        GO TO 10
      END IF

!  permute scaling to correct order

      DO i = 1, n
        fkeep%scaling( i ) = scaling( akeep%invp( i ) )
      END DO

!  copy scaling( : ) to user array scale( : ) if present

       IF ( PRESENT( scale ) ) THEN
         scale( 1 : n ) = scaling( 1 : n )
       END IF

!  cleanup memory

      DEALLOCATE( scaling, STAT = st )

    CASE( 3 ) ! scaling generated during analyse phase for matching-based order

!  no scaling saved from analyse phase

      IF ( .NOT. ALLOCATED( akeep%scaling ) ) THEN
        inform%flag = SSIDS_ERROR_NO_SAVED_SCALING
        GO TO 100
      END IF
      DO i = 1, n
        fkeep%scaling( i ) = akeep%scaling( akeep%invp( i ) )
      END DO

    CASE( 4 : ) ! norm equilibriation algorithm

!  allocate space for scaling

      ALLOCATE( scaling( n ), STAT = st )
      IF ( st /= 0 ) GO TO 10

!  run equilibriation algorithm

      IF ( akeep%check ) THEN
        CALL MS_equilib_scale_sym( n, akeep%ptr, akeep%row, val2, scaling,     &
                                   escontrol, esinform )
      ELSE
        CALL MS_equilib_scale_sym( n, ptr, row, val, scaling,                  &
                                   escontrol, esinform )
      END IF
      IF ( esinform%flag /= 0 ) THEN

!  only possible error is memory allocation failure

        st = esinform%stat
        GO TO 10
      END IF

!  permute scaling to correct order

      DO i = 1, n
        fkeep%scaling( i ) = scaling( akeep%invp( i ) )
      END DO

!  copy scaling(:) to user array scale(:) if present

      IF ( PRESENT( scale ) ) THEN
        DO i = 1, n
          scale( akeep%invp( i ) ) = fkeep%scaling( i )
        END DO
      END IF

!  cleanup memory

      DEALLOCATE( scaling, STAT = st )
    END SELECT

!   IF ( ALLOCATED( fkeep%scaling ) )                                          &
!     PRINT *, "minscale, maxscale = ", minval( fkeep%scaling ),               &
!        MAXVAL( fkeep%scaling )

!  setup data storage

    IF ( ALLOCATED( fkeep%subtree ) ) THEN
      DO i = 1, SIZE( fkeep%subtree )
        IF ( ASSOCIATED( fkeep%subtree( i )%ptr ) ) THEN
          CALL fkeep%subtree( i )%ptr%cleanup( )
          DEALLOCATE( fkeep%subtree( i )%ptr )
        END IF
      END DO
      DEALLOCATE( fkeep%subtree )
    END IF

!  Call the main factorization routine

    IF ( akeep%check ) THEN
!     WRITE( 6,* ) 'val2 = ', val2( :nz )
      CALL fkeep%inner_factor( akeep, val2, control, inform )
    ELSE
!     WRITE( 6,* ) 'val = ', val( :ptr( n + 1 )-1 )
      CALL fkeep%inner_factor( akeep, val, control, inform )
    END IF
    IF ( inform%flag < 0 ) THEN
      fkeep%inform = inform
      GO TO 100
    END IF

    IF ( akeep%n /= inform%matrix_rank ) THEN

!  rank deficient, if we reach this point then must be control%action=.true.

      IF ( control%action ) THEN
        inform%flag = SSIDS_WARNING_FACT_SINGULAR
      ELSE
        inform%flag = SSIDS_ERROR_SINGULAR
      END IF
      CALL inform%print_flag( control, context )
    END IF

!  if required, print diagnostics

    IF ( control%print_level >= 1 .AND. control%unit_diagnostics >= 0 ) THEN
      WRITE( control%unit_diagnostics,'( / A )' )                              &
           ' Completed factorisation with:'                                    
      WRITE( control%unit_diagnostics,                                         &
        '( A, 3( / A, I12 ), 2( / A, ES12.4 ), 5( / A, I12 ) )' )              &
        ' information parameters ( inform% ) :',                               &
        ' flag                   Error flag                               = ', &
        inform%flag,                                                           &
        ' maxfront               Maximum frontsize                        = ', &
        inform%maxfront,                                                       &
        ' maxsupernode           Maximum supernode size                   = ', &
        inform%maxsupernode,                                                   &
        ' num_factor             Number of entries in L                   = ', &
        REAL( inform%num_factor ),                                             &
        ' num_flops              Number of flops performed                = ', &
        REAL( inform%num_flops ),                                              &
        ' num_two                Number of 2x2 pivots used                = ', &
        inform%num_two,                                                        &
        ' num_delay              Number of delayed eliminations           = ', &
        inform%num_delay,                                                      &
        ' rank                   Computed rank                            = ', &
        inform%matrix_rank,                                                    &
        ' num_neg                Computed number of negative eigenvalues  = ', &
        inform%num_neg
    END IF

!  normal return just drops through

100 CONTINUE

!  clean up and return

    fkeep%inform = inform
    CALL inform%print_flag( control, context )
    CALL pop_omp_settings( user_omp_settings )
    RETURN

!  error handling

 10 CONTINUE
    inform%flag = SSIDS_ERROR_ALLOCATION
    inform%stat = st
    GO TO 100

    END SUBROUTINE ssids_factor_ptr64_precision

!-*-*-  G A L A H A D - S S I D S _ s o l v e _ o n e   S U B R O U T I N E  -*-

    SUBROUTINE ssids_solve_one_precision( x1, akeep, fkeep, control, inform,   &
                                          job )

!  solve phase single x

    IMPLICIT NONE

!  On entry, x must be set so that if i has been used to index a variable,
!  x(i) is the corresponding component of the right-hand side.On exit, if i
!  has been used to index a variable, x(i) holds solution for variable i

    REAL( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: x1
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    INTEGER( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: job

!  local variables

    INTEGER( KIND = ip_ ) :: ldx

    ldx = SIZE( x1 )
    IF ( PRESENT( job ) ) THEN
      CALL ssids_solve_mult_precision( 1_ip_, x1, ldx, akeep, fkeep, control,  &
                                       inform, job )
    ELSE
      CALL ssids_solve_mult_precision( 1_ip_, x1, ldx, akeep, fkeep, control,  &
                                       inform )
    END IF
    RETURN

    END SUBROUTINE ssids_solve_one_precision

!-*-  G A L A H A D - S S I D S _ s o l v e _ m u l t  S U B R O U T I N E  -*-

    SUBROUTINE ssids_solve_mult_precision( nrhs, x, ldx, akeep, fkeep,         &
                                           control, inform, job )

!  solve phase multiple x

    IMPLICIT NONE
    INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
    INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx

!  on entry, x must be set so that if i has been used to index a variable, 
!  x(i,j) is the corresponding component of the right-hand side for the jth 
!  system (j = 1,2,..., nrhs). On exit, if i has been used to index a variable,
!  x( i,j ) holds solution for variable i to system j

    REAL( KIND = rp_ ), DIMENSION( ldx,nrhs ), INTENT( INOUT ), target :: x
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep

!  for details of keep, control, inform : see derived type description

    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep !inout for moving data
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  job is used to indicate whether a  partial solution required
!  job = 1 : forward eliminations only (PLX = B)
!  job = 2 : diagonal solve (DX = B) (indefinite case only)
!  job = 3 : backsubs only ((PL)^TX = B)
!  job = 4 : diag and backsubs (D(PL)^TX = B) (indefinite case only)
!  job absent: complete solve performed

    INTEGER( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: job 

!  local variables

    CHARACTER( LEN = 50 )  :: context  !  Procedure name ( used when printing ).
    INTEGER( KIND = ip_ ) :: local_job !  local job parameter
    INTEGER( KIND = ip_ ) :: n

    context = 'ssids_solve'
    inform%flag = SSIDS_SUCCESS

!  perform appropriate printing

    IF ( control%print_level >= 1 .AND. control%unit_diagnostics >= 0 ) THEN
      WRITE( control%unit_diagnostics,'( //a )' ) &
           ' Entering ssids_solve with:'
      WRITE( control%unit_diagnostics, '( A, 4( / A, I12 ),( / A, I12 ) )' )   &
           ' control parameters ( control% ) :',                               &
           ' print_level         Level of diagnostic printing        = ',      &
           control%print_level,                                                &
           ' unit_diagnostics    Unit for diagnostics                = ',      &
           control%unit_diagnostics,                                           &
           ' unit_error          Unit for errors                     = ',      &
           control%unit_error,                                                 &
           ' unit_warning        Unit for warnings                   = ',      &
           control%unit_warning,                                               &
           ' nrhs                                                    = ', nrhs
      IF ( nrhs > 1 ) WRITE( control%unit_diagnostics, '( / A, I12 )' )        &
           ' ldx                                                     = ', ldx
    END IF

    IF ( akeep%nnodes == 0 ) RETURN

!  factorize phase has not been performed

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    inform%flag = MAX( SSIDS_SUCCESS, fkeep%inform%flag ) !  Preserve warnings

!  immediate return if already had an error

    IF ( akeep%inform%flag < 0 .OR. fkeep%inform%flag < 0 ) THEN
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       RETURN
    END IF

    n = akeep%n
    IF ( ldx < n ) THEN
      inform%flag = SSIDS_ERROR_X_SIZE
      CALL inform%print_flag( control, context )
      IF ( control%print_level >= 0 .AND. control%unit_error > 0 )             &
        WRITE( control%unit_error,'( A, I8, A, I8 )' )                         &
           ' Increase ldx from ', ldx, ' to at least ', n
      RETURN
    END IF

    IF ( nrhs < 1 ) THEN
      inform%flag = SSIDS_ERROR_X_SIZE
      CALL inform%print_flag( control, context )
      IF ( control%print_level >= 0 .AND. control%unit_error > 0 )             &
        WRITE ( control%unit_error, '( A, I8, A, I8 )' )                       &
                ' nrhs must be at least 1. nrhs = ', nrhs
       RETURN
    END IF

!  copy previous phases' inform data from akeep and fkeep
 
    inform = fkeep%inform

!  set local_job

    local_job = 0
    IF ( PRESENT( job ) ) THEN
      IF ( job < SSIDS_SOLVE_JOB_FWD .OR. job > SSIDS_SOLVE_JOB_DIAG_BWD )     &
           inform%flag = SSIDS_ERROR_JOB_OOR
      IF ( fkeep%pos_def .AND. job == SSIDS_SOLVE_JOB_DIAG )                   &
           inform%flag = SSIDS_ERROR_JOB_OOR
      IF ( fkeep%pos_def .AND. job == SSIDS_SOLVE_JOB_DIAG_BWD )               &
           inform%flag = SSIDS_ERROR_JOB_OOR
      IF ( inform%flag == SSIDS_ERROR_JOB_OOR ) THEN
        CALL inform%print_flag( control, context )
        RETURN
      END IF
      local_job = job
    END IF

    CALL fkeep%inner_solve( local_job, nrhs, x, ldx, akeep, inform )
    CALL inform%print_flag( control, context )
    RETURN

    END SUBROUTINE ssids_solve_mult_precision

!-*-*-  G A L A H A D - S S I D S _  enquire_posdef  S U B R O U T I N E  -*-*-

    SUBROUTINE ssids_enquire_posdef_precision( akeep, fkeep, control,          &
                                               inform, d )

!  return diagonal entries to user

    IMPLICIT NONE
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), target, INTENT( IN ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    REAL( KIND = rp_ ), DIMENSION( * ), INTENT( OUT ) :: d

!  local variables

    CHARACTER( LEN = 50 ) :: context  ! procedure name ( used when printing )

    context = 'ssids_enquire_posdef'
    inform%flag = SSIDS_SUCCESS

!  factorize phase has not been performed

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  immediate return if there had already been an error

    IF ( akeep%inform%flag < 0 .OR. fkeep%inform%flag < 0 ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    IF ( .NOT. fkeep%pos_def ) THEN
      inform%flag = SSIDS_ERROR_NOT_LLT
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    CALL fkeep%enquire_posdef( akeep, d )
    CALL inform%print_flag( control, context )
    RETURN

    END SUBROUTINE ssids_enquire_posdef_precision

!-*-*-  G A L A H A D - S S I D S _ enquire_indef S U B R O U T I N E  -*-*-

    SUBROUTINE ssids_enquire_indef_precision( akeep, fkeep, control, inform,   &
                                              piv_order, d )

!  In the indefinite case, the pivot sequence used will not necessarily be
!  the same as that passed to ssids_factor ( because of delayed pivots ). 
!  This SUBROUTINE allows the user to obtain the pivot sequence that was
!  actually used. also the entries of D^{-1} are RETURNed using array d.

    IMPLICIT NONE
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), target, INTENT( IN ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  if i is used to index a variable, its position in the pivot sequence
!  will be placed in piv_order(i), with its sign negative if it is
!  part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.

    INTEGER( KIND = ip_ ), DIMENSION( * ), OPTIONAL, INTENT( OUT ) :: piv_order

!  the diagonal entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
!  entries will be placed in d( 2,: ). The entries are held in pivot order.

    REAL( KIND = rp_ ), DIMENSION( 2,* ), OPTIONAL, INTENT( OUT ) :: d

!  local variables

    CHARACTER( LEN = 50 )  :: context  !  Procedure name ( used when printing ).
    INTEGER( KIND = ip_ ) :: i, po

    context = 'ssids_enquire_indef'
    inform%flag = SSIDS_SUCCESS

!  check if factorize phase has been performed

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  immediate return if there has been an error

    IF ( akeep%inform%flag < 0 .OR. fkeep%inform%flag < 0 ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    IF ( fkeep%pos_def ) THEN
      inform%flag = SSIDS_ERROR_NOT_LDLT
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    CALL fkeep%enquire_indef( akeep, inform, piv_order, d )

!  bug fix to give 1-based indices

    IF ( PRESENT( piv_order ) ) THEN
!     WRITE( 6,"( ' ssids: piv_order ', 7I6 )" ) piv_order( : akeep%n )

!  bug fix to determine what a C 0 index means

      IF ( MOD( COUNT( piv_order( : akeep%n ) < 0 ), 2 ) == 0 ) THEN
        DO i = 1, akeep%n
          po = piv_order( i )
          IF ( po >= 0 ) THEN 
            piv_order( i ) = po + 1
          ELSE
            piv_order( i ) = po - 1
          END IF
        END DO
      ELSE
        DO i = 1, akeep%n
          po = piv_order( i )
          IF ( po > 0 ) THEN
            piv_order( i ) = po + 1
          ELSE
            piv_order( i ) = po - 1
          END IF
        END DO
      END IF
!     WRITE( 6,"( ' ssids: revised piv_order ', 7I6 )" ) piv_order( : akeep%n )
    END IF
    CALL inform%print_flag( control, context )
    RETURN

    END SUBROUTINE ssids_enquire_indef_precision

!-*-*-*-  G A L A H A D - S S I D S _  a l t e r  S U B R O U T I N E  -*-*-*-

    SUBROUTINE ssids_alter_precision( d, akeep, fkeep, control, inform )

!  in the indefinite case, change the entries of D^{-1}

    IMPLICIT NONE

!  the required diagonal entries of D^{-1} must be placed in d(1,i) (i = 1,...n)
!  and the off-diagonal entries must be placed in d(2,i) (i = 1,...n-1)

    REAL( KIND = rp_ ), DIMENSION( 2, * ), INTENT( IN ) :: d
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), target, INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  local variables

    CHARACTER( LEN = 50 )  :: context  ! procedure name (used when printing)

    context = 'ssids_alter'
    inform%flag = SSIDS_SUCCESS

!  factorize phase has not been performed

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
      inform%flag = SSIDS_ERROR_CALL_SEQUENCE
      CALL inform%print_flag( control, context )
      RETURN
    END IF

!  immediate return if there had been already an error

    IF ( akeep%inform%flag < 0 .OR. fkeep%inform%flag < 0 ) THEN
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       RETURN
    END IF

    IF ( fkeep%pos_def ) THEN
      inform%flag = SSIDS_ERROR_NOT_LDLT
      CALL inform%print_flag( control, context )
      RETURN
    END IF

    CALL fkeep%alter( d, akeep )
    CALL inform%print_flag( control, context )
    RETURN

    END SUBROUTINE ssids_alter_precision

!-*-*-  G A L A H A D - S S I D S _ f r e e _a k e e p  S U B R O U T I N E  -*-

    SUBROUTINE free_akeep_precision( akeep, flag )
    IMPLICIT NONE
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    INTEGER( KIND = ip_ ), INTENT( OUT ) :: flag

    CALL akeep%free( flag )
    RETURN

    END SUBROUTINE free_akeep_precision

!-*-*-  G A L A H A D - S S I D S _ f r e e _f k e e p  S U B R O U T I N E  -*-

    SUBROUTINE free_fkeep_precision( fkeep, cuda_error )
    IMPLICIT NONE
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    INTEGER( KIND = ip_ ), INTENT( OUT ) :: cuda_error

    CALL fkeep%free( cuda_error )
    RETURN

    END SUBROUTINE free_fkeep_precision

!-*-*-  G A L A H A D - S S I D S _ f r e e _ b o t h  S U B R O U T I N E  -*-

    SUBROUTINE free_both_precision( akeep, fkeep, cuda_error )
    IMPLICIT NONE
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    INTEGER( KIND = ip_ ), INTENT( OUT ) :: cuda_error

!  must free fkeep first as it may reference akeep

    CALL free_fkeep_precision( fkeep, cuda_error )
    IF ( cuda_error /= 0 ) RETURN
    CALL free_akeep_precision( akeep, cuda_error )
    RETURN

    END SUBROUTINE free_both_precision

!-*-*-  G A L A H A D - S S I D S _ push_omp_settings S U B R O U T I N E  -*-*-

    SUBROUTINE push_omp_settings( user_settings, flag )

!  ensure OpenMP ICVs are as required, and store user versions for

    IMPLICIT NONE
    TYPE( omp_settings ), INTENT( OUT ) :: user_settings
    INTEGER( KIND = ip_ ), INTENT( INOUT ) :: flag
!   CHARACTER( LEN = 255 ) :: OMP_CANCELLATION_VAR
!   LOGICAL :: ompgc
!   CALL get_environment_variable( "OMP_CANCELLATION", OMP_CANCELLATION_VAR )
!   WRITE ( *,* ) 'OMP_CANCELLATION = ', TRIM( OMP_CANCELLATION_VAR )
!!$  ompgc = omp_get_cancellation( )
!!$  WRITE( 6,* ) ' ompgc ', ompgc
!  dummy, for now.
!   user_settings%nested = .TRUE.
    user_settings%max_active_levels = HUGE( user_settings%max_active_levels )

!$  !  issue an error if we don't have cancellation ( could lead to segfaults )
!$  IF ( .NOT. omp_get_cancellation( ) ) THEN
!$     flag = SSIDS_ERROR_OMP_CANCELLATION
!$     RETURN
!$  END IF

!$  !  issue a warning if proc_bind is not enabled
!$  IF ( omp_get_proc_bind( ) == OMP_PROC_BIND_FALSE ) &
!$       flag = SSIDS_WARNING_OMP_PROC_BIND

!!$  !  must have nested enabled
!!$  user_settings%nested = omp_get_nested( )
!!$  IF ( .NOT. user_settings%nested ) CALL omp_set_nested( .true. )

!$  !  we need OMP_DYNAMIC to be unset, to guarantee the number of threads
!$  user_settings%dynamic = omp_get_dynamic( )
!$  IF ( user_settings%dynamic ) CALL omp_set_dynamic( .false. )

!$  !  we will need at least 2 active levels
!$  user_settings%max_active_levels = omp_get_max_active_levels( )
!$  IF ( user_settings%max_active_levels < 2 ) CALL omp_set_max_active_levels(2)
    RETURN

    END SUBROUTINE push_omp_settings

!-*-*-  G A L A H A D - S S I D S _  pop_omp_settings S U B R O U T I N E  -*-*-

    SUBROUTINE pop_omp_settings( user_settings )

!  restore user OpenMP ICV values.

    IMPLICIT NONE
    TYPE( omp_settings ), INTENT( IN ) :: user_settings

!!$  IF ( .NOT. user_settings%nested ) CALL omp_set_nested(user_settings%nested)
!$  IF ( user_settings%dynamic ) CALL omp_set_dynamic( user_settings%dynamic )
!$  IF ( user_settings%max_active_levels < 2 ) &
!$       CALL omp_set_max_active_levels( int( user_settings%max_active_levels ))
    RETURN

    END SUBROUTINE pop_omp_settings

!   ============================================================================
!   =================== extracted from SSIDS_ANALYSE module ====================
!   ============================================================================

!- G A L A H A D -  S S I D S _ a n a l y s e _ p h a s e  S U B R O U T I N E -

    SUBROUTINE analyse_phase( n, ptr, row, ptr2, row2, order, invp,            &
                              akeep, control, inform  )

!  calls core analyse & builds data strucutres

!  this routine requires the LOWER and UPPER triangular parts of A to be held 
!  in CSC format using ptr2 and row2  and lower triangular part held using ptr
!  and row
!
!  on exit from this routine, order is set to order input to factorization

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

    CHARACTER( 50 )  :: context !  Procedure name ( used when printing ).
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

      IF ( device == 0 ) THEN !  CPU

!print  * , numa_region, "init cpu subtree ", i, akeep%part( i ), &
!    akeep%part( i + 1 ) - 1

        akeep%subtree( i )%ptr => construct_cpu_symbolic_subtree( akeep%n,     &
          akeep%part( i ), akeep%part( i + 1 ), akeep%sptr, akeep%sparent,     &
          akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist,                    &
          contrib_dest( akeep%contrib_ptr( i ) :                               &
                        akeep%contrib_ptr( i + 1 ) - 1 ), control )

      ELSE !  GPU
        device = akeep%topology( numa_region )%gpus( device )

!print  * , numa_region, "init gpu subtree ", i, akeep%part( i ), &
!    akeep%part( i + 1 ) - 1, "device", device

!  return a dummy GPU call in lieu of a proper implementation

        akeep%subtree( i )%ptr => NULL( )

!       akeep%subtree( i )%ptr => construct_gpu_symbolic_subtree( device,      &
!         akeep%n, akeep%part( i ), akeep%part( i + 1 ), akeep%sptr,           &
!         akeep%sparent, akeep%rptr, akeep%rlist, akeep%nptr,                  &
!         akeep%nlist, control )
!       akeep%subtree( i )%ptr => dummy
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

!-*-  G A L A H A D -  S S I D S _ c h e c k _ o r d e r  S U B R O U T I N E -*

    SUBROUTINE check_order( n, order, invp, control, inform )

!  check order is a valid permutation

!  this routine requires the LOWER triangular part of A to be held in CSC 
!  format. The user has supplied a pivot order and this routine checks it 
!  is OK and returns an error if not. Also sets perm, invp.

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
    RETURN

    END SUBROUTINE check_order

!  G A L A H A D -  S S I D S _ e x p a n d _ p a t t e r n  S U B R O U T I N E

    SUBROUTINE expand_pattern( n, nz, ptr, row, aptr, arow )

!  specialised half->full matrix conversion

!  given lower triangular part of A held in row and ptr, expand to
!  upper and lower triangular parts (pattern only). No checks
!
!  note: we do not use half_to_full here to expand A since, if we did, we would
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
    RETURN

    END SUBROUTINE expand_pattern

!-  G A L A H A D -  S S I D S _ e x p a n d _ m a t r i x  S U B R O U T I N E 

    SUBROUTINE expand_matrix( n, nz, ptr, row, val, aptr, arow, aval )

!  specialised half->full matrix conversion

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
    RETURN

    END SUBROUTINE expand_matrix

!-*-  G A L A H A D -  S S I D S _ c o m p u t e _ f l o p s  F U N C T I O N -*

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

!-  G A L A H A D -  S S I D S _ find_subtree_partition  F U N C T I O N -

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
    RETURN

    END SUBROUTINE find_subtree_partition

!-  G A L A H A D -  S S I D S _ calc_exec_alloc  F U N C T I O N -

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
    RETURN

    END FUNCTION calc_exec_alloc

!-*-  G A L A H A D -  S S I D S _ s p l i t _ t r e e  S U B R O U T I N E -*-

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
    RETURN

    END SUBROUTINE split_tree

!-  G A L A H A D -  S S I D S _ create_size_order  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE create_size_order

!-  G A L A H A D -  S S I D S _ p r i n t _ a t r e e  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE print_atree

!-  G A L A H A D -  S S I D S _ print_atree_part  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE print_atree_part

!-  G A L A H A D -  S S I D S _ b u i l d _ m a p   S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE build_map

!   ============================================================================
!   ================ extracted from SPRAL_CORE_ANALYSE module ==================
!   ============================================================================

!-  G A L A H A D -  S S I D S _ b a s i c _ a n a l y s e  S U B R O U T I N E 

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

!-*-  G A L A H A D -  S S I D S _ f i n d _ e t r e e  S U B R O U T I N E -*-

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
    RETURN

    END SUBROUTINE find_etree

!-  G A L A H A D -  S S I D S _ find_postorder  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE find_postorder

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Column count routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-  G A L A H A D -  S S I D S _  find_col_counts  S U B R O U T I N E -

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
    RETURN

    END FUNCTION FIND

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Supernode amalgamation routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-  G A L A H A D -  S S I D S _ find_supernodes  S U B R O U T I N E -

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

!-  G A L A H A D -  S S I D S _ sort_by_val S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE sort_by_val

!-  G A L A H A D -  S S I D S _ sort_by_val_ms  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE sort_by_val_ms

!-*-  G A L A H A D -  S S I D S _ d o _ m e r g e  F U C T T I O N -*-

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
    RETURN

    END FUNCTION do_merge

!-  G A L A H A D -  S S I D S _ merge_nodes  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE merge_nodes

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Statistics routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-  G A L A H A D -  S S I D S _  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE calc_stats

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Row list routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-  G A L A H A D -  S S I D S _ find_row_lists  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE find_row_lists

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Assorted auxilary routines
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!-  G A L A H A D -  S S I D S _  dbl_tr_sort  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE dbl_tr_sort

!-  G A L A H A D -  S S I D S _  apply_perm  S U B R O U T I N E -

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
    RETURN

    END SUBROUTINE apply_perm

!   ============================================================================
!   =================== extracted from SSIDS_AKEEP module ======================
!   ============================================================================
!   ============= define ssids_akeep type and associated procedures ============
!   ============================================================================

!-*-  G A L A H A D -  S S I D S _ F R E E _ A K E E P  S U B R O U T I N E  -*-

    SUBROUTINE free_akeep( akeep, flag )

!  free all allocated components of the type akeep

    CLASS( ssids_akeep_type ), INTENT( INOUT ) :: akeep
    INTEGER( ip_ ), INTENT( OUT ) :: flag

!  local variables

    INTEGER( ip_ ) :: i, st

    flag = 0

    DEALLOCATE( akeep%part, STAT = st )
    IF ( ALLOCATED( akeep%subtree ) ) THEN
      DO i = 1, SIZE( akeep%subtree )
        IF ( ASSOCIATED( akeep%subtree( i )%ptr ) ) THEN
          CALL akeep%subtree( i )%ptr%cleanup(  )
          DEALLOCATE( akeep%subtree( i )%ptr )
          NULLIFY( akeep%subtree( i )%ptr )
        END IF
      END DO
      DEALLOCATE( akeep%subtree, STAT = st )
    END IF
    DEALLOCATE( akeep%contrib_ptr, akeep%contrib_idx, akeep%invp,              &
                akeep%nlist, akeep%nptr,  akeep%rlist, akeep%rptr,             &
                akeep%sparent, akeep%sptr, akeep%ptr, akeep%row,               &
                akeep%map, akeep%scaling, akeep%topology, STAT = st )
    RETURN

    END SUBROUTINE free_akeep

!   ============================================================================
!   =================== extracted from SSIDS_FKEEP module ======================
!   ============================================================================
!   ====== define ssids_fkeep type and associated procedures (CPU version) =====
!   ============================================================================

!-*-  G A L A H A D -  S S I D S _inner _ factor _ cpu  S U B R O U T I N E  -*-

     SUBROUTINE inner_factor_cpu( fkeep, akeep, val, control, inform )
     IMPLICIT none
     TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
     CLASS( SSIDS_fkeep_type ), TARGET, INTENT( INOUT ) :: fkeep
     REAL( KIND = rp_ ), DIMENSION( * ), TARGET, INTENT( IN ) :: val
     TYPE( SSIDS_control_type ), INTENT( IN ) :: control
     TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER( KIND = ip_ ) :: i, numa_region, exec_loc, my_loc
     INTEGER( KIND = ip_ ) :: total_threads, max_gpus, to_launch, thread_num
     INTEGER( KIND = ip_ ) :: nth ! Number of threads within a region
     INTEGER( KIND = ip_ ) :: ngpus ! Number of GPUs in a given NUMA region
     LOGICAL :: abort, all_region
     TYPE( contrib_type ), DIMENSION( : ), ALLOCATABLE :: child_contrib
     TYPE( ssids_inform_type ), DIMENSION( : ), ALLOCATABLE :: thread_inform

!  begin profile trace (noop if not enabled)

#ifdef PROFILE
     CALL profile_begin( akeep%topology ) 
#endif

!  allocate space for subtrees

     ALLOCATE( fkeep%subtree( akeep%nparts ), STAT = inform%stat )
     IF ( inform%stat /= 0 ) GO TO 200

!  determine resources

     total_threads = 0
     max_gpus = 0
     DO i = 1, SIZE( akeep%topology )
       total_threads = total_threads + akeep%topology( i )%nproc
       max_gpus = MAX( max_gpus, SIZE( akeep%topology( i )%gpus ) )
     END DO

     ! CALL subtree factor routines
     allocate( child_contrib( akeep%nparts ), STAT = inform%stat )
     IF ( inform%stat /= 0 ) GO TO 200

!  split into numa regions; parallelism within a region is responsibility
!  of subtrees

     to_launch = SIZE( akeep%topology ) * ( 1 + max_gpus )
     ALLOCATE( thread_inform( to_launch ), STAT = inform%stat )
     IF ( inform%stat /= 0 ) GO TO 200
     all_region = .FALSE.

!$omp parallel proc_bind( spread ) num_threads( to_launch )                    &
!$omp    default( none )                                                       &
!$omp    private( abort, i, exec_loc, numa_region, my_loc, thread_num )        &
!$omp    private( nth, ngpus )                                                 &
!$omp    shared( akeep, fkeep, val, control, thread_inform, child_contrib,     &
!$omp           all_region )                                                   &
!$omp    if ( to_launch.gt.1 )

     thread_num = 0
!$  thread_num = omp_get_thread_num( )
     numa_region = MOD( thread_num, SIZE( akeep%topology ) ) + 1
     my_loc = thread_num + 1
     IF ( thread_num < SIZE( akeep%topology ) ) THEN
       ngpus = SIZE( akeep%topology( numa_region )%gpus, 1 )

!  CPU, control number of inner threads (not needed for gpu)

       nth = akeep%topology( numa_region )%nproc
!      nth = nth - ngpus
     ELSE
       nth = 1
     END IF

!$   CALL omp_set_num_threads( int( nth ) )

!  split into threads for this NUMA region (unless we're running a GPU)

     exec_loc = - 1 ! avoid compiler warning re uninitialized
     abort = .FALSE.

!$omp parallel proc_bind( close ) default( shared ) &
!$omp    num_threads( nth ) &
!$omp    if ( my_loc.le.size( akeep%topology ) )

!$omp single
!$omp taskgroup

     DO i = 1, akeep%nparts
       exec_loc = akeep%subtree( i )%exec_loc
       IF ( numa_region == 1 .AND. exec_loc == -1 ) all_region = .TRUE.
       IF ( exec_loc /= my_loc ) CYCLE

!$omp task untied default( shared ) firstprivate( i, exec_loc ) &
!$omp    if ( my_loc.le.size( akeep%topology ) )
       IF ( abort ) GO TO 10
       IF ( ALLOCATED( fkeep%scaling ) ) THEN
         fkeep%subtree( i )%ptr                                                &
           => akeep%subtree( i )%ptr%factor( fkeep%pos_def, val,               &
                child_contrib( akeep%contrib_ptr( i ) :                        &
                               akeep%contrib_ptr( i + 1 ) - 1 ),               &
                control, thread_inform( my_loc ), scaling = fkeep%scaling )
       ELSE
         fkeep%subtree( i )%ptr                                                &
           => akeep%subtree( i )%ptr%factor( fkeep%pos_def, val,               &
                child_contrib( akeep%contrib_ptr( i ) :                        &
                               akeep%contrib_ptr( i + 1 ) - 1 ),               &
                control, thread_inform( my_loc ) )
       END IF
       IF ( thread_inform( my_loc )%flag < 0 ) THEN
         abort = .TRUE.
         GO TO 10
! !$omp    cancel taskgroup
       END IF

 ! there is a parent subtree to contribute to

       IF ( akeep%contrib_idx( i ) <= akeep%nparts ) THEN
         child_contrib( akeep%contrib_idx( i ) )                               &
            = fkeep%subtree( i )%ptr%get_contrib(  )
!$omp    flush
         child_contrib( akeep%contrib_idx( i ) )%ready = .TRUE.
       END IF
 10    CONTINUE ! jump target for abort
!$omp end task
     END DO

!$omp end taskgroup
!$omp end single
!$omp end parallel
!$omp end parallel

     DO i = 1, SIZE( thread_inform )
       CALL inform%reduce( thread_inform( i ) )
     END DO
     IF ( inform%flag < 0 ) GO TO 100 ! cleanup and exit

!  at least some all region subtrees exist

     IF ( all_region ) THEN
#ifdef PROFILE
       CALL profile_add_event( "EV_ALL_REGIONS",                               &
                               "Starting processing root subtree", 0 )
#endif

!$omp parallel num_threads( total_threads ) default( shared )
!$omp single
       DO i = 1, akeep%nparts
         exec_loc = akeep%subtree( i )%exec_loc
         IF ( exec_loc /= - 1 ) CYCLE
         IF ( ALLOCATED( fkeep%scaling ) ) THEN
           fkeep%subtree( i )%ptr                                              &
             => akeep%subtree( i )%ptr%factor( fkeep%pos_def, val,             &
                  child_contrib( akeep%contrib_ptr( i ) :                      &
                                 akeep%contrib_ptr( i + 1 ) - 1 ),             &
                  control, inform, scaling=fkeep%scaling )
         ELSE
           fkeep%subtree( i )%ptr                                              &
             => akeep%subtree( i )%ptr%factor( fkeep%pos_def, val,             &
                  child_contrib( akeep%contrib_ptr( i ) :                      &
                                 akeep%contrib_ptr( i + 1 ) - 1 ),             &
                  control, inform )
         END IF
         IF ( akeep%contrib_idx( i ) > akeep%nparts ) CYCLE ! part is a root
         child_contrib( akeep%contrib_idx( i ) )                               &
           = fkeep%subtree( i )%ptr%get_contrib( )
!$omp    flush
         child_contrib( akeep%contrib_idx( i ) )%ready = .TRUE.
       END DO
!$omp end single
!$omp end parallel
     END IF

!  cleanup and exit

 100 CONTINUE

!  end profile trace (noop if not enabled)
#ifdef PROFILE
     CALL profile_end( )
#endif
     RETURN

 200 CONTINUE
     inform%flag = SSIDS_ERROR_ALLOCATION
     GO TO 100 ! cleanup and exit
     RETURN

     END SUBROUTINE inner_factor_cpu

!-*-  G A L A H A D -  S S I D S _ inner _ solve _ cpu  S U B R O U T I N E  -*-

     SUBROUTINE inner_solve_cpu( local_job, nrhs, x, ldx, akeep, fkeep, inform )
     TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
     CLASS( ssids_fkeep_type ), INTENT( INOUT ) :: fkeep
     INTEGER( KIND = ip_ ), INTENT( INOUT ) :: local_job
     INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
     INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx
     REAL( KIND = rp_ ), DIMENSION( ldx, nrhs ), TARGET, INTENT( INOUT ) :: x
     TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER( KIND = ip_ ) :: i, r, part
     INTEGER( KIND = ip_ ) :: n
     REAL( KIND = rp_ ), DIMENSION( : , : ), ALLOCATABLE :: x2

     n = akeep%n

     ALLOCATE( x2( n, nrhs ), STAT = inform%stat )
     IF ( inform%stat /= 0 ) GO TO 100

!  permute/scale

     IF ( ALLOCATED( fkeep%scaling ) .AND.                                     &
           ( local_job == SSIDS_SOLVE_JOB_ALL .OR.                             &
             local_job == SSIDS_SOLVE_JOB_FWD ) ) THEN

!  copy and scale

       DO r = 1, nrhs
         DO i = 1, n
           x2( i, r ) = x( akeep%invp( i ), r ) * fkeep%scaling( i )
         END DO
       END DO

!  just copy

     ELSE
       DO r = 1, nrhs
         x2( 1 : n, r ) = x( akeep%invp( 1 : n ), r )
       END DO
     END if

!  perform relevant solves

     IF ( local_job == SSIDS_SOLVE_JOB_FWD .OR.                                &
          local_job == SSIDS_SOLVE_JOB_ALL ) THEN
       DO part = 1, akeep%nparts
         CALL fkeep%subtree( part )%ptr%solve_fwd( nrhs, x2, n, inform )
         IF ( inform%stat /= 0 ) GO TO 100
       END DO
     END IF

     IF ( local_job == SSIDS_SOLVE_JOB_DIAG ) THEN
       DO part = 1, akeep%nparts
         CALL fkeep%subtree( part )%ptr%solve_diag( nrhs, x2, n, inform )
         IF ( inform%stat /= 0 ) GO TO 100
       END DO
     END IF

     IF ( local_job == SSIDS_SOLVE_JOB_BWD ) THEN
       DO part = akeep%nparts, 1, -1
         CALL fkeep%subtree( part )%ptr%solve_bwd( nrhs, x2, n, inform )
         IF ( inform%stat /= 0 ) GO TO 100
       END DO
     END IF

     IF ( local_job == SSIDS_SOLVE_JOB_DIAG_BWD .OR.                           &
          local_job == SSIDS_SOLVE_JOB_ALL ) THEN
       DO part = akeep%nparts, 1, - 1
         CALL fkeep%subtree( part )%ptr%solve_diag_bwd( nrhs, x2, n, inform )
         IF ( inform%stat /= 0 ) GO TO 100
       END DO
     END IF

!  unscale/unpermute

     IF ( ALLOCATED( fkeep%scaling ) .AND.                                     &
           ( local_job == SSIDS_SOLVE_JOB_ALL .OR.                             &
             local_job == SSIDS_SOLVE_JOB_BWD .OR.                             &
             local_job == SSIDS_SOLVE_JOB_DIAG_BWD ) ) THEN

!  copy and scale

        DO r = 1, nrhs
          DO i = 1, n
            x( akeep%invp( i ),r ) = x2( i,r ) * fkeep%scaling( i )
          END DO
        END DO

!  just copy

     ELSE
       DO r = 1, nrhs
        x( akeep%invp( 1 : n ), r ) = x2( 1 : n, r )
       END DO
     END IF

     RETURN

 100 CONTINUE
     inform%flag = SSIDS_ERROR_ALLOCATION
     RETURN

     END SUBROUTINE inner_solve_cpu

!-*-  G A L A H A D -  S S I D S _ enquire_posdef_cpu  S U B R O U T I N E  -*-

     SUBROUTINE enquire_posdef_cpu( akeep, fkeep, d )
     TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
     CLASS( ssids_fkeep_type ), TARGET, INTENT( IN ) :: fkeep
     REAL( KIND = rp_ ), DIMENSION( * ), INTENT( OUT ) :: d

!  local variables

     INTEGER( KIND = ip_ ) :: n, part, sa, en
     
     n = akeep%n

!  ensure d is not returned undefined

     d( 1 : n ) = 0.0

     DO part = 1, akeep%nparts
       sa = akeep%part( part )
       en = akeep%part( part + 1 ) - 1
       ASSOCIATE( subtree => fkeep%subtree( part )%ptr )
         SELECT TYPE( subtree )
         TYPE IS ( cpu_numeric_subtree )
           CALL subtree%enquire_posdef( d( sa : en ) )
         END SELECT
       END ASSOCIATE
     END DO

     END SUBROUTINE enquire_posdef_cpu

!-*-  G A L A H A D -  S S I D S _  enquire_indef_cpu  S U B R O U T I N E  -*-

     SUBROUTINE enquire_indef_cpu( akeep, fkeep, inform, piv_order, d )
     TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
     CLASS( ssids_fkeep_type ), TARGET, INTENT( IN ) :: fkeep
     TYPE( ssids_inform_type ), INTENT( INOUT ) :: inform

!  if i is used to index a variable, its position in the pivot sequence
!  will be placed in piv_order(i), with its sign negative if it is
!  part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.

     INTEGER( KIND = ip_ ), DIMENSION( akeep%n ), OPTIONAL,                    &
                                                  INTENT( OUT ) :: piv_order

!  the diagonal entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
!  entries will be placed in d(2,:). The entries are held in pivot order

     REAL( KIND = rp_ ), DIMENSION( 2,akeep%n ), OPTIONAL, INTENT( OUT ) :: d

!  local variables

     INTEGER( KIND = ip_ ) :: i, n, part, sa
     INTEGER( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: po

     n = akeep%n

!  ensure d is not returned undefined

     IF ( PRESENT( d ) ) THEN
       d( 1 : 2, 1 : n ) = 0.0_rp_
     END if

!  we need to apply the invp externally to piv_order

     IF ( PRESENT( piv_order ) ) THEN
       ALLOCATE( po( akeep%n ), STAT = inform%stat )
       IF ( inform%stat /= 0 ) THEN
         inform%flag = SSIDS_ERROR_ALLOCATION
         RETURN
       END IF
     END IF

!  FIXME: should probably return nelim from each part, due to delays passing
!  between them

     DO part = 1, akeep%nparts
       sa = akeep%part( part )
       ASSOCIATE( subtree => fkeep%subtree( 1 )%ptr )
         SELECT TYPE( subtree )
         TYPE IS ( cpu_numeric_subtree )
           IF ( PRESENT( d ) ) THEN
             IF ( PRESENT( piv_order ) ) THEN
               CALL subtree%enquire_indef( piv_order = po( sa : n ),           &
                                           d = d( 1 : 2, sa : n ) )
             ELSE
               CALL subtree%enquire_indef( d = d( 1 : 2, sa : ) )
             END IF
           ELSE
             IF ( PRESENT( piv_order ) ) THEN
               CALL subtree%enquire_indef( piv_order=po( sa:akeep%n ) )
             ELSE ! No-op: should we report an error here? (or done higher up?)
             END IF
           END IF
         END SELECT
       END ASSOCIATE
     END DO

!  apply invp to piv_order

     IF ( PRESENT( piv_order ) ) THEN
       DO i = 1, akeep%n
         piv_order( akeep%invp( i ) ) = po( i )
!        piv_order(  i  ) = po( i )
       END DO
     END IF
     RETURN

     END SUBROUTINE enquire_indef_cpu

!-*-  G A L A H A D -  S S I D S _ a l t e r _ c p u  S U B R O U T I N E  -*-

     SUBROUTINE alter_cpu( d, akeep, fkeep )

!  alter D values

!  the required diagonal entries of D^{-1} must be placed in d(1,i) (i = 1,...n)
!  and the off-diagonal entries must be placed in d(2,i) (i = 1,...n-1)

     REAL( KIND = rp_ ), DIMENSION( 2, * ), INTENT( IN ) :: d
     TYPE( ssids_akeep_type ), INTENT( IN ) :: akeep
     CLASS( ssids_fkeep_type ), TARGET, INTENT( INOUT ) :: fkeep

     INTEGER( KIND = ip_ ) :: part

     DO part = 1, akeep%nparts
       ASSOCIATE( subtree => fkeep%subtree( 1 )%ptr )
         SELECT TYPE( subtree )
         TYPE IS ( cpu_numeric_subtree )
           CALL subtree%alter( d( 1 : 2, akeep%part( part ) :                  &
                                         akeep%part( part + 1 ) - 1 ) )
         END SELECT
       END ASSOCIATE
     END DO
     RETURN

     END SUBROUTINE alter_cpu

!-*-  G A L A H A D -  S S I D S _ f r e e _ f k e e p  S U B R O U T I N E  -*-

     SUBROUTINE free_fkeep( fkeep, flag )

!  free all allocated components of the type fkeep

     CLASS( ssids_fkeep_type ), INTENT( INOUT ) :: fkeep

!  not used for cpu version, set to 0

     INTEGER( KIND = ip_ ), INTENT( OUT ) :: flag

!  local variables

     INTEGER( KIND = ip_ ) :: i, st

!  not used for basic SSIDS, just set to zero

     flag = 0

     DEALLOCATE( fkeep%scaling, STAT = st )
     IF ( ALLOCATED( fkeep%subtree ) ) THEN
       DO i = 1, size( fkeep%subtree )
         IF ( ASSOCIATED( fkeep%subtree( i )%ptr ) ) THEN
           CALL fkeep%subtree( i )%ptr%cleanup(  )
           DEALLOCATE( fkeep%subtree( i )%ptr )
           NULLIFY( fkeep%subtree( i )%ptr )
         END IF
       END DO
       DEALLOCATE( fkeep%subtree )
     END IF
     END SUBROUTINE free_fkeep

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
!!$    RETURN
!!$
!!$    END SUBROUTINE writePPM

  END MODULE GALAHAD_SSIDS_precision
