! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 13:10 GMT
! (consistent with SPRAL up to issue #250)

#include "spral_procedures.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S S I D S   M O D U L E  *-*-*-*-*-*-*-*-*-

!  COPYRIGHT (c) 2011 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  authors: Jonathan Hogg and Jennifer Scott
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

MODULE GALAHAD_SSIDS_precision

!     ------------------------------------------------------------
!    |                                                            |
!    | SSIDS, a sparse, symmetric, idefinite direct linear solver |
!    |                                                            |
!     ------------------------------------------------------------

!$  USE omp_lib
  USE GALAHAD_KINDS_precision
  USE SPRAL_HW_TOPOLOGY, only : guess_topology, numa_region
  USE SPRAL_MATCH_ORDER_precision, ONLY: match_order_metis
  USE SPRAL_MATRIX_UTIL_precision, ONLY: SPRAL_MATRIX_REAL_SYM_INDEF,          &
                                         SPRAL_MATRIX_REAL_SYM_PSDEF,          &
                                         convert_coord_to_cscl,                &
                                         clean_cscl_oop,                       &
                                         apply_conversion_map
  USE SPRAL_SCALING_precision, ONLY: auction_scale_sym, equilib_scale_sym,     &
                                     hungarian_scale_sym,                      &
                                     equilib_control_type,                     &
                                     equilib_inform_type,                      &
                                     hungarian_control_type,                   &
                                     hungarian_inform_type
  USE SPRAL_RAL_BOEING_precision, ONLY: rb_write_options, rb_write
  USE GALAHAD_NODEND_precision, ONLY: NODEND_half_order, NODEND_control_type,  &
                                      NODEND_inform_type
  USE GALAHAD_SSIDS_analyse_precision, ONLY: analyse_phase, check_order,       &
                                             expand_matrix, expand_pattern
  USE GALAHAD_SSIDS_types_precision
  USE GALAHAD_SSIDS_akeep_precision, ONLY: SSIDS_akeep_type
  USE GALAHAD_SSIDS_fkeep_precision, ONLY: SSIDS_fkeep_type
  USE GALAHAD_SSIDS_inform_precision, ONLY: SSIDS_inform_type

  IMPLICIT none
  PRIVATE

!  data types

  PUBLIC :: SSIDS_akeep_type, SSIDS_fkeep_type, SSIDS_control_type,            &
            SSIDS_inform_type

!  user interface routines

  PUBLIC :: ssids_analyse,         & ! Analyse phase, CSC-lower input
            ssids_analyse_coord,   & ! Analyse phase, Coordinate input
            ssids_factor,          & ! Factorize phase
            ssids_solve,           & ! Solve phase
            ssids_free,            & ! Free akeep and/or fkeep
            ssids_enquire_posdef,  & ! Pivot information in posdef case
            ssids_enquire_indef,   & ! Pivot information in indef case
            ssids_alter              ! Alter diagonal

   LOGICAL, PUBLIC, PROTECTED :: ssids_available = .TRUE.

!----------------------
!   I n t e r f a c e s
!----------------------

  INTERFACE ssids_analyse
    MODULE PROCEDURE analyse_precision, analyse_precision_ptr32
  END INTERFACE ssids_analyse

  INTERFACE ssids_analyse_coord
    MODULE PROCEDURE ssids_analyse_coord_precision
  END INTERFACE ssids_analyse_coord

  INTERFACE ssids_factor
    MODULE PROCEDURE ssids_factor_ptr32_precision, ssids_factor_ptr64_precision
  END INTERFACE ssids_factor

  INTERFACE ssids_solve
    MODULE PROCEDURE ssids_solve_one_precision
    MODULE PROCEDURE ssids_solve_mult_precision
  END INTERFACE ssids_solve

  INTERFACE ssids_free
    MODULE PROCEDURE free_akeep_precision
    MODULE PROCEDURE free_fkeep_precision
    MODULE PROCEDURE free_both_precision
  END INTERFACE ssids_free

  INTERFACE ssids_enquire_posdef
    MODULE PROCEDURE ssids_enquire_posdef_precision
  END INTERFACE ssids_enquire_posdef

  INTERFACE ssids_enquire_indef
    MODULE PROCEDURE ssids_enquire_indef_precision
  END INTERFACE ssids_enquire_indef

  INTERFACE ssids_alter
    MODULE PROCEDURE ssids_alter_precision
  END INTERFACE ssids_alter

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

  TYPE :: omp_settings ! CACHES user OpenMP ICV values for later restoration
    LOGICAL :: dynamic
    INTEGER( ip_ ) :: max_active_levels
  END TYPE omp_settings

CONTAINS

!-*-*-  G A L A H A D - S S I D S _ analyse_ptr32  S U B R O U T I N E  -*-*-

    SUBROUTINE analyse_precision_ptr32( check, n, ptr, row, akeep, control,    &
                                        inform, order, val, topology )

!  provides a wrapper around analyse_precision( ) that copies the
!  32-bit ptr to a 64-bit array before calling the 64-bit version

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: check
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( i4_ ), INTENT( IN ) :: ptr( : )
    INTEGER( ip_ ), INTENT( IN ) :: row( : )
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    INTEGER( ip_ ), OPTIONAL, INTENT( INOUT ) :: order( : )
    REAL( rp_ ), OPTIONAL, INTENT( IN ) :: val( : )
    TYPE( numa_region ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: topology

    INTEGER( i8_ ), DIMENSION( : ), ALLOCATABLE :: ptr64

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

!  analyse phase: mMatrix entered in CSC format ( lower triangle ).
!  The user OPTIONALly inputs the pivot order. If not, metis called.
!  Structure is then expanded, supervariables are computed and then
!  the assembly tree is constructed and the data structures required
!  by the factorization are set up.
!
!  There is no checking of the user's data if check = .false.
!  Otherwise, matrix_util routines are used to clean data.
!
!  See user documentation for full detail on parameters.
!
!  check Clean matrix data if true ( cleaned version stored in akeep ).
!  n Order of A.
!  ptr Column pointers of A.
!  row Row indices of A.
!  akeep Symbolic factorization out.
!  control User-supplied options.
!  inform Stats/information RETURNed to user.
!  order RETURN ordering to user / allow user to supply order.
!  val Values of A. Only required if matching-based ordering requested.
!  topology Specify machine topology to work with.

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: check
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( i8_ ), INTENT( IN ) :: ptr( : )
    INTEGER( ip_ ), INTENT( IN ) :: row( : )
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    INTEGER( ip_ ), OPTIONAL, INTENT( INOUT ) :: order( : )
    REAL( rp_ ), OPTIONAL, INTENT( IN ) :: val( : )
    TYPE( numa_region ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: topology

    CHARACTER( LEN = 50 ) :: context !  Procedure name ( used when printing )
    INTEGER( ip_ ) :: mu_flag        !  error flag for matrix_util routines
    INTEGER( long_ ) :: nz           !  entries in expanded matrix
    INTEGER( ip_ ) :: st             !  stat parameter
    INTEGER( ip_ ) :: flag           !  error flag for metis

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: order2

!  col ptrs and row indices for the expanded matrix

    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: ptr2
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: row2  

!  The following are only used for matching-based orderings
!  cleaned values if val is present and checking is required

    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: val_clean

!  expanded matrix if val is present

    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: val2

    INTEGER( ip_ ) :: mo_flag
    INTEGER( ip_ ) :: free_flag
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

    IF ( ( control%ordering < 0 ) .OR. ( control%ordering > 2 ) ) THEN
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
        CALL clean_cscl_oop( SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ptr, row,      &
                             akeep%ptr, akeep%row, mu_flag, val_in = val,      &
                             val_out = val_clean, lmap = akeep%lmap,           &
                             map = akeep%map, noor = inform%matrix_outrange,   &
                             ndup = inform%matrix_dup )
      ELSE
        CALL clean_cscl_oop( SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ptr, row,      &
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
      order2( 1:n ) = order( 1:n )
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
!  it and then call match_order_metis( ) with full matrix supplied

    CASE( 2 )
      IF ( check ) THEN
        CALL expand_matrix( n, nz, akeep%ptr, akeep%row, val_clean, ptr2,      &
                            row2, val2 )
        DEALLOCATE ( val_clean, STAT = st )
      ELSE
        CALL expand_matrix( n, nz, ptr, row, val, ptr2, row2, val2 )
      END IF
      CALL match_order_metis( n, ptr2, row2, val2, order2, akeep%scaling,      &
                              control%nodend_control, inform%nodend_inform,    &
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
      CALL guess_topology( akeep%topology, st )
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

    IMPLICIT none
    TYPE( numa_region ), DIMENSION( : ), ALLOCATABLE,                          &
                                         INTENT( INOUT ) :: topology
    TYPE( ssids_control_type ), INTENT( IN ) :: control
    INTEGER( ip_ ), INTENT( OUT ) :: st

    LOGICAL :: no_omp
    INTEGER( ip_ ) :: i, j, ngpu
    TYPE( numa_region ), DIMENSION( : ), ALLOCATABLE :: new_topology

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
               MIN( new_topology( 1 )%gpus( 1 ), minval( topology( i )%gpus ) )
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

    ELSE IF ( ( SIZE( topology ) > 1 ) .AND. control%ignore_numa ) THEN
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

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: n !  order of A
    INTEGER( long_ ), INTENT( IN ) :: ne !  entries to be input by user
    INTEGER( ip_ ), INTENT( IN ) :: row( : ) !  row indices
    INTEGER( ip_ ), INTENT( IN ) :: col( : ) !  col indices
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep !  See derived-type declaration
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control !  See derived-type declaration
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform !  See derived-type declaration
    INTEGER( ip_ ), INTENT( INOUT ), OPTIONAL  :: order( : )
  !  Must be present and set on entry if control%ordering = 0
  !  i is used to index a variable, order( i ) must
  !  hold its position in the pivot sequence.
  !  If i is not used to index a variable,
  !  order( i ) must be set to zero.
  !  On exit, holds the pivot order to be used by factorization.
    REAL( rp_ ), OPTIONAL, INTENT( IN ) :: val( : ) !  must be present
  !  if a matching-based elimination ordering is required
  !  ( control%ordering = 2 ).
  !  If present, val( k ) must hold value of entry in row( k ) and col( k ).
    TYPE( numa_region ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: topology
  !  user specified topology

    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: ptr2 !  col ptrs and row indices
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: row2!  for the expanded matrix
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: order2 !  pivot order

    INTEGER( ip_ ) :: mo_flag

    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: val_clean !  cleaned values if
  !  val is present.
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: val2 !  expanded matrix ( val present )

    CHARACTER( 50 )  :: context!  procedure name ( used when printing ).
    INTEGER( ip_ ) :: mu_flag  !  error flag for matrix_util routines
    INTEGER( long_ ) :: nz  !  entries in expanded matrix
!   INTEGER( ip_ ) :: flag  !  error flag for metis
    INTEGER( ip_ ) :: st    !  stat parameter
    INTEGER( ip_ ) :: free_flag

    TYPE( ssids_inform_type ) :: inform_default

!  Initialise
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

!  Output status on entry
    CALL control%print_summary_analyse( context )
    IF ( ( control%print_level >= 1 ) .AND. &
        ( control%unit_diagnostics >= 0 ) ) THEN
       WRITE ( control%unit_diagnostics,'( a,i15 )' ) &
            ' n                         =  ', n
       WRITE ( control%unit_diagnostics,'( a,i15 )' ) &
            ' ne                        =  ', ne
    END IF

    akeep%check = .true.
    akeep%n = n

!
!  Checking of matrix data
!
    IF ( ( n < 0 ) .OR. ( ne < 0 ) ) THEN
       inform%flag = SSIDS_ERROR_A_N_OOR
       akeep%inform = inform
       CALL inform%print_flag( control, context )
       RETURN
    END IF

    IF ( n == 0 ) THEN
       akeep%nnodes = 0
       ALLOCATE( akeep%sptr( 0 ), STAT = st ) !  used to check if analyse has been run
       IF ( st /= 0 ) GO TO 490
       akeep%inform = inform
       RETURN
    END IF

!  check control%ordering has a valid value
    IF ( ( control%ordering < 0 ) .OR. ( control%ordering > 2 ) ) THEN
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
       CALL convert_coord_to_cscl( SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ne, row,  &
            col, akeep%ptr, akeep%row, mu_flag, val_in=val, val_out=val_clean, &
            lmap=akeep%lmap, map=akeep%map,                                    &
            noor=inform%matrix_outrange,  ndup=inform%matrix_dup )
    ELSE
       CALL convert_coord_to_cscl( SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ne, row,  &
            col, akeep%ptr, akeep%row, mu_flag, lmap=akeep%lmap, map=akeep%map,&
            noor=inform%matrix_outrange, ndup=inform%matrix_dup )
    END IF

!  Check for errors
    IF ( mu_flag < 0 ) THEN
       IF ( mu_flag == -1 )  inform%flag = SSIDS_ERROR_ALLOCATION
       IF ( mu_flag == -10 ) inform%flag = SSIDS_ERROR_A_ALL_OOR
       akeep%inform = inform
       CALL inform%print_flag( control, context )
       RETURN
    END IF

!  Check whether warning needs to be raised
!  Note: same numbering of positive flags as in matrix_util
    IF ( mu_flag > 0 ) inform%flag = mu_flag

    nz = akeep%ptr( n + 1 ) - 1

!  If the pivot order is not supplied, we need to compute an order
!  here, before we expand the matrix structure.
!  Otherwise, we must check the supplied order.

    ALLOCATE( akeep%invp( n ),order2( n ),ptr2( n + 1 ),row2( 2*nz ),STAT = st )
    IF ( st /= 0 ) GO TO 490
    IF ( control%ordering == 2 ) THEN
       ALLOCATE( val2( 2*nz ),akeep%scaling( n ),STAT = st )
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
       CALL check_order( n,order,akeep%invp,control,inform )
       IF ( inform%flag < 0 ) GO TO 490
       order2( 1 : n ) = order( 1 : n )
       CALL expand_pattern( n, nz, akeep%ptr, akeep%row, ptr2, row2 )

!  METIS ordering

    CASE( 1 )
       CALL NODEND_half_order( n, akeep%ptr, akeep%row, order2, &
                              control%nodend_control,  &
                              inform%nodend_inform )
       inform%flag = inform%nodend_inform%status
       st = inform%nodend_inform%alloc_status
       IF ( inform%flag < 0 ) GO TO 490
       CALL expand_pattern( n, nz, akeep%ptr, akeep%row, ptr2, row2 )

!  matching-based ordering required

    CASE( 2 )
       CALL expand_matrix( n, nz, akeep%ptr, akeep%row, val_clean, ptr2, row2, &
            val2 )
       DEALLOCATE ( val_clean,STAT = st )

       CALL match_order_metis( n, ptr2, row2, val2, order2, akeep%scaling,     &
                               control%nodend_control, inform%nodend_inform,   &
                               mo_flag, inform%stat )

       SELECT CASE( mo_flag )
       CASE( 0 )

!  success; do nothing

       CASE( 1 )

!  singularity warning required

          inform%flag = SSIDS_WARNING_ANALYSIS_SINGULAR
       CASE( -1 )
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

       DEALLOCATE( val2,STAT = st )
    END SELECT

!  assess topology: user supplied

    IF ( PRESENT( topology ) ) THEN
      ALLOCATE( akeep%topology( SIZE( topology ) ), STAT = st )
      IF ( st /= 0 ) GO TO 490
      akeep%topology( : ) = topology( : )

!  guess it

    ELSE
      CALL guess_topology( akeep%topology, st )
      IF ( st /= 0 ) GO TO 490
    END IF

!  we now have the expanded structure held using ptr2, row2
!  and we are ready to get on with the analyse phase.
    CALL analyse_phase( n, akeep%ptr, akeep%row, ptr2, row2, order2,  &
         akeep%invp, akeep, control, inform )
    IF ( inform%flag < 0 ) GO TO 490

    IF ( PRESENT( order ) ) order( 1:n ) = abs( order2( 1:n ) )
    IF ( control%print_level > DEBUG_PRINT_LEVEL ) &
         print *, "order = ", order2( 1:n )

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

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: posdef
    REAL( rp_ ), DIMENSION( * ), target, INTENT( IN ) :: val
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    REAL( rp_ ), DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: scale
    INTEGER( i4_ ), DIMENSION( akeep%n + 1 ), INTENT( IN ) :: ptr
    INTEGER( ip_ ), DIMENSION( * ), OPTIONAL, INTENT( IN ) :: row

    INTEGER( i8_ ), DIMENSION( : ), ALLOCATABLE :: ptr64

!  copy from 32-bit to 64-bit ptr

    ALLOCATE( ptr64( akeep%n + 1 ), stat=inform%stat )
    IF ( inform%stat /= 0 ) THEN
      inform%flag = SSIDS_ERROR_ALLOCATION
      CALL inform%print_flag( control, 'ssids_factor' )
      fkeep%inform = inform
      RETURN
    END IF
    ptr64( 1 : akeep%n + 1 ) = ptr( 1 : akeep%n + 1 )

!  Call 64-bit routine
    CALL ssids_factor_ptr64_precision( posdef, val, akeep, fkeep, control,     &
                                       inform, scale = scale, ptr = ptr64,     &
                                       row = row )
    RETURN

    END SUBROUTINE ssids_factor_ptr32_precision

!-*-*-  G A L A H A D - S S I D S _ factor_ptr64  S U B R O U T I N E  -*-*-

  SUBROUTINE ssids_factor_ptr64_precision( posdef, val, akeep, fkeep, control, &
                                           inform, scale, ptr, row )

!  factorize phase ( 64-bit pointers )

    IMPLICIT none
    LOGICAL, INTENT( IN ) :: posdef
    REAL( rp_ ), DIMENSION( * ), target, INTENT( IN ) :: val !  A values ( lwr triangle )
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    REAL( rp_ ), DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: scale !  used to hold
  !  row and column scaling factors. Must be set on entry if
  !  control%scaling <= 0
  !  Note: Has to be assumed shape, not assumed size or fixed size to work
  !  around funny compiler bug
    INTEGER( i8_ ), DIMENSION( akeep%n + 1 ), OPTIONAL, INTENT( IN ) :: ptr !  must be
  !  present if on call to analyse phase, check = .false.. Must be unchanged
  !  since that call.
    INTEGER( ip_ ), DIMENSION( * ), OPTIONAL, INTENT( IN ) :: row !  must be present if
  !  on call to analyse phase, check = .false.. Must be unchanged
  !  since that call.

    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE, target :: val2
    CHARACTER( len=50 ) :: context

    INTEGER( ip_ ) :: i
    INTEGER( ip_ ) :: n
    INTEGER( i8_ ) :: nz
    INTEGER( ip_ ) :: st
!  Solve parameters. Tree is broken up into multiple chunks. Parent-child
!  relations between chunks are stored in fwd_ptr and fwd ( see solve routine
!  comments )
    INTEGER( ip_ ) :: matrix_type
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: scaling

!  TYPEs related to scaling routines
    TYPE( hungarian_control_type ) :: hscontrol
    TYPE( hungarian_inform_type ) :: hsinform
    TYPE( equilib_control_type ) :: escontrol
    TYPE( equilib_inform_type ) :: esinform

    TYPE( omp_settings ) :: user_omp_settings
    TYPE( rb_WRITE_options ) :: rb_control
    INTEGER( ip_ ) :: flag

!  Setup for any printing we may require
    context = 'ssids_factor'

!  Print summary of input control ( depending on print level etc. )
    CALL control%print_summary_factor( posdef, context )

!  Check for error in call sequence
    IF ( ( .NOT. ALLOCATED( akeep%sptr ) ) .OR. ( akeep%inform%flag < 0 ) ) THEN
!  Analyse cannot have been run
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       fkeep%inform = inform
       RETURN
    END IF

!  Initialize
    inform = akeep%inform
    inform%num_factor = 0
    inform%num_flops = 0
    st = 0
    n = akeep%n

!  Ensure OpenMP setup is as required
    CALL push_omp_settings( user_omp_settings, inform%flag )
    IF ( inform%flag < 0 ) THEN
       fkeep%inform = inform
       CALL inform%print_flag( control, context )
       RETURN
    END IF

!  Immediate RETURN if analyse detected singularity and control%action=false
    IF ( ( .NOT. control%action ) .AND. &
        ( akeep%n /= akeep%inform%matrix_rank ) ) THEN
       inform%flag = SSIDS_ERROR_SINGULAR
       GO TO 100
    END IF

!  Immediate RETURN for trivial matrix
    IF ( akeep%nnodes == 0 ) THEN
       inform%flag = SSIDS_SUCCESS
       inform%matrix_rank = 0
       GO TO 100
    END IF

    fkeep%pos_def = posdef
    IF ( posdef ) THEN
       matrix_type = SPRAL_MATRIX_REAL_SYM_PSDEF
    ELSE
       matrix_type = SPRAL_MATRIX_REAL_SYM_INDEF
    END IF

!  If matrix has been checked, produce a clean version of val in val2
    IF ( akeep%check ) THEN
       nz = akeep%ptr( n + 1 ) - 1
       ALLOCATE( val2( nz ),STAT = st )
       IF ( st /= 0 ) GO TO 10
       CALL apply_conversion_map( matrix_type, akeep%lmap, akeep%map, val, &
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

!  At this point, either  ptr, row, val
!                  or    akeep%ptr, akeep%row, val2
!  hold the lower triangular part of A

!  Dump matrix if required
    IF ( ALLOCATED( control%rb_dump ) ) THEN
       WRITE( control%unit_warning,* ) "Dumping matrix to '", control%rb_dump, "'"
       IF ( akeep%check ) THEN
          CALL rb_WRITE( control%rb_dump, SPRAL_MATRIX_REAL_SYM_INDEF, &
               n, n, akeep%ptr, akeep%row, rb_control, flag, val=val2 )
       ELSE
          CALL rb_WRITE( control%rb_dump, SPRAL_MATRIX_REAL_SYM_INDEF, &
               n, n, ptr, row, rb_control, flag, val=val )
       END IF
       IF ( flag /= 0 ) THEN
          inform%flag = SSIDS_ERROR_UNKNOWN
          GO TO 100
       END IF
    END IF

!
!  Perform scaling if required
!
    IF ( ( control%scaling > 0 ) .OR. PRESENT( scale ) ) THEN
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

    IF ( ALLOCATED( akeep%scaling ) .AND. ( control%scaling /= 3 ) ) THEN
       inform%flag = SSIDS_WARNING_MATCH_ORD_NO_SCALE
       CALL inform%print_flag( control, context )
    END IF

    SELECT case ( control%scaling )
    CASE( :0 ) !  User supplied or none
       IF ( PRESENT( scale ) ) THEN
          DO i = 1, n
             fkeep%scaling( i ) = scale( akeep%invp( i ) )
          END DO
       END IF
    CASE( 1 ) !  Matching-based scaling by Hungarian Algorithm ( MC64 algorithm )
!  ALLOCATE space for scaling
       ALLOCATE( scaling( n ), STAT = st )
       IF ( st /= 0 ) GO TO 10
!  Run Hungarian algorithm
       hscontrol%scale_if_singular = control%action
       IF ( akeep%check ) THEN
          CALL hungarian_scale_sym( n, akeep%ptr, akeep%row, val2, scaling, &
               hscontrol, hsinform )
       ELSE
          CALL hungarian_scale_sym( n, ptr, row, val, scaling, &
               hscontrol, hsinform )
       END IF
       SELECT CASE( hsinform%flag )
       CASE( -1 )
   !  Allocation error
          st = hsinform%stat
          GO TO 10
       CASE( -2 )
   !  Structually singular matrix and control%action=.false.
          inform%flag = SSIDS_ERROR_SINGULAR
          GO TO 100
       END SELECT
!  Permute scaling to correct order
       DO i = 1, n
          fkeep%scaling( i ) = scaling( akeep%invp( i ) )
       END DO
!  Copy scaling( : ) to user array scale( : ) if present
       IF ( PRESENT( scale ) ) THEN
          scale( 1:n ) = scaling( 1:n )
       END IF
!  Cleanup memory
       DEALLOCATE( scaling, STAT = st )

    CASE( 2 ) !  Matching-based scaling by Auction Algorithm
!  ALLOCATE space for scaling
       ALLOCATE( scaling( n ), STAT = st )
       IF ( st /= 0 ) GO TO 10
!  Run auction algorithm
       IF ( akeep%check ) THEN
          CALL auction_scale_sym( n, akeep%ptr, akeep%row, val2, scaling, &
               control%auction, inform%auction )
       ELSE
          CALL auction_scale_sym( n, ptr, row, val, scaling, &
               control%auction, inform%auction )
       END IF
       IF ( inform%auction%flag /= 0 ) THEN
   !  only possible error is allocation failed
          st = inform%auction%stat
          GO TO 10
       END IF
!  Permute scaling to correct order
       DO i = 1, n
          fkeep%scaling( i ) = scaling( akeep%invp( i ) )
       END DO
!  Copy scaling( : ) to user array scale( : ) if present
       IF ( PRESENT( scale ) ) THEN
          scale( 1:n ) = scaling( 1:n )
       END IF
!  Cleanup memory
       DEALLOCATE( scaling, STAT = st )

    CASE( 3 ) !  Scaling generated during analyse phase for matching-based order
       IF ( .NOT. ALLOCATED( akeep%scaling ) ) THEN
   !  No scaling saved from analyse phase
          inform%flag = SSIDS_ERROR_NO_SAVED_SCALING
          GO TO 100
       END IF
       DO i = 1, n
          fkeep%scaling( i ) = akeep%scaling( akeep%invp( i ) )
       END DO

    CASE( 4 : ) !  Norm equilibriation algorithm ( MC77 algorithm )
!  ALLOCATE space for scaling
       ALLOCATE( scaling( n ), STAT = st )
       IF ( st /= 0 ) GO TO 10
!  Run equilibriation algorithm
       IF ( akeep%check ) THEN
          CALL equilib_scale_sym( n, akeep%ptr, akeep%row, val2, scaling, &
               escontrol, esinform )
       ELSE
          CALL equilib_scale_sym( n, ptr, row, val, scaling, &
               escontrol, esinform )
       END IF
       IF ( esinform%flag /= 0 ) THEN
   !  Only possible error is memory allocation failure
          st = esinform%stat
          GO TO 10
       END IF
!  Permute scaling to correct order
       DO i = 1, n
          fkeep%scaling( i ) = scaling( akeep%invp( i ) )
       END DO
!  Copy scaling( : ) to user array scale( : ) if present
       IF ( PRESENT( scale ) ) THEN
          DO i = 1, n
             scale( akeep%invp( i ) ) = fkeep%scaling( i )
          END DO
       END IF
!  Cleanup memory
       DEALLOCATE( scaling, STAT = st )
    END SELECT

!IF ( ALLOCATED( fkeep%scaling ) ) &
!   print *, "minscale, maxscale = ", minval( fkeep%scaling ), &
!      maxval( fkeep%scaling )

!  Setup data storage
    IF ( ALLOCATED( fkeep%subtree ) ) THEN
       DO i = 1, SIZE( fkeep%subtree )
          IF ( ASSOCIATED( fkeep%subtree( i )%ptr ) ) THEN
             CALL fkeep%subtree( i )%ptr%cleanup( )
             DEALLOCATE( fkeep%subtree( i )%ptr )
          END IF
       END DO
       DEALLOCATE( fkeep%subtree )
    END IF

!  Call main factorization routine
    IF ( akeep%check ) THEN
!WRITE( 6,* ) 'val2 = ', val2( :nz )
       CALL fkeep%inner_factor( akeep, val2, control, inform )
    ELSE
!WRITE( 6,* ) 'val = ', val( :ptr( n + 1 )-1 )
       CALL fkeep%inner_factor( akeep, val, control, inform )
    END IF
    IF ( inform%flag < 0 ) THEN
       fkeep%inform = inform
       GO TO 100
    END IF

    IF ( akeep%n /= inform%matrix_rank ) THEN
!  Rank deficient
!  Note: If we reach this point then must be control%action=.true.
       IF ( control%action ) THEN
          inform%flag = SSIDS_WARNING_FACT_SINGULAR
       ELSE
          inform%flag = SSIDS_ERROR_SINGULAR
       END IF
       CALL inform%print_flag( control, context )
    END IF
WRITE( 99,* ) ' c'

    IF ( ( control%print_level >= 1 ) .AND. &
        ( control%unit_diagnostics >= 0 ) ) THEN
       WRITE ( control%unit_diagnostics,'( /a )' ) &
            ' Completed factorisation with:'
       WRITE ( control%unit_diagnostics, &
         '( a,3( /a,i12 ),2( /a,es12.4 ),5( /a,i12 ) )' ) &
         ' information parameters ( inform% ) :', &
         ' flag                   Error flag                               = ',&
         inform%flag, &
         ' maxfront               Maximum frontsize                        = ',&
         inform%maxfront, &
         ' maxsupernode           Maximum supernode size                   = ',&
         inform%maxsupernode, &
         ' num_factor             Number of entries in L                   = ',&
         REAL( inform%num_factor ), &
         ' num_flops              Number of flops performed                = ',&
         REAL( inform%num_flops ), &
         ' num_two                Number of 2x2 pivots used                = ',&
         inform%num_two, &
         ' num_delay              Number of delayed eliminations           = ',&
         inform%num_delay, &
         ' rank                   Computed rank                            = ',&
         inform%matrix_rank, &
         ' num_neg                Computed number of negative eigenvalues  = ',&
          inform%num_neg
    END IF

!  normal return just drops through

100 CONTINUE

!  Clean up and return

    fkeep%inform = inform
    CALL inform%print_flag( control, context )
    CALL pop_omp_settings( user_omp_settings )
    RETURN

!  Error handling

10  CONTINUE
    inform%flag = SSIDS_ERROR_ALLOCATION
    inform%stat = st
    GO TO 100
  END SUBROUTINE ssids_factor_ptr64_precision

!-*-*-  G A L A H A D - S S I D S _ solve_one   S U B R O U T I N E  -*-*-

  SUBROUTINE ssids_solve_one_precision( x1, akeep, fkeep, control, inform, job )

!  solve phase single x

    IMPLICIT none
    REAL( rp_ ), DIMENSION( : ), INTENT( INOUT ) :: x1 !  On entry, x must
  !  be set so that if i has been used to index a variable,
  !  x( i ) is the corresponding component of the
  !  right-hand side.On exit, if i has been used to index a variable,
  !  x( i ) holds solution for variable i
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: job

    INTEGER( ip_ ) :: ldx

    ldx = SIZE( x1 )
    IF ( PRESENT( job ) ) THEN
      CALL ssids_solve_mult_precision( 1_ip_, x1, ldx, akeep, fkeep, control,  &
                                       inform, job )
    ELSE
      CALL ssids_solve_mult_precision( 1_ip_, x1, ldx, akeep, fkeep, control,  &
                                       inform )
    END IF
  END SUBROUTINE ssids_solve_one_precision

!-*-*-  G A L A H A D - S S I D S _  solve_mult  S U B R O U T I N E  -*-*-

  SUBROUTINE ssids_solve_mult_precision( nrhs, x, ldx, akeep, fkeep, control, &
                                        inform, job )

!  solve phase multiple x

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: nrhs
    INTEGER( ip_ ), INTENT( IN ) :: ldx
    REAL( rp_ ), DIMENSION( ldx,nrhs ), INTENT( INOUT ), target :: x
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep !  On entry, x must
  !  be set so that if i has been used to index a variable,
  !  x( i,j ) is the corresponding component of the
  !  right-hand side for the jth system ( j = 1,2,..., nrhs ).
  !  On exit, if i has been used to index a variable,
  !  x( i,j ) holds solution for variable i to system j
!  For details of keep, control, inform : see derived TYPE description
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep !inout for moving data
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: job !  used to indicate whether
  !  partial solution required
  !  job = 1 : forward eliminations only ( PLX = B )
  !  job = 2 : diagonal solve ( DX = B ) ( indefinite case only )
  !  job = 3 : backsubs only ( ( PL )^TX = B )
  !  job = 4 : diag and backsubs ( D( PL )^TX = B ) ( indefinite case only )
  !  job absent: complete solve performed

    CHARACTER( 50 )  :: context  !  Procedure name ( used when printing ).
    INTEGER( ip_ ) :: local_job !  local job parameter
    INTEGER( ip_ ) :: n

    inform%flag = SSIDS_SUCCESS

!  Perform appropriate printing
    IF ( ( control%print_level >= 1 ) .AND. &
        ( control%unit_diagnostics >= 0 ) ) THEN
       WRITE ( control%unit_diagnostics,'( //a )' ) &
            ' Entering ssids_solve with:'
       WRITE ( control%unit_diagnostics,'( a,4( /a,i12 ),( /a,i12 ) )' ) &
            ' control parameters ( control% ) :', &
            ' print_level         Level of diagnostic printing        = ', &
            control%print_level, &
            ' unit_diagnostics    Unit for diagnostics                = ', &
            control%unit_diagnostics, &
            ' unit_error          Unit for errors                     = ', &
            control%unit_error, &
            ' unit_warning        Unit for warnings                   = ', &
            control%unit_warning, &
            ' nrhs                                                    = ', &
            nrhs
       IF ( nrhs > 1 ) WRITE ( control%unit_diagnostics,'( /a,i12 )' ) &
            ' ldx                                                     = ', &
            ldx
    END IF

    context = 'ssids_solve'

    IF ( akeep%nnodes == 0 ) RETURN

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
!  factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       RETURN
    END IF

    inform%flag = max( SSIDS_SUCCESS, fkeep%inform%flag ) !  Preserve warnings
!  immediate RETURN if already had an error
    IF ( ( akeep%inform%flag < 0 ) .OR. ( fkeep%inform%flag < 0 ) ) THEN
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       RETURN
    END IF

    n = akeep%n
    IF ( ldx < n ) THEN
       inform%flag = SSIDS_ERROR_X_SIZE
       CALL inform%print_flag( control, context )
       IF ( ( control%print_level >= 0 ) .AND. ( control%unit_error > 0 ) ) &
            WRITE ( control%unit_error,'( a,i8,a,i8 )' ) &
            ' Increase ldx from ', ldx, ' to at least ', n
       RETURN
    END IF

    IF ( nrhs < 1 ) THEN
       inform%flag = SSIDS_ERROR_X_SIZE
       CALL inform%print_flag( control, context )
       IF ( ( control%print_level >= 0 ) .AND. ( control%unit_error > 0 ) ) &
            WRITE ( control%unit_error,'( a,i8,a,i8 )' ) &
            ' nrhs must be at least 1. nrhs = ', nrhs
       RETURN
    END IF

!  Copy previous phases' inform data from akeep and fkeep
    inform = fkeep%inform

!  Set local_job
    local_job = 0
    IF ( PRESENT( job ) ) THEN
       IF ( ( job < SSIDS_SOLVE_JOB_FWD ) .OR. &
           ( job > SSIDS_SOLVE_JOB_DIAG_BWD ) ) &
            inform%flag = SSIDS_ERROR_JOB_OOR
       IF ( fkeep%pos_def .AND. ( job == SSIDS_SOLVE_JOB_DIAG ) ) &
            inform%flag = SSIDS_ERROR_JOB_OOR
       IF ( fkeep%pos_def .AND. ( job == SSIDS_SOLVE_JOB_DIAG_BWD ) ) &
            inform%flag = SSIDS_ERROR_JOB_OOR
       IF ( inform%flag == SSIDS_ERROR_JOB_OOR ) THEN
          CALL inform%print_flag( control, context )
          RETURN
       END IF
       local_job = job
    END IF

    CALL fkeep%inner_solve( local_job, nrhs, x, ldx, akeep, inform )
    CALL inform%print_flag( control, context )
  END SUBROUTINE ssids_solve_mult_precision

!-*-*-  G A L A H A D - S S I D S _  enquire_posdef  S U B R O U T I N E  -*-*-

  SUBROUTINE ssids_enquire_posdef_precision( akeep, fkeep, control, inform, d )

!  RETURN diagonal entries to user

    IMPLICIT none
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), target, INTENT( IN ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform
    REAL( rp_ ), DIMENSION( * ), INTENT( OUT ) :: d

    CHARACTER( 50 )  :: context  !  Procedure name ( used when printing ).

    context = 'ssids_enquire_posdef'
    inform%flag = SSIDS_SUCCESS

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
!  factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       RETURN
    END IF

    IF ( ( akeep%inform%flag < 0 ) .OR. ( fkeep%inform%flag < 0 ) ) THEN
!  immediate RETURN if had an error
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
  END SUBROUTINE ssids_enquire_posdef_precision

!-*-*-  G A L A H A D - S S I D S _ enquire_indef S U B R O U T I N E  -*-*-

    SUBROUTINE ssids_enquire_indef_precision( akeep, fkeep, control, inform,   &
                                              piv_order, d )

!  In the indefinite case, the pivot sequence used will not necessarily be
!  the same as that passed to ssids_factor ( because of delayed pivots ). 
!  This SUBROUTINE allows the user to obtain the pivot sequence that was
!  actually used. also the entries of D^{-1} are RETURNed using array d.

    IMPLICIT none
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), target, INTENT( IN ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

!  if i is used to index a variable, its position in the pivot sequence
!  will be placed in piv_order(i), with its sign negative if it is
!  part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.

    INTEGER( ip_ ), DIMENSION( * ), OPTIONAL, INTENT( OUT ) :: piv_order

!  the diagonal entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
!  entries will be placed in d( 2,: ). The entries are held in pivot order.

    REAL( rp_ ), DIMENSION( 2,* ), OPTIONAL, INTENT( OUT ) :: d
    CHARACTER( 50 )  :: context  !  Procedure name ( used when printing ).
    INTEGER( ip_ ) :: i, po
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
    END SUBROUTINE ssids_enquire_indef_precision

!-*-*-*-  G A L A H A D - S S I D S _  a l t e r  S U B R O U T I N E  -*-*-*-

!  In indefinite case, the entries of D^{-1} may be changed using this routine

  SUBROUTINE ssids_alter_precision( d, akeep, fkeep, control, inform )
    IMPLICIT none
    REAL( rp_ ), DIMENSION( 2, * ), INTENT( IN ) :: d  !  The required diagonal entries
  !  of D^{-1} must be placed in d( 1,i ) ( i = 1,...n )
  !  and the off-diagonal entries must be placed in d( 2,i ) ( i = 1,...n-1 ).
    TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
    TYPE( SSIDS_fkeep_type ), target, INTENT( INOUT ) :: fkeep
    TYPE( SSIDS_control_type ), INTENT( IN ) :: control
    TYPE( SSIDS_inform_type ), INTENT( OUT ) :: inform

    CHARACTER( 50 )  :: context  !  Procedure name ( used when printing ).

    context = 'ssids_alter'
    inform%flag = SSIDS_SUCCESS

    IF ( .NOT. ALLOCATED( fkeep%subtree ) ) THEN
!  factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       CALL inform%print_flag( control, context )
       RETURN
    END IF

!  immediate RETURN if already had an error
    IF ( ( akeep%inform%flag < 0 ) .OR. ( fkeep%inform%flag < 0 ) ) THEN
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
  END SUBROUTINE ssids_alter_precision

!-*-*-  G A L A H A D - S S I D S _ f r e e _a k e e p  S U B R O U T I N E  -*-

  SUBROUTINE free_akeep_precision( akeep, flag )
    IMPLICIT none
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    INTEGER( ip_ ), INTENT( OUT ) :: flag

    CALL akeep%free( flag )
  END SUBROUTINE free_akeep_precision

!-*-*-  G A L A H A D - S S I D S _ f r e e _f k e e p  S U B R O U T I N E  -*-

  SUBROUTINE free_fkeep_precision( fkeep, cuda_error )
    IMPLICIT none
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    INTEGER( ip_ ), INTENT( OUT ) :: cuda_error

    CALL fkeep%free( cuda_error )
  END SUBROUTINE free_fkeep_precision

!-*-*-  G A L A H A D - S S I D S _ f r e e _ b o t h  S U B R O U T I N E  -*-

  SUBROUTINE free_both_precision( akeep, fkeep, cuda_error )
    IMPLICIT none
    TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
    TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
    INTEGER( ip_ ), INTENT( OUT ) :: cuda_error

!  NB: Must free fkeep first as it may reference akeep
    CALL free_fkeep_precision( fkeep, cuda_error )
    IF ( cuda_error /= 0 ) RETURN
    CALL free_akeep_precision( akeep, cuda_error )
    IF ( cuda_error /= 0 ) RETURN
  END SUBROUTINE free_both_precision

!-*-*-  G A L A H A D - S S I D S _ push_omp_settings S U B R O U T I N E  -*-*-

  SUBROUTINE push_omp_settings( user_settings, flag )

!  ensure OpenMP ICVs are as required, and store user versions for

    IMPLICIT none
    TYPE( omp_settings ), INTENT( OUT ) :: user_settings
    INTEGER( ip_ ), INTENT( INOUT ) :: flag
!   CHARACTER( LEN = 255 ) :: OMP_CANCELLATION_VAR
!   LOGICAL :: ompgc
!   CALL get_environment_variable( "OMP_CANCELLATION", OMP_CANCELLATION_VAR )
!   WRITE ( *,* ) 'OMP_CANCELLATION = ', TRIM( OMP_CANCELLATION_VAR )
!!$  ompgc = omp_get_cancellation( )
!!$  WRITE( 6,* ) ' ompgc ', ompgc
!  Dummy, for now.
!   user_settings%nested = .true.
    user_settings%max_active_levels = huge( user_settings%max_active_levels )

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
!$  IF ( user_settings%max_active_levels < 2 ) CALL omp_set_max_active_levels( 2 )
  END SUBROUTINE push_omp_settings

!-*-*-  G A L A H A D - S S I D S _  pop_omp_settings S U B R O U T I N E  -*-*-

  SUBROUTINE pop_omp_settings( user_settings )

!  restore user OpenMP ICV values.

    IMPLICIT none
    TYPE( omp_settings ), INTENT( IN ) :: user_settings

!!$  IF ( .NOT. user_settings%nested ) CALL omp_set_nested( user_settings%nested )
!$  IF ( user_settings%dynamic ) CALL omp_set_dynamic( user_settings%dynamic )
!$  IF ( user_settings%max_active_levels < 2 ) &
!$       CALL omp_set_max_active_levels( int( user_settings%max_active_levels ) )
  END SUBROUTINE pop_omp_settings

END MODULE GALAHAD_SSIDS_precision
