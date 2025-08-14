! THIS VERSION: GALAHAD 5.3 - 2025-08-14 AT 13:10 GMT

#ifdef REAL_32
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_single_64
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_single_64
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_single_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_single_ciface_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_single
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_single
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_single_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_single_ciface
#endif
#elif REAL_128
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple_64
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_quadruple_64
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_quadruple_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_quadruple_ciface_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_quadruple
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_quadruple_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_quadruple_ciface
#endif
#else
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_double_64
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_double_64
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_double_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_double_ciface_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_double
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_double
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_double_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_double_ciface
#endif
#endif

#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ S S I D S   C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.4. January 3rd 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module to GALAHAD_SSIDS types and interfaces

  MODULE GALAHAD_SSIDS_precision_ciface
    USE SPRAL_KINDS_precision
    USE GALAHAD_SSIDS_precision, ONLY : f_ssids_analyse => ssids_analyse,      &
                                        f_ssids_analyse_coord                  &
                                          => ssids_analyse_coord,              &
                                        f_ssids_factor => ssids_factor,        &
                                        f_ssids_solve => ssids_solve,          &
                                        f_ssids_free => ssids_free,            &
                                        f_ssids_enquire_posdef                 &
                                          => ssids_enquire_posdef,             &
                                        f_ssids_enquire_indef                  &
                                          => ssids_enquire_indef,              &
                                        f_ssids_alter => ssids_alter,          &
                                        f_ssids_control_type                   &
                                          => ssids_control_type,               &
                                        f_ssids_inform_type                    &
                                          => ssids_inform_type,                &
                                        f_ssids_akeep_type                     &
                                          => ssids_akeep_type,                 &
                                        f_ssids_fkeep_type                     &
                                          => ssids_fkeep_type
    USE GALAHAD_NODEND_precision_ciface, ONLY:                                 &
        nodend_inform_type, nodend_control_type,                               &
        copy_nodend_control_in => copy_control_in,                             &
        copy_nodend_control_out => copy_control_out,                           &
        copy_nodend_inform_out => copy_inform_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: galahad_ssids_control_type
       INTEGER ( KIND = ipc_ ) :: array_base
       INTEGER ( KIND = ipc_ ) :: print_level
       INTEGER ( KIND = ipc_ ) :: unit_diagnostics
       INTEGER ( KIND = ipc_ ) :: unit_error
       INTEGER ( KIND = ipc_ ) :: unit_warning
       INTEGER ( KIND = ipc_ ) :: ordering
       INTEGER ( KIND = ipc_ ) :: nemin
       LOGICAL ( KIND = C_BOOL ) :: ignore_numa
       LOGICAL ( KIND = C_BOOL ) :: use_gpu
       LOGICAL ( KIND = C_BOOL ) :: gpu_only
       INTEGER ( KIND = longc_ ) :: min_gpu_work
       REAL ( KIND = spc_ ) :: max_load_inbalance
       REAL ( KIND = spc_ ) :: gpu_perf_coeff
       INTEGER ( KIND = ipc_ ) :: scaling
       INTEGER ( KIND = longc_ ) :: small_subtree_threshold
       INTEGER ( KIND = ipc_ ) :: cpu_block_size
       LOGICAL ( KIND = C_BOOL ) :: action
       INTEGER ( KIND = ipc_ ) :: pivot_method
       REAL ( KIND = rpc_ ) :: small
       REAL ( KIND = rpc_ ) :: u
       TYPE ( nodend_control_type ) :: nodend_control
       INTEGER ( KIND = ipc_ ) :: nstream
       REAL ( KIND = rpc_ ) :: multiplier
!     type(auction_control) :: auction
       REAL ( KIND = spc_ ) :: min_loadbalance
!    character(len=:), allocatable :: rb_dump
       INTEGER ( KIND = ipc_ ) :: failed_pivot_method
    END TYPE galahad_ssids_control_type

    TYPE, BIND( C ) :: galahad_ssids_inform_type
       INTEGER ( KIND = ipc_ ) :: flag
       INTEGER ( KIND = ipc_ ) :: matrix_dup
       INTEGER ( KIND = ipc_ ) :: matrix_missing_diag
       INTEGER ( KIND = ipc_ ) :: matrix_outrange
       INTEGER ( KIND = ipc_ ) :: matrix_rank
       INTEGER ( KIND = ipc_ ) :: maxdepth
       INTEGER ( KIND = ipc_ ) :: maxfront
       INTEGER ( KIND = ipc_ ) :: maxsupernode
       INTEGER ( KIND = ipc_ ) :: num_delay
       INTEGER ( KIND = longc_ ) :: num_factor
       INTEGER ( KIND = longc_ ) :: num_flops
       INTEGER ( KIND = ipc_ ) :: num_neg
       INTEGER ( KIND = ipc_ ) :: num_sup
       INTEGER ( KIND = ipc_ ) :: num_two
       INTEGER ( KIND = ipc_ ) :: stat
!    type(auction_inform) :: auction
       INTEGER ( KIND = ipc_ ) :: cuda_error
       INTEGER ( KIND = ipc_ ) :: cublas_error
       TYPE ( nodend_inform_type ) :: nodend_inform
       INTEGER ( KIND = ipc_ ) :: not_first_pass
       INTEGER ( KIND = ipc_ ) :: not_second_pass
       INTEGER ( KIND = ipc_ ) :: nparts
       INTEGER ( KIND = longc_ ) :: cpu_flops
       INTEGER ( KIND = longc_ ) :: gpu_flops
!      CHARACTER(C_CHAR) :: unused(76)
    END TYPE galahad_ssids_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C options parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, cindexed )
    TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_ssids_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, INTENT( OUT ) :: cindexed

    cindexed = ccontrol%array_base == 0
    fcontrol%print_level = ccontrol%print_level
    fcontrol%unit_diagnostics = ccontrol%unit_diagnostics
    fcontrol%unit_error = ccontrol%unit_error
    fcontrol%unit_warning = ccontrol%unit_warning
    fcontrol%ordering = ccontrol%ordering
    fcontrol%nemin = ccontrol%nemin
    fcontrol%ignore_numa = ccontrol%ignore_numa
    fcontrol%use_gpu = ccontrol%use_gpu
    fcontrol%gpu_only = ccontrol%gpu_only
    fcontrol%min_gpu_work = ccontrol%min_gpu_work
    fcontrol%max_load_inbalance = ccontrol%max_load_inbalance
    fcontrol%gpu_perf_coeff = ccontrol%gpu_perf_coeff
    fcontrol%scaling = ccontrol%scaling
    fcontrol%small_subtree_threshold = ccontrol%small_subtree_threshold
    fcontrol%cpu_block_size = ccontrol%cpu_block_size
    fcontrol%action = ccontrol%action
    fcontrol%pivot_method = ccontrol%pivot_method
    fcontrol%small = ccontrol%small
    fcontrol%u = ccontrol%u
    CALL copy_nodend_control_in( ccontrol%nodend_control,                      &
                                 fcontrol%nodend_control )
    fcontrol%nstream = ccontrol%nstream
    fcontrol%multiplier = ccontrol%multiplier
    fcontrol%min_loadbalance = ccontrol%min_loadbalance
    fcontrol%failed_pivot_method = ccontrol%failed_pivot_method
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_ssids_inform_type ), INTENT( IN ) :: finform
    TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

    cinform%flag = finform%flag
    cinform%matrix_dup = finform%matrix_dup
    cinform%matrix_missing_diag = finform%matrix_missing_diag
    cinform%matrix_outrange = finform%matrix_outrange
    cinform%matrix_rank = finform%matrix_rank
    cinform%maxdepth = finform%maxdepth
    cinform%maxfront = finform%maxfront
    cinform%maxsupernode = finform%maxsupernode
    cinform%num_delay = finform%num_delay
    cinform%num_factor = finform%num_factor
    cinform%num_flops = finform%num_flops
    cinform%num_neg = finform%num_neg
    cinform%num_sup = finform%num_sup
    cinform%num_two = finform%num_two
    cinform%stat = finform%stat
    cinform%cuda_error = finform%cuda_error
    cinform%cublas_error = finform%cublas_error
    CALL copy_nodend_inform_out( finform%nodend_inform, cinform%nodend_inform )
    cinform%not_first_pass = finform%not_first_pass
    cinform%not_second_pass = finform%not_second_pass
    cinform%nparts = finform%nparts
    cinform%cpu_flops = finform%cpu_flops
    cinform%gpu_flops = finform%gpu_flops
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SSIDS_precision_ciface

!  ------------------
!  Revitalize options
!  ------------------

  SUBROUTINE galahad_ssids_default_control( ccontrol ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( galahad_ssids_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_ssids_control_type ) :: default_control

  ccontrol%array_base              = 0 ! C
  ccontrol%print_level             = default_control%print_level
  ccontrol%unit_diagnostics        = default_control%unit_diagnostics
  ccontrol%unit_error              = default_control%unit_error
  ccontrol%unit_warning            = default_control%unit_warning
  ccontrol%ordering                = default_control%ordering
  ccontrol%nemin                   = default_control%nemin
  ccontrol%ignore_numa             = default_control%ignore_numa
  ccontrol%use_gpu                 = default_control%use_gpu
  ccontrol%min_gpu_work            = default_control%min_gpu_work
  ccontrol%max_load_inbalance      = default_control%max_load_inbalance
  ccontrol%gpu_perf_coeff          = default_control%gpu_perf_coeff
  ccontrol%scaling                 = default_control%scaling
  ccontrol%small_subtree_threshold = default_control%small_subtree_threshold
  ccontrol%cpu_block_size          = default_control%cpu_block_size
  ccontrol%action                  = default_control%action
  ccontrol%pivot_method            = default_control%pivot_method
  ccontrol%small                   = default_control%small
  ccontrol%u                       = default_control%u
  CALL copy_nodend_control_out( default_control%nodend_control,                &
                                ccontrol%nodend_control )
  ccontrol%nstream                 = default_control%nstream
  ccontrol%multiplier              = default_control%multiplier
  ccontrol%min_loadbalance         = default_control%min_loadbalance
  ccontrol%failed_pivot_method     = default_control%failed_pivot_method
  END SUBROUTINE galahad_ssids_default_control

!  ------------------------------------
!  C interface to fortran ssids_analyse
!  ------------------------------------

  SUBROUTINE galahad_ssids_analyse( ccheck, n, corder, cptr, crow, cval,       &
                                  cakeep, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: ccheck
  INTEGER ( KIND = ipc_ ), VALUE :: n
  TYPE ( C_PTR ), value :: corder
  INTEGER ( KIND = longc_ ), TARGET, DIMENSION( n + 1 ) :: cptr
  TYPE ( C_PTR ), value :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform
  INTEGER ( KIND = ipc_ ), TARGET,                                             &
    DIMENSION( cptr( n + 1 ) - ccontrol%array_base ) :: crow

!  local variables

  INTEGER ( KIND = longc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = longc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  LOGICAL :: fcheck
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: forder
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: forder_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fval
  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform
  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  fcheck = ccheck
  IF ( C_ASSOCIATED( corder ) ) THEN
    CALL C_F_POINTER( corder, forder, shape = (/ n /) )
  ELSE
    NULLIFY( forder )
  END IF
  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    ALLOCATE( forder_alloc( n ) )
    forder_alloc( : ) = forder( : ) + 1
    forder => forder_alloc
  END IF
  fptr => cptr
  IF ( cindexed ) THEN
    ALLOCATE( fptr_alloc( n + 1 ) )
    fptr_alloc( : ) = fptr( : ) + 1
    fptr => fptr_alloc
  END IF
  frow => crow
  IF ( cindexed ) THEN
    ALLOCATE( frow_alloc( fptr( n + 1 ) - 1 ) )
    frow_alloc( : ) = frow( : ) + 1
    frow => frow_alloc
  END IF
  IF ( C_ASSOCIATED( cval ) ) THEN
    CALL C_F_POINTER( cval, fval, shape = (/ fptr( n + 1 ) - 1 /) )
  ELSE
    NULLIFY( fval )
  END IF

!  reuse old pointer

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )

!  create new pointer
   
  ELSE
    ALLOCATE( fakeep )
    cakeep = C_LOC( fakeep )
  END IF

!  call fortran routine

write(99,*) ' n ', n
write(99,*) ' fptr ', fptr( : n + 1 )
write(99,*) ' frow ', frow( : fptr( n + 1 ) - 1 )
close(99)
  IF ( ASSOCIATED( forder ) ) THEN
    IF ( ASSOCIATED( fval ) ) THEN
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform,  &
                            order = forder, val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform,  &
                            order = forder )
    END IF
  ELSE
    IF (ASSOCIATED(fval)) THEN
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform,  &
                            val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform )
    END IF
  END IF

!  copy arguments out

  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    CALL C_F_POINTER( corder, forder, shape = (/ n /) )
    forder( : ) = forder_alloc( : ) - 1
  END IF
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_analyse

!  ---------------------------------------------------------
!  C interface to fortrans sids_analyse with 32-bit pointers
!  ---------------------------------------------------------

  SUBROUTINE galahad_ssids_analyse_ptr32( ccheck, n, corder, cptr, crow, cval, &
                                        cakeep, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: ccheck
  INTEGER ( KIND = ipc_ ), VALUE :: n
  TYPE ( C_PTR ), VALUE :: corder
  INTEGER ( KIND = ipc_ ), TARGET, DIMENSION( n + 1 ) :: cptr
  TYPE ( C_PTR ), VALUE :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform
  INTEGER ( KIND = ipc_ ), TARGET,                                             &
    DIMENSION( cptr( n + 1 ) - ccontrol%array_base ) :: crow

!  local variables

  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  LOGICAL :: fcheck
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: forder
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: forder_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fval
  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in(ccontrol, fcontrol, cindexed)

!  translate arguments

  fcheck = ccheck
  IF ( C_ASSOCIATED( corder ) ) THEN
    CALL C_F_POINTER( corder, forder, shape = (/ n /) )
  ELSE
    NULLIFY( forder )
  END IF
  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    ALLOCATE( forder_alloc( n ) )
    forder_alloc( : ) = forder( : ) + 1
    forder => forder_alloc
  END IF
  fptr => cptr
  IF ( cindexed ) THEN
    ALLOCATE( fptr_alloc( n + 1 ) )
    fptr_alloc( : ) = fptr( : ) + 1
    fptr => fptr_alloc
  END IF
  frow => crow
  IF ( cindexed ) THEN
    ALLOCATE( frow_alloc( fptr( n + 1 ) - 1 ) )
    frow_alloc( : ) = frow( : ) + 1
    frow => frow_alloc
  END IF
  IF ( C_ASSOCIATED( cval ) ) THEN
    CALL C_F_POINTER( cval, fval, shape = (/ fptr( n + 1 ) - 1 /) )
  ELSE
    NULLIFY( fval )
  END IF

!  reuse old pointer

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER(cakeep, fakeep)

!  create new pointer

  ELSE
    ALLOCATE( fakeep)
    cakeep = C_LOC( fakeep)
  END IF

!  call fortran routine

  IF ( ASSOCIATED( forder ) ) THEN
    IF ( ASSOCIATED( fval ) ) THEN
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform,  &
                            order = forder, val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform,  &
                            order = forder )
    END IF
  ELSE
    IF ( ASSOCIATED( fval ) ) THEN
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform,  &
                            val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, fcontrol, finform )
    END IF
  END IF

!  copy arguments out

  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    CALL C_F_POINTER(corder, forder, shape = (/ n /) )
    forder( : ) = forder_alloc( : ) - 1
  END IF
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_analyse_ptr32

!  ------------------------------------------
!  C interface to fortrans sids_analyse_coord
!  ------------------------------------------

  SUBROUTINE galahad_ssids_analyse_coord( n, corder, ne, crow, ccol, cval,     &
                                          cakeep, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), VALUE :: n
  TYPE ( C_PTR ), VALUE :: corder
  INTEGER ( KIND = longc_ ), VALUE :: ne
  INTEGER ( KIND = ipc_ ), TARGET, DIMENSION( ne ) :: crow
  INTEGER ( KIND = ipc_ ), TARGET, DIMENSION( ne ) :: ccol
  TYPE ( C_PTR ), VALUE :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fcol
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fcol_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: forder
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: forder_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fval
  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in(ccontrol, fcontrol, cindexed)

!  translate arguments

  IF ( C_ASSOCIATED( corder ) ) THEN
    CALL C_F_POINTER( corder, forder, shape=(/ n /) )
  ELSE
     NULLIFY( forder )
  END IF
  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    ALLOCATE( forder_alloc( n ) )
    forder_alloc( : ) = forder( : ) + 1
    forder => forder_alloc
  END IF
  frow => crow
  if ( cindexed ) then
    ALLOCATE( frow_alloc( ne ) )
    frow_alloc( : ) = frow( : ) + 1
    frow => frow_alloc
  END IF
  fcol => ccol
  IF (cindexed) THEN
    ALLOCATE( fcol_alloc( ne ) )
    fcol_alloc( : ) = fcol( : ) + 1
    fcol => fcol_alloc
  END IF
  IF ( C_ASSOCIATED( cval ) ) THEN
    CALL C_F_POINTER( cval, fval, shape=(/ ne /) )
  ELSE
    NULLIFY( fval )
  END IF
 
!  reuse old pointer

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )

!  create new pointer

  ELSE
    ALLOCATE( fakeep )
    cakeep = C_LOC( fakeep )
  END IF

!  call fortran routine

  IF ( ASSOCIATED( forder ) ) THEN
    IF ( ASSOCIATED( fval ) ) THEN
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, fcontrol,         &
                                  finform,  order = forder, val = fval )
    ELSE
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, fcontrol,         &
                                  finform, order = forder )
    END IF
  ELSE
    IF ( ASSOCIATED( fval ) ) THEN
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, fcontrol,         &
                                  finform, val = fval )
    ELSE
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, fcontrol, finform )
    END IF
  END IF

!  copy arguments out

  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    CALL C_F_POINTER(corder, forder, shape = (/ n /) )
    forder( : ) = forder_alloc( : ) - 1
  END IF
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_analyse_coord

!  -----------------------------------
!  C interface to fortran ssids_factor
!  -----------------------------------

  SUBROUTINE galahad_ssids_factor( cposdef, cptr, crow, val, cscale, cakeep,   &
                                   cfkeep, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: cposdef
  TYPE ( C_PTR ), VALUE :: cptr
  TYPE ( C_PTR ), VALUE :: crow
  REAL ( KIND = rpc_ ), DIMENSION( * ), INTENT( IN ) :: val
  TYPE ( C_PTR ), VALUE :: cscale
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  LOGICAL :: fposdef
  INTEGER ( KIND = longc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = longc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fscale
  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  fposdef = cposdef
  CALL C_F_POINTER( cakeep, fakeep ) ! Pulled forward so we can use it
  IF ( C_ASSOCIATED( cptr )  .AND.  C_ASSOCIATED( crow ) ) THEN
    CALL C_F_POINTER( cptr, fptr, shape = (/ fakeep%n + 1 /) )
    IF ( cindexed ) THEN
       ALLOCATE( fptr_alloc( fakeep%n + 1 ) )
       fptr_alloc( : ) = fptr( : ) + 1
       fptr => fptr_alloc
    END IF
    CALL C_F_POINTER( crow, frow, shape = (/ fptr( fakeep%n + 1 ) - 1 /) )
    IF ( cindexed ) THEN
       ALLOCATE( frow_alloc( fptr(fakeep%n + 1 ) - 1 ) )
       frow_alloc( : ) = frow( : ) + 1
       frow => frow_alloc
    END IF
  ELSE
    NULLIFY( fptr, frow )
  end if
  IF ( C_ASSOCIATED( cscale ) ) THEN
    CALL C_F_POINTER( cscale, fscale, shape = (/ fakeep%n /) )
  ELSE
    NULLIFY( fscale )
  END IF

!  reuse old pointer

  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )

!  create new pointer

  ELSE
    ALLOCATE( ffkeep)
    cfkeep = C_LOC( ffkeep )
  END IF

!  call fortran routine

write(99, "( ' before factor ' )" )

  IF ( ASSOCIATED( fptr ) .AND. ASSOCIATED( frow ) ) THEN
    IF ( ASSOCIATED( fscale ) ) THEN
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform,    &
                           ptr = fptr, row = frow, scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform,    &
                           ptr = fptr, row = frow )
    END IF
  ELSE
    IF ( ASSOCIATED( fscale ) ) THEN
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform,    &
                           scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform )
    END IF
  END IF
write(99, "( ' after factor ' )" )
close(99)

!  copy arguments out

  CALL copy_inform_out(finform, cinform)

  END SUBROUTINE galahad_ssids_factor

!  --------------------------------------------------------
!  C interface to fortran ssids_factor with 32-bit pointers
!  --------------------------------------------------------

  SUBROUTINE galahad_ssids_factor_ptr32( cposdef, cptr, crow, val, cscale,     &
                                         cakeep, cfkeep, ccontrol,             &
                                         cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: cposdef
  TYPE ( C_PTR ), VALUE :: cptr
  TYPE ( C_PTR ), VALUE :: crow
  REAL ( KIND = rpc_ ), DIMENSION( * ), INTENT( IN ) :: val
  TYPE ( C_PTR ), VALUE :: cscale
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  LOGICAL :: fposdef
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fscale
  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  fposdef = cposdef
  CALL C_F_POINTER( cakeep, fakeep ) ! Pulled forward so we can use it
  IF ( C_ASSOCIATED( cptr )  .and.  C_ASSOCIATED( crow ) ) THEN
    CALL C_F_POINTER( cptr, fptr, shape = (/ fakeep%n + 1 /) )
    IF ( cindexed ) THEN
      ALLOCATE( fptr_alloc( fakeep%n+1 ) )
      fptr_alloc( : ) = fptr( : ) + 1
      fptr => fptr_alloc
    END IF
    CALL C_F_POINTER( crow, frow, shape = (/ fptr(fakeep%n + 1 ) - 1 /) )
    IF ( cindexed ) THEN
      ALLOCATE( frow_alloc( fptr(fakeep%n+1)-1 ) )
      frow_alloc( : ) = frow( : ) + 1
      frow => frow_alloc
    END IF
  ELSE
     NULLIFY( fptr, frow )
  end if
  IF ( C_ASSOCIATED( cscale ) ) THEN
    CALL C_F_POINTER( cscale, fscale, shape = (/ fakeep%n /) )
  ELSE
    NULLIFY( fscale )
  END IF

!  reuse old pointer

  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )

!  create new pointer

  ELSE
    ALLOCATE( ffkeep )
    cfkeep = C_LOC( ffkeep )
  END IF

!  call fortran routine

  IF ( ASSOCIATED( fptr ) .AND. ASSOCIATED( frow ) ) THEN
    IF ( ASSOCIATED( fscale ) ) THEN
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform,    &
                           ptr = fptr, row = frow, scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform,    &
                           ptr = fptr, row = frow )
    END IF
  ELSE
    IF ( ASSOCIATED( fscale ) ) THEN
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform,    &
                           scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, fcontrol, finform )
    END IF
  END IF

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_factor_ptr32

!  ---------------------------------------------------------------
!  C interface to fortran galahad_ssids_solve with 1 right-hand side
!  ---------------------------------------------------------------

  SUBROUTINE galahad_ssids_solve1( job, cx1, cakeep, cfkeep, ccontrol,         &
                                    cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), VALUE :: job
  REAL ( KIND = rpc_ ), TARGET, DIMENSION( * ) :: cx1
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fx1
  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )
  ELSE
    NULLIFY( fakeep )
  END IF
  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )
  ELSE
    NULLIFY( ffkeep )
  END IF
  fx1 => cx1( 1 : fakeep%n )

!  call fortran routine

  IF ( job == 0 ) THEN

!  note: job=0 is an out of range value (but is valid internally!)

     CALL f_ssids_solve( fx1, fakeep, ffkeep, fcontrol, finform )
  ELSE
     CALL f_ssids_solve( fx1, fakeep, ffkeep, fcontrol, finform, job = job )
  END IF

!  copy arguments out

  CALL copy_inform_out(finform, cinform)

  END SUBROUTINE galahad_ssids_solve1

!  -----------------------------------------------------------------------
!  C interface to fortran galahad_ssids_solve with multiple right-hand sides
!  -----------------------------------------------------------------------

  SUBROUTINE galahad_ssids_solve( job, nrhs, x, ldx, cakeep, cfkeep, ccontrol, &
                                  cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), VALUE :: job
  INTEGER ( KIND = ipc_ ), VALUE :: nrhs
  REAL ( KIND = rpc_ ), DIMENSION( ldx, nrhs ) :: x
  INTEGER ( KIND = ipc_ ), VALUE :: ldx
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )
  ELSE
    NULLIFY( fakeep )
  END IF
  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )
  ELSE
    NULLIFY( ffkeep )
  END IF

!  call fortran routine

  IF ( job == 0 ) THEN
    CALL f_ssids_solve( nrhs, x, ldx, fakeep, ffkeep, fcontrol, finform )
  ELSE
    CALL f_ssids_solve( nrhs, x, ldx, fakeep, ffkeep, fcontrol, finform,       &
                      job = job )
  END IF

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_solve

!  ------------------------------------------------
!  C interface to fortran ssids_free to free cakeep
!  ------------------------------------------------

  INTEGER ( KIND = ipc_ ) function galahad_ssids_free_akeep( cakeep ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep

!  local variables

  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep

!  nothing to free

  IF ( .NOT. C_ASSOCIATED( cakeep ) ) THEN
    galahad_ssids_free_akeep = 0
    RETURN
  END IF

  CALL C_F_POINTER( cakeep, fakeep )
  CALL f_ssids_free( fakeep, galahad_ssids_free_akeep )
  DEALLOCATE( fakeep )
  cakeep = C_NULL_PTR

  END FUNCTION galahad_ssids_free_akeep

!  ------------------------------------------------
!  C interface to fortran ssids_free to free cfkeep
!  ------------------------------------------------

  INTEGER ( KIND = ipc_ ) function galahad_ssids_free_fkeep( cfkeep ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep

!  local variables

  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep

!  nothing to free

  IF ( .NOT. C_ASSOCIATED( cfkeep ) ) THEN
    galahad_ssids_free_fkeep = 0
    RETURN
  END IF

  CALL C_F_POINTER( cfkeep, ffkeep )
  CALL f_ssids_free( ffkeep, galahad_ssids_free_fkeep )
  DEALLOCATE( ffkeep )
  cfkeep = C_NULL_PTR

  END FUNCTION galahad_ssids_free_fkeep

!  ---------------------------------
!  C interface to fortran ssids_free
!  ---------------------------------

  INTEGER ( KIND = ipc_ ) FUNCTION galahad_ssids_free( cakeep,                 &
                                                       cfkeep ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep

  INTERFACE
    INTEGER ( KIND = ipc_ )                                                    &
        FUNCTION galahad_ssids_free_akeep( cakeep ) BIND( C )
      USE iso_c_binding
      USE SPRAL_KINDS_precision
      IMPLICIT NONE
      TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
    END FUNCTION galahad_ssids_free_akeep
    INTEGER ( KIND = ipc_ )                                                    &
        FUNCTION galahad_ssids_free_fkeep( cfkeep ) BIND( C )
      USE iso_c_binding
      USE SPRAL_KINDS_precision
      IMPLICIT NONE
      TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep
    END FUNCTION galahad_ssids_free_fkeep
  END INTERFACE

  galahad_ssids_free = galahad_ssids_free_akeep( cakeep )
  IF ( galahad_ssids_free /= 0_ipc_ ) RETURN
  galahad_ssids_free = galahad_ssids_free_fkeep( cfkeep )

  END FUNCTION galahad_ssids_free

!  -------------------------------------------
!  C interface to fortran ssids_enquire_posdef
!  -------------------------------------------

  SUBROUTINE galahad_ssids_enquire_posdef( cakeep, cfkeep, ccontrol,           &
                                           cinform, d ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform
  REAL ( KIND = rpc_ ), DIMENSION( * ), INTENT( OUT ) :: d

!  local variables

  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in(ccontrol, fcontrol, cindexed)

!  translate arguments

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )
  ELSE
    NULLIFY( fakeep )
  END IF
  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )
  ELSE
    NULLIFY( ffkeep )
  END IF

!  call fortran routine

  CALL f_ssids_enquire_posdef( fakeep, ffkeep, fcontrol, finform, d )

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_enquire_posdef

!  ------------------------------------------
!  C interface to fortran ssids_enquire_indef
!  ------------------------------------------

  SUBROUTINE galahad_ssids_enquire_indef( cakeep, cfkeep, ccontrol, cinform,   &
                                          cpiv_order, cd ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform
  TYPE ( C_PTR ), VALUE :: cpiv_order
  TYPE ( C_PTR ), VALUE :: cd

!  local variables

  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fpiv_order
  REAL ( KIND = rpc_ ), DIMENSION( :,: ), POINTER :: fd

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )
  ELSE
    NULLIFY( fakeep )
  END IF
  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )
  ELSE
    NULLIFY( ffkeep )
  END IF
  IF ( C_ASSOCIATED( cpiv_order ) ) THEN
    CALL C_F_POINTER( cpiv_order, fpiv_order, shape = (/ fakeep%n /) )
  ELSE
    NULLIFY( fpiv_order )
  END IF
  IF ( C_ASSOCIATED( cd ) ) THEN
    CALL C_F_POINTER( cd, fd, shape = (/ 2_ipc_, fakeep%n /) )
  ELSE
    NULLIFY( fd )
  END IF

!  call fortran routine

  IF ( ASSOCIATED( fpiv_order ) ) THEN
    IF ( ASSOCIATED( fd ) ) THEN
      CALL f_ssids_enquire_indef( fakeep, ffkeep, fcontrol, finform,           &
                                piv_order = fpiv_order, d = fd )
    ELSE
      CALL f_ssids_enquire_indef( fakeep, ffkeep, fcontrol, finform,           &
                                piv_order=fpiv_order )
    END IF
  ELSE
    IF ( ASSOCIATED( fd ) ) THEN
      CALL f_ssids_enquire_indef( fakeep, ffkeep, fcontrol, finform, d = fd )
    ELSE
      CALL f_ssids_enquire_indef( fakeep, ffkeep, fcontrol, finform )
    END IF
  END IF

!  copy arguments out

!  note: we use abs value of piv_order in C indexing, as 0 and -0 are the same

  IF ( ASSOCIATED( fpiv_order ) .AND. cindexed )                               &
    fpiv_order( : ) = ABS( fpiv_order( : ) ) - 1
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_enquire_indef

!  ----------------------------------
!  C interface to fortran ssids_alter
!  ----------------------------------

  SUBROUTINE galahad_ssids_alter( d, cakeep, cfkeep, ccontrol,                 &
                                  cinform ) BIND( C )
  USE GALAHAD_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  REAL ( KIND = rpc_ ), DIMENSION( 2, * ), INTENT( IN ) :: d
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( galahad_ssids_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( galahad_ssids_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_ssids_akeep_type ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep_type ), POINTER :: ffkeep
  TYPE ( f_ssids_control_type ) :: fcontrol
  TYPE ( f_ssids_inform_type ) :: finform

  LOGICAL :: cindexed

!  copy control in first to find out whether we use fortran or C indexing

  CALL copy_control_in( ccontrol, fcontrol, cindexed )

!  translate arguments

  IF ( C_ASSOCIATED( cakeep ) ) THEN
    CALL C_F_POINTER( cakeep, fakeep )
  ELSE
    NULLIFY( fakeep )
  END IF
  IF ( C_ASSOCIATED( cfkeep ) ) THEN
    CALL C_F_POINTER( cfkeep, ffkeep )
  ELSE
    NULLIFY( ffkeep )
  END IF

!  call fortran routine

  CALL f_ssids_alter( d, fakeep, ffkeep, fcontrol, finform )

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE galahad_ssids_alter
