! THIS VERSION: GALAHAD 5.3 - 2025-08-07 AT 12:00 GMT

#ifdef REAL_32
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_single_64
#define SPRAL_SSIDS_precision SPRAL_SSIDS_single_64
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_single_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_single_ciface_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_single
#define SPRAL_SSIDS_precision SPRAL_SSIDS_single
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_single_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_single_ciface
#endif
#elif REAL_128
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple_64
#define SPRAL_SSIDS_precision SPRAL_SSIDS_quadruple_64
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_quadruple_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_quadruple_ciface_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple
#define SPRAL_SSIDS_precision SPRAL_SSIDS_quadruple
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_quadruple_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_quadruple_ciface
#endif
#else
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_double_64
#define SPRAL_SSIDS_precision SPRAL_SSIDS_double_64
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_double_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_double_ciface_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_double
#define SPRAL_SSIDS_precision SPRAL_SSIDS_double
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_double_ciface
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

!  C interface module to SPRAL_SSIDS types and interfaces

  MODULE SPRAL_SSIDS_precision_ciface
    USE SPRAL_KINDS_precision
    USE SPRAL_SSIDS_precision, ONLY : f_ssids_analyse => ssids_analyse,        &
                                      f_ssids_analyse_coord                    &
                                        => ssids_analyse_coord,                &
                                      f_ssids_factor => ssids_factor,          &
                                      f_ssids_solve => ssids_solve,            &
                                      f_ssids_free => ssids_free,              &
                                      f_ssids_enquire_posdef                   &
                                        => ssids_enquire_posdef,               &
                                      f_ssids_enquire_indef                    &
                                        => ssids_enquire_indef,                &
                                      f_ssids_alter => ssids_alter,            &
                                      f_ssids_options => ssids_options,        &
                                      f_ssids_inform => ssids_inform,          &
                                      f_ssids_akeep => ssids_akeep,            &
                                      f_ssids_fkeep => ssids_fkeep
   USE GALAHAD_NODEND_precision_ciface, ONLY:                                  &
        nodend_inform_type, nodend_control_type,                               &
        copy_nodend_options_in => copy_control_in,                             &
        copy_nodend_options_out => copy_control_out,                           &
        copy_nodend_inform_out => copy_inform_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: spral_ssids_options
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
       TYPE ( nodend_control_type ) :: nodend_options
       INTEGER ( KIND = ipc_ ) :: nstream
       REAL ( KIND = rpc_ ) :: multiplier
!     type(auction_options) :: auction
       REAL ( KIND = spc_ ) :: min_loadbalance
!    character(len=:), allocatable :: rb_dump
       INTEGER ( KIND = ipc_ ) :: failed_pivot_method
    END TYPE spral_ssids_options

    TYPE, BIND( C ) :: spral_ssids_inform
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
    END TYPE spral_ssids_inform

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C options parameters to fortran

    SUBROUTINE copy_options_in( coptions, foptions, cindexed )
    TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
    TYPE ( f_ssids_options ), INTENT( OUT ) :: foptions
    LOGICAL, INTENT( OUT ) :: cindexed

    cindexed = coptions%array_base == 0
    foptions%print_level = coptions%print_level
    foptions%unit_diagnostics = coptions%unit_diagnostics
    foptions%unit_error = coptions%unit_error
    foptions%unit_warning = coptions%unit_warning
    foptions%ordering = coptions%ordering
    foptions%nemin = coptions%nemin
    foptions%ignore_numa = coptions%ignore_numa
    foptions%use_gpu = coptions%use_gpu
    foptions%gpu_only = coptions%gpu_only
    foptions%min_gpu_work = coptions%min_gpu_work
    foptions%max_load_inbalance = coptions%max_load_inbalance
    foptions%gpu_perf_coeff = coptions%gpu_perf_coeff
    foptions%scaling = coptions%scaling
    foptions%small_subtree_threshold = coptions%small_subtree_threshold
    foptions%cpu_block_size = coptions%cpu_block_size
    foptions%action = coptions%action
    foptions%pivot_method = coptions%pivot_method
    foptions%small = coptions%small
    foptions%u = coptions%u
    CALL copy_nodend_options_in( coptions%nodend_options,                      &
                                 foptions%nodend_options )
    foptions%nstream = coptions%nstream
    foptions%multiplier = coptions%multiplier
    foptions%min_loadbalance = coptions%min_loadbalance
    foptions%failed_pivot_method = coptions%failed_pivot_method
    RETURN

    END SUBROUTINE copy_options_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_ssids_inform ), INTENT( IN ) :: finform
    TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

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

  END MODULE SPRAL_SSIDS_precision_ciface

!  ------------------
!  Revitalize options
!  ------------------

  SUBROUTINE spral_ssids_default_options( coptions ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( spral_ssids_options ), INTENT( OUT ) :: coptions

!  local variables

  TYPE ( f_ssids_options ) :: default_options

  coptions%array_base              = 0 ! C
  coptions%print_level             = default_options%print_level
  coptions%unit_diagnostics        = default_options%unit_diagnostics
  coptions%unit_error              = default_options%unit_error
  coptions%unit_warning            = default_options%unit_warning
  coptions%ordering                = default_options%ordering
  coptions%nemin                   = default_options%nemin
  coptions%ignore_numa             = default_options%ignore_numa
  coptions%use_gpu                 = default_options%use_gpu
  coptions%min_gpu_work            = default_options%min_gpu_work
  coptions%max_load_inbalance      = default_options%max_load_inbalance
  coptions%gpu_perf_coeff          = default_options%gpu_perf_coeff
  coptions%scaling                 = default_options%scaling
  coptions%small_subtree_threshold = default_options%small_subtree_threshold
  coptions%cpu_block_size          = default_options%cpu_block_size
  coptions%action                  = default_options%action
  coptions%pivot_method            = default_options%pivot_method
  coptions%small                   = default_options%small
  coptions%u                       = default_options%u
  CALL copy_nodend_options_out( default_options%nodend_options,                &
                                coptions%nodend_options )
  coptions%nstream                 = default_options%nstream
  coptions%multiplier              = default_options%multiplier
  coptions%min_loadbalance         = default_options%min_loadbalance
  coptions%failed_pivot_method     = default_options%failed_pivot_method
  END SUBROUTINE spral_ssids_default_options

!  ------------------------------------
!  C interface to fortran ssids_analyse
!  ------------------------------------

  SUBROUTINE spral_ssids_analyse( ccheck, n, corder, cptr, crow, cval,         &
                                  cakeep, coptions, cinform ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: ccheck
  INTEGER ( KIND = ipc_ ), VALUE :: n
  TYPE ( C_PTR ), value :: corder
  INTEGER ( KIND = longc_ ), TARGET, DIMENSION( n + 1 ) :: cptr
  TYPE ( C_PTR ), value :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform
  INTEGER ( KIND = ipc_ ), TARGET,                                             &
    DIMENSION( cptr( n + 1 ) - coptions%array_base ) :: crow

!  local variables

  INTEGER ( KIND = longc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = longc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  LOGICAL :: fcheck
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: forder
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: forder_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fval
  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform
  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

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
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform,  &
                            order = forder, val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform,  &
                            order = forder )
    END IF
  ELSE
    IF (ASSOCIATED(fval)) THEN
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform,  &
                            val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform )
    END IF
  END IF

!  copy arguments out

  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    CALL C_F_POINTER( corder, forder, shape = (/ n /) )
    forder( : ) = forder_alloc( : ) - 1
  END IF
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_analyse

!  ---------------------------------------------------------
!  C interface to fortrans sids_analyse with 32-bit pointers
!  ---------------------------------------------------------

  SUBROUTINE spral_ssids_analyse_ptr32( ccheck, n, corder, cptr, crow, cval,   &
                                        cakeep, coptions, cinform ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: ccheck
  INTEGER ( KIND = ipc_ ), VALUE :: n
  TYPE ( C_PTR ), VALUE :: corder
  INTEGER ( KIND = ipc_ ), TARGET, DIMENSION( n + 1 ) :: cptr
  TYPE ( C_PTR ), VALUE :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform
  INTEGER ( KIND = ipc_ ), TARGET,                                             &
    DIMENSION( cptr( n + 1 ) - coptions%array_base ) :: crow

!  local variables

  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  LOGICAL :: fcheck
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: forder
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: forder_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fval
  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in(coptions, foptions, cindexed)

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
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform,  &
                            order = forder, val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform,  &
                            order = forder )
    END IF
  ELSE
    IF ( ASSOCIATED( fval ) ) THEN
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform,  &
                            val = fval )
    ELSE
      CALL f_ssids_analyse( fcheck, n, fptr, frow, fakeep, foptions, finform )
    END IF
  END IF

!  copy arguments out

  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    CALL C_F_POINTER(corder, forder, shape = (/ n /) )
    forder( : ) = forder_alloc( : ) - 1
  END IF
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_analyse_ptr32

!  ------------------------------------------
!  C interface to fortrans sids_analyse_coord
!  ------------------------------------------

  SUBROUTINE spral_ssids_analyse_coord( n, corder, ne, crow, ccol, cval,       &
                                        cakeep, coptions, cinform ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), VALUE :: n
  TYPE ( C_PTR ), VALUE :: corder
  INTEGER ( KIND = longc_ ), VALUE :: ne
  INTEGER ( KIND = ipc_ ), TARGET, DIMENSION( ne ) :: crow
  INTEGER ( KIND = ipc_ ), TARGET, DIMENSION( ne ) :: ccol
  TYPE ( C_PTR ), VALUE :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

!  local variables

  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fcol
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fcol_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: forder
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: forder_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fval
  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in(coptions, foptions, cindexed)

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
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, foptions,         &
                                  finform,  order = forder, val = fval )
    ELSE
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, foptions,         &
                                  finform, order = forder )
    END IF
  ELSE
    IF (ASSOCIATED(fval)) THEN
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, foptions,         &
                                  finform, val = fval )
    ELSE
      CALL f_ssids_analyse_coord( n, ne, frow, fcol, fakeep, foptions, finform )
    END IF
  END IF

!  copy arguments out

  IF ( ASSOCIATED( forder ) .AND. cindexed ) THEN
    CALL C_F_POINTER(corder, forder, shape = (/ n /) )
    forder( : ) = forder_alloc( : ) - 1
  END IF
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_analyse_coord

!  -----------------------------------
!  C interface to fortran ssids_factor
!  -----------------------------------

  SUBROUTINE spral_ssids_factor( cposdef, cptr, crow, val, cscale, cakeep,     &
                                 cfkeep, coptions, cinform ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: cposdef
  TYPE ( C_PTR ), VALUE :: cptr
  TYPE ( C_PTR ), VALUE :: crow
  REAL ( KIND = rpc_ ), DIMENSION( * ), INTENT( IN ) :: val
  TYPE ( C_PTR ), VALUE :: cscale
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

!  local variables

  LOGICAL :: fposdef
  INTEGER ( KIND = longc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = longc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fscale
  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

!  translate arguments

  fposdef = cposdef
  CALL C_F_POINTER( cakeep, fakeep ) ! Pulled forward so we can use it
  IF ( C_ASSOCIATED( cptr )  .AND.  C_ASSOCIATED( crow ) ) THEN
    CALL C_F_POINTER( cptr, fptr, shape = (/ fakeep%n + 1 /) )
    if (cindexed) then
       ALLOCATE( fptr_alloc( fakeep%n + 1 ) )
       fptr_alloc( : ) = fptr( : ) + 1
       fptr => fptr_alloc
    END IF
    CALL C_F_POINTER( crow, frow, shape = (/ fptr( fakeep%n + 1 ) - 1 /) )
    IF (cindexed) THEN
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
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform,    &
                           ptr = fptr, row = frow, scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform,    &
                           ptr = fptr, row = frow )
    END IF
  ELSE
    IF ( ASSOCIATED( fscale ) ) THEN
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform,    &
                           scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform )
    END IF
  END IF
write(99, "( ' after factor ' )" )
close(99)

!  copy arguments out

  CALL copy_inform_out(finform, cinform)

  END SUBROUTINE spral_ssids_factor

!  --------------------------------------------------------
!  C interface to fortran ssids_factor with 32-bit pointers
!  --------------------------------------------------------

  SUBROUTINE spral_ssids_factor_ptr32( cposdef, cptr, crow, val, cscale,       &
                                       cakeep, cfkeep, coptions, cinform )     &
                                       BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  LOGICAL ( KIND = C_BOOL ), VALUE :: cposdef
  TYPE ( C_PTR ), VALUE :: cptr
  TYPE ( C_PTR ), VALUE :: crow
  REAL ( KIND = rpc_ ), DIMENSION( * ), INTENT( IN ) :: val
  TYPE ( C_PTR ), VALUE :: cscale
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

!  local variables

  LOGICAL :: fposdef
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fptr
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: fptr_alloc
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: frow
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), ALLOCATABLE, TARGET :: frow_alloc
  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fscale
  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

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
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform,    &
                           ptr = fptr, row = frow, scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform,    &
                           ptr = fptr, row = frow )
    END IF
  ELSE
    IF ( ASSOCIATED( fscale ) ) THEN
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform,    &
                           scale = fscale )
    ELSE
      CALL f_ssids_factor( fposdef, val, fakeep, ffkeep, foptions, finform )
    END IF
  END IF

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_factor_ptr32

!  ---------------------------------------------------------------
!  C interface to fortran spral_ssids_solve with 1 right-hand side
!  ---------------------------------------------------------------

  SUBROUTINE spral_ssids_solve1( job, cx1, cakeep, cfkeep, coptions, cinform ) &
                                 BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), VALUE :: job
  REAL ( KIND = rpc_ ), TARGET, DIMENSION( * ) :: cx1
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

!  local variables

  REAL ( KIND = rpc_ ), DIMENSION( : ), POINTER :: fx1
  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

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

     CALL f_ssids_solve( fx1, fakeep, ffkeep, foptions, finform )
  ELSE
     CALL f_ssids_solve( fx1, fakeep, ffkeep, foptions, finform, job = job )
  END IF

!  copy arguments out

  CALL copy_inform_out(finform, cinform)

  END SUBROUTINE spral_ssids_solve1

!  -----------------------------------------------------------------------
!  C interface to fortran spral_ssids_solve with multiple right-hand sides
!  -----------------------------------------------------------------------

  SUBROUTINE spral_ssids_solve( job, nrhs, x, ldx, cakeep, cfkeep, coptions,   &
                                cinform ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), VALUE :: job
  INTEGER ( KIND = ipc_ ), VALUE :: nrhs
  REAL ( KIND = rpc_ ), DIMENSION( ldx, nrhs ) :: x
  INTEGER ( KIND = ipc_ ), VALUE :: ldx
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

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
    CALL f_ssids_solve( nrhs, x, ldx, fakeep, ffkeep, foptions, finform )
  ELSE
    CALL f_ssids_solve( nrhs, x, ldx, fakeep, ffkeep, foptions, finform,       &
                      job = job )
  END IF

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_solve

!  ------------------------------------------------
!  C interface to fortran ssids_free to free cakeep
!  ------------------------------------------------

  INTEGER ( KIND = ipc_ ) function spral_ssids_free_akeep( cakeep ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep

!  local variables

  TYPE ( f_ssids_akeep ), POINTER :: fakeep

!  nothing to free

  IF ( .NOT. C_ASSOCIATED( cakeep ) ) THEN
    spral_ssids_free_akeep = 0
    RETURN
  END IF

  CALL C_F_POINTER( cakeep, fakeep )
  CALL f_ssids_free( fakeep, spral_ssids_free_akeep )
  DEALLOCATE( fakeep )
  cakeep = C_NULL_PTR

  END FUNCTION spral_ssids_free_akeep

!  ------------------------------------------------
!  C interface to fortran ssids_free to free cfkeep
!  ------------------------------------------------

  INTEGER ( KIND = ipc_ ) function spral_ssids_free_fkeep( cfkeep ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep

!  local variables

  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep

!  nothing to free

  IF ( .NOT. C_ASSOCIATED( cfkeep ) ) THEN
    spral_ssids_free_fkeep = 0
    RETURN
  END IF

  CALL C_F_POINTER( cfkeep, ffkeep )
  CALL f_ssids_free( ffkeep, spral_ssids_free_fkeep )
  DEALLOCATE( ffkeep )
  cfkeep = C_NULL_PTR

  END FUNCTION spral_ssids_free_fkeep

!  ---------------------------------
!  C interface to fortran ssids_free
!  ---------------------------------

  INTEGER ( KIND = ipc_ ) FUNCTION spral_ssids_free( cakeep, cfkeep ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
  TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep

  INTERFACE
    INTEGER ( KIND = ipc_ ) FUNCTION spral_ssids_free_akeep( cakeep ) BIND( C )
      USE iso_c_binding
      USE SPRAL_KINDS_precision
      IMPLICIT NONE
      TYPE ( C_PTR ), INTENT( INOUT ) :: cakeep
    END FUNCTION spral_ssids_free_akeep
    INTEGER ( KIND = ipc_ ) FUNCTION spral_ssids_free_fkeep( cfkeep ) BIND( C )
      USE iso_c_binding
      USE SPRAL_KINDS_precision
      IMPLICIT NONE
      TYPE ( C_PTR ), INTENT( INOUT ) :: cfkeep
    END FUNCTION spral_ssids_free_fkeep
  END INTERFACE

  spral_ssids_free = spral_ssids_free_akeep( cakeep )
  IF ( spral_ssids_free /= 0_ipc_ ) RETURN
  spral_ssids_free = spral_ssids_free_fkeep( cfkeep )

  END FUNCTION spral_ssids_free

!  -------------------------------------------
!  C interface to fortran ssids_enquire_posdef
!  -------------------------------------------

  SUBROUTINE spral_ssids_enquire_posdef( cakeep, cfkeep, coptions, cinform,    &
                                         d ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform
  REAL ( KIND = rpc_ ), DIMENSION( * ), INTENT( OUT ) :: d

!  local variables

  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in(coptions, foptions, cindexed)

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

  CALL f_ssids_enquire_posdef( fakeep, ffkeep, foptions, finform, d )

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_enquire_posdef

!  ------------------------------------------
!  C interface to fortran ssids_enquire_indef
!  ------------------------------------------

  SUBROUTINE spral_ssids_enquire_indef( cakeep, cfkeep, coptions, cinform,     &
                                        cpiv_order, cd ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform
  TYPE ( C_PTR ), VALUE :: cpiv_order
  TYPE ( C_PTR ), VALUE :: cd

!  local variables

  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform
  INTEGER ( KIND = ipc_ ), DIMENSION( : ), POINTER :: fpiv_order
  REAL ( KIND = rpc_ ), DIMENSION( :,: ), POINTER :: fd

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

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
      CALL f_ssids_enquire_indef( fakeep, ffkeep, foptions, finform,           &
                                piv_order = fpiv_order, d = fd )
    ELSE
      CALL f_ssids_enquire_indef( fakeep, ffkeep, foptions, finform,           &
                                piv_order=fpiv_order )
    END IF
  ELSE
    IF ( ASSOCIATED( fd ) ) THEN
      CALL f_ssids_enquire_indef( fakeep, ffkeep, foptions, finform, d = fd )
    ELSE
      CALL f_ssids_enquire_indef( fakeep, ffkeep, foptions, finform )
    END IF
  END IF

!  copy arguments out

!  note: we use abs value of piv_order in C indexing, as 0 and -0 are the same

  IF ( ASSOCIATED( fpiv_order ) .AND. cindexed )                               &
    fpiv_order( : ) = ABS( fpiv_order( : ) ) - 1
  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_enquire_indef

!  ----------------------------------
!  C interface to fortran ssids_alter
!  ----------------------------------

  SUBROUTINE spral_ssids_alter( d, cakeep, cfkeep, coptions, cinform ) BIND( C )
  USE SPRAL_SSIDS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  REAL ( KIND = rpc_ ), DIMENSION( 2, * ), INTENT( IN ) :: d
  TYPE ( C_PTR ), VALUE :: cakeep
  TYPE ( C_PTR ), VALUE :: cfkeep
  TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
  TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_ssids_akeep ), POINTER :: fakeep
  TYPE ( f_ssids_fkeep ), POINTER :: ffkeep
  TYPE ( f_ssids_options ) :: foptions
  TYPE ( f_ssids_inform ) :: finform

  LOGICAL :: cindexed

!  copy options in first to find out whether we use fortran or C indexing

  CALL copy_options_in( coptions, foptions, cindexed )

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

  CALL f_ssids_alter( d, fakeep, ffkeep, foptions, finform )

!  copy arguments out

  CALL copy_inform_out( finform, cinform )

  END SUBROUTINE spral_ssids_alter
