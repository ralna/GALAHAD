! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  G L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. November 30th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_GLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_GLS_precision, ONLY:                                           &
        f_gls_control => GLS_control,                                          &
        f_gls_ainfo => GLS_ainfo,                                              &
        f_gls_finfo => GLS_finfo,                                              &
        f_gls_sinfo => GLS_sinfo,                                              &
        f_gls_full_data_type => GLS_full_data_type,                            &
        f_gls_initialize => GLS_initialize,                                    &
        f_gls_reset_control => GLS_reset_control,                              &
        f_gls_information => GLS_information,                                  &
        f_gls_finalize => GLS_finalize

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: gls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: lp
      INTEGER ( KIND = ipc_ ) :: wp
      INTEGER ( KIND = ipc_ ) :: mp
      INTEGER ( KIND = ipc_ ) :: ldiag
      INTEGER ( KIND = ipc_ ) :: btf
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: factor_blocking
      INTEGER ( KIND = ipc_ ) :: solve_blas
      INTEGER ( KIND = ipc_ ) :: la
      INTEGER ( KIND = ipc_ ) :: la_int
      INTEGER ( KIND = ipc_ ) :: maxla
      INTEGER ( KIND = ipc_ ) :: pivoting
      INTEGER ( KIND = ipc_ ) :: fill_in
      REAL ( KIND = rpc_ ) :: multiplier
      REAL ( KIND = rpc_ ) :: reduce
      REAL ( KIND = rpc_ ) :: u
      REAL ( KIND = rpc_ ) :: switch_full
      REAL ( KIND = rpc_ ) :: drop
      REAL ( KIND = rpc_ ) :: tolerance
      REAL ( KIND = rpc_ ) :: cgce
      LOGICAL ( KIND = C_BOOL ) :: diagonal_pivoting
      LOGICAL ( KIND = C_BOOL ) :: struct_abort
    END TYPE gls_control_type

    TYPE, BIND( C ) :: GLS_ainfo_type
      INTEGER ( KIND = ipc_ ) :: flag
      INTEGER ( KIND = ipc_ ) :: more
      INTEGER ( KIND = ipc_ ) :: len_analyse
      INTEGER ( KIND = ipc_ ) :: len_factorize
      INTEGER ( KIND = ipc_ ) :: ncmpa
      INTEGER ( KIND = ipc_ ) :: rank
      INTEGER ( KIND = ipc_ ) :: drop
      INTEGER ( KIND = ipc_ ) :: struc_rank
      INTEGER ( KIND = ipc_ ) :: oor
      INTEGER ( KIND = ipc_ ) :: dup
      INTEGER ( KIND = ipc_ ) :: stat
      INTEGER ( KIND = ipc_ ) :: lblock
      INTEGER ( KIND = ipc_ ) :: sblock
      INTEGER ( KIND = ipc_ ) :: tblock
      REAL ( KIND = rpc_ ) :: ops
    END TYPE GLS_ainfo_type

    TYPE, BIND( C ) :: GLS_finfo_type
      INTEGER ( KIND = ipc_ ) :: flag
      INTEGER ( KIND = ipc_ ) :: more
      INTEGER ( KIND = ipc_ ) :: size_factor
      INTEGER ( KIND = ipc_ ) :: len_factorize
      INTEGER ( KIND = ipc_ ) :: drop
      INTEGER ( KIND = ipc_ ) :: rank
      INTEGER ( KIND = ipc_ ) :: stat
      REAL ( KIND = rpc_ ) :: ops
    END TYPE GLS_finfo_type

    TYPE, BIND( C ) :: GLS_sinfo_type
      INTEGER ( KIND = ipc_ ) :: flag
      INTEGER ( KIND = ipc_ ) :: more
      INTEGER ( KIND = ipc_ ) :: stat
    END TYPE GLS_sinfo_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( gls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_gls_control ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%lp = ccontrol%lp
    fcontrol%wp = ccontrol%wp
    fcontrol%mp = ccontrol%mp
    fcontrol%ldiag = ccontrol%ldiag
    fcontrol%btf = ccontrol%btf
    fcontrol%maxit = ccontrol%maxit
    fcontrol%factor_blocking = ccontrol%factor_blocking
    fcontrol%solve_blas = ccontrol%solve_blas
    fcontrol%la = ccontrol%la
    fcontrol%la_int = ccontrol%la_int
    fcontrol%maxla = ccontrol%maxla
    fcontrol%pivoting = ccontrol%pivoting
    fcontrol%fill_in = ccontrol%fill_in

    ! Reals
    fcontrol%multiplier = ccontrol%multiplier
    fcontrol%reduce = ccontrol%reduce
    fcontrol%u = ccontrol%u
    fcontrol%switch = ccontrol%switch_full
    fcontrol%drop = ccontrol%drop
    fcontrol%tolerance = ccontrol%tolerance
    fcontrol%cgce = ccontrol%cgce

    ! Logicals
    fcontrol%diagonal_pivoting = ccontrol%diagonal_pivoting
    fcontrol%struct = ccontrol%struct_abort
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_gls_control ), INTENT( IN ) :: fcontrol
    TYPE ( gls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%lp = fcontrol%lp
    ccontrol%wp = fcontrol%wp
    ccontrol%mp = fcontrol%mp
    ccontrol%ldiag = fcontrol%ldiag
    ccontrol%btf = fcontrol%btf
    ccontrol%maxit = fcontrol%maxit
    ccontrol%factor_blocking = fcontrol%factor_blocking
    ccontrol%solve_blas = fcontrol%solve_blas
    ccontrol%la = fcontrol%la
    ccontrol%la_int = fcontrol%la_int
    ccontrol%maxla = fcontrol%maxla
    ccontrol%pivoting = fcontrol%pivoting
    ccontrol%fill_in = fcontrol%fill_in

    ! Reals
    ccontrol%multiplier = fcontrol%multiplier
    ccontrol%reduce = fcontrol%reduce
    ccontrol%u = fcontrol%u
    ccontrol%switch_full = fcontrol%switch
    ccontrol%drop = fcontrol%drop
    ccontrol%tolerance = fcontrol%tolerance
    ccontrol%cgce = fcontrol%cgce

    ! Logicals
    ccontrol%diagonal_pivoting = fcontrol%diagonal_pivoting
    ccontrol%struct_abort = fcontrol%struct
    RETURN

    END SUBROUTINE copy_control_out

!  copy C ainfo parameters to fortran

    SUBROUTINE copy_ainfo_in( cainfo, fainfo )
    TYPE ( gls_ainfo_type ), INTENT( IN ) :: cainfo
    TYPE ( f_gls_ainfo ), INTENT( OUT ) :: fainfo

    ! Integers
    fainfo%flag = cainfo%flag
    fainfo%more = cainfo%more
    fainfo%len_analyse = cainfo%len_analyse
    fainfo%len_factorize = cainfo%len_factorize
    fainfo%ncmpa = cainfo%ncmpa
    fainfo%rank = cainfo%rank
    fainfo%drop = cainfo%drop
    fainfo%struc_rank = cainfo%struc_rank
    fainfo%oor = cainfo%oor
    fainfo%dup = cainfo%dup
    fainfo%stat = cainfo%stat
    fainfo%lblock = cainfo%lblock
    fainfo%sblock = cainfo%sblock
    fainfo%tblock = cainfo%tblock

    ! Reals
    fainfo%ops = cainfo%ops

    RETURN

    END SUBROUTINE copy_ainfo_in

!  copy fortran ainfo parameters to C

    SUBROUTINE copy_ainfo_out( fainfo, cainfo )
    TYPE ( f_gls_ainfo ), INTENT( IN ) :: fainfo
    TYPE ( gls_ainfo_type ), INTENT( OUT ) :: cainfo

    ! Integers
    cainfo%flag = fainfo%flag
    cainfo%more = fainfo%more
    cainfo%len_analyse = fainfo%len_analyse
    cainfo%len_factorize = fainfo%len_factorize
    cainfo%ncmpa = fainfo%ncmpa
    cainfo%rank = fainfo%rank
    cainfo%drop = fainfo%drop
    cainfo%struc_rank = fainfo%struc_rank
    cainfo%oor = fainfo%oor
    cainfo%dup = fainfo%dup
    cainfo%stat = fainfo%stat
    cainfo%lblock = fainfo%lblock
    cainfo%sblock = fainfo%sblock
    cainfo%tblock = fainfo%tblock

    ! Reals
    cainfo%ops = fainfo%ops

    RETURN

    END SUBROUTINE copy_ainfo_out

!  copy C finfo parameters to fortran

    SUBROUTINE copy_finfo_in( cfinfo, ffinfo )
    TYPE ( gls_finfo_type ), INTENT( IN ) :: cfinfo
    TYPE ( f_gls_finfo ), INTENT( OUT ) :: ffinfo

    ! Integers
    ffinfo%flag = cfinfo%flag
    ffinfo%more = cfinfo%more
    ffinfo%size_factor = cfinfo%size_factor
    ffinfo%len_factorize = cfinfo%len_factorize
    ffinfo%drop = cfinfo%drop
    ffinfo%rank = cfinfo%rank
    ffinfo%stat = cfinfo%stat

    ! Reals
    ffinfo%ops = cfinfo%ops

    RETURN

    END SUBROUTINE copy_finfo_in

!  copy fortran finfo parameters to C

    SUBROUTINE copy_finfo_out( ffinfo, cfinfo )
    TYPE ( f_gls_finfo ), INTENT( IN ) :: ffinfo
    TYPE ( gls_finfo_type ), INTENT( OUT ) :: cfinfo

    ! Integers
    cfinfo%flag = ffinfo%flag
    cfinfo%more = ffinfo%more
    cfinfo%size_factor = ffinfo%size_factor
    cfinfo%len_factorize = ffinfo%len_factorize
    cfinfo%drop = ffinfo%drop
    cfinfo%rank = ffinfo%rank
    cfinfo%stat = ffinfo%stat

    ! Reals
    cfinfo%ops = ffinfo%ops

    RETURN

    END SUBROUTINE copy_finfo_out

!  copy C sinfo parameters to fortran

    SUBROUTINE copy_sinfo_in( csinfo, fsinfo )
    TYPE ( gls_sinfo_type ), INTENT( IN ) :: csinfo
    TYPE ( f_gls_sinfo ), INTENT( OUT ) :: fsinfo

    ! Integers
    fsinfo%flag = csinfo%flag
    fsinfo%more = csinfo%more
    fsinfo%stat = csinfo%stat

    RETURN

    END SUBROUTINE copy_sinfo_in

!  copy fortran sinfo parameters to C

    SUBROUTINE copy_sinfo_out( fsinfo, csinfo )
    TYPE ( f_gls_sinfo ), INTENT( IN ) :: fsinfo
    TYPE ( gls_sinfo_type ), INTENT( OUT ) :: csinfo

    ! Integers
    csinfo%flag = fsinfo%flag
    csinfo%more = fsinfo%more
    csinfo%stat = fsinfo%stat

    RETURN

    END SUBROUTINE copy_sinfo_out

  END MODULE GALAHAD_GLS_precision_ciface

!  -------------------------------------
!  C interface to fortran gls_initialize
!  -------------------------------------

  SUBROUTINE gls_initialize( cdata, ccontrol ) BIND( C )
  USE GALAHAD_GLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( gls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_gls_full_data_type ), POINTER :: fdata
  TYPE ( f_gls_control ) :: fcontrol
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_gls_initialize( fdata, fcontrol )

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE gls_initialize

!  ---------------------------------------
!  C interface to fortran gls_reset_control
!  ----------------------------------------

  SUBROUTINE gls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_GLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( gls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_gls_control ) :: fcontrol
  TYPE ( f_gls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_GLS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE gls_reset_control

!  --------------------------------------
!  C interface to fortran gls_information
!  --------------------------------------

  SUBROUTINE gls_information( cdata, cainfo, cfinfo, csinfo, status ) BIND( C )
  USE GALAHAD_GLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( gls_ainfo_type ), INTENT( INOUT ) :: cainfo
  TYPE ( gls_finfo_type ), INTENT( INOUT ) :: cfinfo
  TYPE ( gls_sinfo_type ), INTENT( INOUT ) :: csinfo
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_gls_full_data_type ), pointer :: fdata
  TYPE ( f_gls_ainfo ) :: fainfo
  TYPE ( f_gls_finfo ) :: ffinfo
  TYPE ( f_gls_sinfo ) :: fsinfo

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain GLS solution information

  CALL f_gls_information( fdata, fainfo, ffinfo, fsinfo, status )

!  copy infos out

  CALL copy_ainfo_out( fainfo, cainfo )
  CALL copy_finfo_out( ffinfo, cfinfo )
  CALL copy_sinfo_out( fsinfo, csinfo )
  RETURN

  END SUBROUTINE gls_information

!  ------------------------------------
!  C interface to fortran gls_finalize
!  ------------------------------------

  SUBROUTINE gls_finalize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_GLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( gls_control_type ), INTENT( IN ) :: ccontrol
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_gls_full_data_type ), pointer :: fdata
  TYPE ( f_gls_control ) :: fcontrol
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_gls_finalize( fdata, fcontrol, status )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE gls_finalize
