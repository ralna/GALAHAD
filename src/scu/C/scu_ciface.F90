! THIS VERSION: GALAHAD 4.1 - 2023-05-05 AT 09:10 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  S C U    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SCU_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SCU_precision, ONLY:                                           &
        f_scu_control_type   => SCU_control_type,                              &
        f_scu_inform_type    => SCU_inform_type,                               &
        f_scu_full_data_type => SCU_full_data_type,                            &
        f_scu_initialize     => SCU_initialize,                                &
!       f_scu_read_specfile  => SCU_read_specfile,                             &
!       f_scu_import         => SCU_import,                                    &
!       f_scu_reset_control  => SCU_reset_control,                             &
        f_scu_information    => SCU_information,                               &
        f_scu_terminate => SCU_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

     TYPE, BIND( C ) :: scu_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
!  no components at present
     END TYPE scu_control_type

    TYPE, BIND( C ) :: scu_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ), DIMENSION( 3 ) :: inertia
    END TYPE scu_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( scu_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_scu_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_scu_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( scu_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( scu_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_scu_inform_type ), INTENT( OUT ) :: finform

    ! Integers
    finform%alloc_status = cinform%alloc_status
    finform%inertia( 1 : 3 ) = cinform%inertia( 1 : 3 )
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_scu_inform_type ), INTENT( IN ) :: finform
    TYPE ( scu_inform_type ), INTENT( OUT ) :: cinform

    ! Integers
    cinform%status = 0
    cinform%alloc_status = finform%alloc_status
    cinform%inertia( 1 : 3 ) = finform%inertia( 1 : 3 )
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SCU_precision_ciface

!  -------------------------------------
!  C interface to fortran scu_initialize
!  -------------------------------------

  SUBROUTINE scu_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_SCU_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( scu_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_scu_full_data_type ), POINTER :: fdata
  TYPE ( f_scu_control_type ) :: fcontrol
  TYPE ( f_scu_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_scu_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE scu_initialize

!  --------------------------------------
!  C interface to fortran scu_information
!  --------------------------------------

  SUBROUTINE scu_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SCU_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( scu_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_scu_full_data_type ), POINTER :: fdata
  TYPE ( f_scu_inform_type ) :: finform

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  obtain SCU solution information

  CALL f_scu_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE scu_information

!  ------------------------------------
!  C interface to fortran scu_terminate
!  ------------------------------------

  SUBROUTINE scu_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SCU_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( scu_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( scu_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_scu_full_data_type ), pointer :: fdata
  INTEGER ( KIND = ipc_ ) :: status
  TYPE ( f_scu_inform_type ) :: finform
! LOGICAL :: f_indexing

!  copy control in

! CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_scu_terminate( fdata, status, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  cinform%status = status

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE scu_terminate
