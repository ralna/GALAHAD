! THIS VERSION: GALAHAD 4.0 - 2022-01-28 AT 17:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  S C U    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SCU_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_SCU_double, ONLY:                                              &
        f_scu_control_type   => SCU_control_type,                              &
        f_scu_inform_type    => SCU_inform_type,                               &
        f_scu_full_data_type => SCU_full_data_type,                            &
!       f_scu_initialize     => SCU_initialize,                                &
!       f_scu_read_specfile  => SCU_read_specfile,                             &
!       f_scu_import         => SCU_import,                                    &
!       f_scu_reset_control  => SCU_reset_control,                             &
!       f_scu_information    => SCU_information,                               &
        f_scu_terminate => SCU_terminate

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

     TYPE, BIND( C ) :: scu_control_type
      INTEGER ( KIND = C_INT ) :: dummy
!  no components at present
     END TYPE scu_control_type

    TYPE, BIND( C ) :: scu_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      INTEGER ( KIND = C_INT ), DIMENSION( 3 ) :: inertia
    END TYPE scu_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

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
    cinform%alloc_status = finform%alloc_status
    cinform%inertia( 1 : 3 ) = finform%inertia( 1 : 3 )
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SCU_double_ciface

!  ------------------------------------
!  C interface to fortran scu_terminate
!  ------------------------------------

  SUBROUTINE scu_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SCU_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( scu_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( scu_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_scu_full_data_type ), pointer :: fdata
  INTEGER ( KIND = C_INT ) :: status
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
