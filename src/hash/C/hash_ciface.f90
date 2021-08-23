! THIS VERSION: GALAHAD 3.3 - 11/08/2021 AT 15:39 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  H A S H    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. August 11th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_HASH_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_HASH, ONLY:                                                    &
        f_hash_control_type => HASH_control_type,                              &
        f_hash_inform_type => HASH_inform_type,                                &
        f_hash_full_data_type => HASH_full_data_type,                          &
        f_hash_initialize => HASH_initialize,                                  &
        f_hash_terminate => HASH_terminate

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: HASH_control_type
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE HASH_control_type

    TYPE, BIND( C ) :: HASH_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
    END TYPE HASH_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol ) 
    TYPE ( hash_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_hash_control_type ), INTENT( OUT ) :: fcontrol
    INTEGER :: i
    
    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level

    ! Logicals
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol ) 
    TYPE ( f_hash_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( hash_control_type ), INTENT( OUT ) :: ccontrol
    INTEGER :: i
    
    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level

    ! Logicals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Strings
    DO i = 1, LEN( fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( hash_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_hash_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status

    ! Strings
    DO i = 1, 81
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_hash_inform_type ), INTENT( IN ) :: finform
    TYPE ( hash_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( LEN( finform%bad_alloc ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_HASH_ciface

!  -------------------------------------
!  C interface to fortran hash_initialize
!  -------------------------------------

  SUBROUTINE hash_initialize( nchar, length, cdata, ccontrol,                  &
                              cinform ) BIND( C ) 
  USE GALAHAD_HASH_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ) :: nchar, length
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( hash_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( hash_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_hash_full_data_type ), POINTER :: fdata
  TYPE ( f_hash_control_type ) :: fcontrol
  TYPE ( f_hash_inform_type ) :: finform

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_hash_initialize( nchar, length, fdata, fcontrol, finform )

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE hash_initialize

!  ------------------------------------
!  C interface to fortran hash_terminate
!  ------------------------------------

  SUBROUTINE hash_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_HASH_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( hash_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( hash_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_hash_full_data_type ), pointer :: fdata
  TYPE ( f_hash_control_type ) :: fcontrol
  TYPE ( f_hash_inform_type ) :: finform

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_hash_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE hash_terminate
