! THIS VERSION: GALAHAD 3.3 - 02/08/2021 AT 10:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ L H S   C   I N T E R F A C E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Jaroslav Fowkes

!  History -
!   originally released GALAHAD Version 3.3. August 2nd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module to GALAHAD_LHS types and interfaces

  MODULE GALAHAD_LHS_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_LHS_double, only:                                              &
        f_lhs_inform_type   => LHS_inform_type,                                &
        f_lhs_control_type  => LHS_control_type,                               &
        f_lhs_data_type     => LHS_data_type,                                  &
        f_lhs_initialize    => LHS_initialize,                                 &
        f_lhs_read_specfile => LHS_read_specfile,                              &
        f_lhs_ihs           => LHS_ihs,                                        &
        f_lhs_get_seed      => LHS_get_seed,                                   &
        f_lhs_terminate     => LHS_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: lhs_control_type
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: duplication
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), dimension( 31 ) :: prefix 
    END TYPE lhs_control_type

    TYPE, BIND( C ) :: lhs_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), dimension( 81 ) :: bad_alloc
    END TYPE lhs_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy fortran control parameters to C

    SUBROUTINE copy_control_in( ccontrol, fcontrol ) 
    TYPE ( lhs_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_lhs_control_type ), INTENT( OUT ) :: fcontrol
    INTEGER :: i
    
    ! integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%duplication = ccontrol%duplication

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
    TYPE ( f_lhs_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( lhs_control_type ), INTENT( OUT ) :: ccontrol
    INTEGER :: i
    
    ! integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%duplication = fcontrol%duplication

    ! logicals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! strings
    DO i = 1,  LEN(  fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C information to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( lhs_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_lhs_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status

    ! strings
    DO i = 1, 81
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran information to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_lhs_inform_type ), INTENT( IN ) :: finform
    TYPE ( lhs_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i

    ! integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status

    ! strings
    DO i = 1, LEN( finform%bad_alloc)
        cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( LEN( finform%bad_alloc ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END module GALAHAD_LHS_double_ciface

!  -------------------------------------
!  C interface to fortran lhs_initialize
!  -------------------------------------

  SUBROUTINE lhs_initialize( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_LHS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( lhs_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( lhs_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_lhs_data_type ), POINTER :: fdata
  TYPE ( f_lhs_control_type ) :: fcontrol
  TYPE ( f_lhs_inform_type ) :: finform

!  allocate fdata 

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_lhs_initialize( fdata, fcontrol, finform ) 

! copy control out

  CALL copy_control_out( fcontrol, ccontrol )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE lhs_initialize

!  ----------------------------------------
!  C interface to fortran lhs_read_specfile
!  ----------------------------------------

  SUBROUTINE lhs_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_LHS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( lhs_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), value :: cspecfile

!  local variables

  TYPE ( f_lhs_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile

!  device unit number for specfile

  INTEGER ( KIND = C_INT ), parameter :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol )
  
!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_lhs_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol )
  RETURN

  END SUBROUTINE lhs_read_specfile

!  ------------------------------
!  C interface to fortran lhs_ihs
!  ------------------------------

  SUBROUTINE lhs_ihs( n_dimen, n_points, seed, X, ccontrol, cinform,           &
                      cdata ) BIND( C )
  USE GALAHAD_LHS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), VALUE, INTENT( IN ) :: n_dimen, n_points
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: seed
  INTEGER ( KIND = C_INT ), DIMENSION( n_dimen, n_points ) :: X
  TYPE ( lhs_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( lhs_inform_type ), INTENT( INOUT ) :: cinform
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_lhs_control_type ) :: fcontrol
  TYPE ( f_lhs_inform_type ) :: finform
  TYPE ( f_lhs_data_type ), POINTER :: fdata

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol )

!  associate data pointers

  CALL C_F_POINTER(cdata, fdata)

!  call the improved distributed hyper-cube sampling (ihs) algorithm

  CALL f_lhs_ihs(n_dimen, n_points, seed, X, fcontrol, finform, fdata )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE lhs_ihs

!  -----------------------------------
!  C interface to fortran lhs_get_seed
!  -----------------------------------

  SUBROUTINE lhs_get_seed( seed ) BIND( C )
  USE GALAHAD_LHS_double_ciface
  IMPLICIT NONE

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: seed

!  get the random-number seed

  CALL f_lhs_get_seed( seed )
  RETURN

  END SUBROUTINE lhs_get_seed

!  ------------------------------------
!  C interface to fortran lhs_terminate
!  ------------------------------------

  SUBROUTINE lhs_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_LHS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( lhs_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( lhs_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_lhs_control_type ) :: fcontrol
  TYPE ( f_lhs_inform_type ) :: finform
  TYPE ( f_lhs_data_type ), POINTER :: fdata

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol )

!  associate data pointers

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_lhs_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate fdata

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE lhs_terminate
