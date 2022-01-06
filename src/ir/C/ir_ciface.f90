! THIS VERSION: GALAHAD 4.0 - 2022-01-06 AT 11:45 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  I R    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.4. January 4th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_IR_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_IR_double, ONLY:                                               &
        f_ir_control_type => IR_control_type,                                  &
        f_ir_inform_type => IR_inform_type

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: ir_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: itref_max
      REAL ( KIND = wp ) :: acceptable_residual_relative
      REAL ( KIND = wp ) :: acceptable_residual_absolute
      REAL ( KIND = wp ) :: required_residual_relative
      LOGICAL ( KIND = C_BOOL ) :: record_residuals
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE ir_control_type

    TYPE, BIND( C ) :: ir_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      REAL ( KIND = wp ) :: norm_initial_residual
      REAL ( KIND = wp ) :: norm_final_residual
    END TYPE ir_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( ir_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_ir_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%itref_max = ccontrol%itref_max

    ! Reals
    fcontrol%acceptable_residual_relative                                      &
      = ccontrol%acceptable_residual_relative
    fcontrol%acceptable_residual_absolute                                      &
      = ccontrol%acceptable_residual_absolute
    fcontrol%required_residual_relative                                        &
      = ccontrol%required_residual_relative

    ! Logicals
    fcontrol%record_residuals = ccontrol%record_residuals
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Strings
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_ir_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( ir_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i, l
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%itref_max = fcontrol%itref_max

    ! Reals
    ccontrol%acceptable_residual_relative                                      &
      = fcontrol%acceptable_residual_relative
    ccontrol%acceptable_residual_absolute                                      &
      = fcontrol%acceptable_residual_absolute
    ccontrol%required_residual_relative                                        &
      = fcontrol%required_residual_relative

    ! Logicals
    ccontrol%record_residuals = fcontrol%record_residuals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Strings
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( ir_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_ir_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status

    ! Reals
    finform%norm_initial_residual = cinform%norm_initial_residual
    finform%norm_final_residual = cinform%norm_final_residual

    ! Strings
     DO i = 1, LEN( finform%bad_alloc )
       IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
       finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
     END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_ir_inform_type ), INTENT( IN ) :: finform
    TYPE ( ir_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status

    ! Reals
    cinform%norm_initial_residual = finform%norm_initial_residual
    cinform%norm_final_residual = finform%norm_final_residual

    ! Strings
!   l = LEN( finform%bad_alloc )
!   DO i = 1, l
!     cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
!   END DO
!   cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_IR_double_ciface
