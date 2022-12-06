! THIS VERSION: GALAHAD 4.0 - 2022-02-25 AT 07:13 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  C O N V E R T    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. February 25th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_CONVERT_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_CONVERT_double, ONLY:                                          &
        f_convert_control_type   => CONVERT_control_type,                      &
        f_convert_time_type      => CONVERT_time_type,                         &
        f_convert_inform_type    => CONVERT_inform_type,                       &
        f_convert_full_data_type => CONVERT_full_data_type
!       f_convert_initialize     => CONVERT_initialize,                        &
!       f_convert_read_specfile  => CONVERT_read_specfile,                     &
!       f_convert_import         => CONVERT_import,                            &
!       f_convert_reset_control  => CONVERT_reset_control,                     &
!       f_convert_information    => CONVERT_information,                       &
!       f_convert_terminate => CONVERT_terminate

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: convert_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: transpose
      LOGICAL ( KIND = C_BOOL ) :: sum_duplicates
      LOGICAL ( KIND = C_BOOL ) :: order
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE convert_control_type

    TYPE, BIND( C ) :: convert_time_type
      REAL ( KIND = wp ) :: total
      REAL ( KIND = wp ) :: clock_total
    END TYPE convert_time_type

    TYPE, BIND( C ) :: convert_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      INTEGER ( KIND = C_INT ) :: duplicates
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( convert_time_type ) :: time
    END TYPE convert_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( convert_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_convert_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level

    ! Logicals
    fcontrol%transpose = ccontrol%transpose
    fcontrol%sum_duplicates = ccontrol%sum_duplicates
    fcontrol%order = ccontrol%order
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
    TYPE ( f_convert_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( convert_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level

    ! Logicals
    ccontrol%transpose = fcontrol%transpose
    ccontrol%sum_duplicates = fcontrol%sum_duplicates
    ccontrol%order = fcontrol%order
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

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( convert_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_convert_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%clock_total = ctime%clock_total
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_convert_time_type ), INTENT( IN ) :: ftime
    TYPE ( convert_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%clock_total = ftime%clock_total
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( convert_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_convert_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%duplicates = cinform%duplicates

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_convert_inform_type ), INTENT( IN ) :: finform
    TYPE ( convert_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%duplicates = finform%duplicates

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_CONVERT_double_ciface

