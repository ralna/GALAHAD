! THIS VERSION: GALAHAD 3.3 - 22/08/2021 AT 15:28 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  N L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. August 22nd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_NLS_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_NLS_double, ONLY: &
        f_nls_subproblem_control_type => NLS_subproblem_control_type, &
        f_nls_control_type => NLS_control_type, &
        f_nls_time_type => NLS_time_type, &
        f_nls_subproblem_inform_type => NLS_subproblem_inform_type, &
        f_nls_inform_type => NLS_inform_type, &
        f_nls_full_data_type => NLS_full_data_type, &
        f_nls_initialize => NLS_initialize, &
        f_nls_read_specfile => NLS_read_specfile, &
        f_nls_import => NLS_import, &
        f_nls_information => NLS_information, &
        f_nls_terminate => NLS_terminate

!!$    USE GALAHAD_RQS_double_ciface, ONLY: &
!!$        rqs_inform_type, &
!!$        rqs_control_type, &
!!$        copy_rqs_inform_in => copy_inform_in, &
!!$        copy_rqs_inform_out => copy_inform_out, &
!!$        copy_rqs_control_in => copy_control_in, &
!!$        copy_rqs_control_out => copy_control_out
!!$
!!$    USE GALAHAD_GLRT_double_ciface, ONLY: &
!!$        glrt_inform_type, &
!!$        glrt_control_type, &
!!$        copy_glrt_inform_in => copy_inform_in, &
!!$        copy_glrt_inform_out => copy_inform_out, &
!!$        copy_glrt_control_in => copy_control_in, &
!!$        copy_glrt_control_out => copy_control_out
!!$
!!$    USE GALAHAD_PSLS_double_ciface, ONLY: &
!!$        psls_inform_type, &
!!$        psls_control_type, &
!!$        copy_psls_inform_in => copy_inform_in, &
!!$        copy_psls_inform_out => copy_inform_out, &
!!$        copy_psls_control_in => copy_control_in, &
!!$        copy_psls_control_out => copy_control_out
!!$
!!$    USE GALAHAD_BSC_double_ciface, ONLY: &
!!$        bsc_inform_type, &
!!$        bsc_control_type, &
!!$        copy_bsc_inform_in => copy_inform_in, &
!!$        copy_bsc_inform_out => copy_inform_out, &
!!$        copy_bsc_control_in => copy_control_in, &
!!$        copy_bsc_control_out => copy_control_out
!!$
!!$    USE GALAHAD_ROOTS_double_ciface, ONLY: &
!!$        roots_inform_type, &
!!$        roots_control_type, &
!!$        copy_roots_inform_in => copy_inform_in, &
!!$        copy_roots_inform_out => copy_inform_out, &
!!$        copy_roots_control_in => copy_control_in, &
!!$        copy_roots_control_out => copy_control_out

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: nls_subproblem_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: start_print
      INTEGER ( KIND = C_INT ) :: stop_print
      INTEGER ( KIND = C_INT ) :: print_gap
      INTEGER ( KIND = C_INT ) :: maxit
      INTEGER ( KIND = C_INT ) :: alive_unit
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: alive_file
      INTEGER ( KIND = C_INT ) :: jacobian_available
      INTEGER ( KIND = C_INT ) :: hessian_available
      INTEGER ( KIND = C_INT ) :: model
      INTEGER ( KIND = C_INT ) :: norm
      INTEGER ( KIND = C_INT ) :: non_monotone
      INTEGER ( KIND = C_INT ) :: weight_update_strategy
      REAL ( KIND = wp ) :: stop_c_absolute
      REAL ( KIND = wp ) :: stop_c_relative
      REAL ( KIND = wp ) :: stop_g_absolute
      REAL ( KIND = wp ) :: stop_g_relative
      REAL ( KIND = wp ) :: stop_s
      REAL ( KIND = wp ) :: power
      REAL ( KIND = wp ) :: initial_weight
      REAL ( KIND = wp ) :: minimum_weight
      REAL ( KIND = wp ) :: initial_inner_weight
      REAL ( KIND = wp ) :: eta_successful
      REAL ( KIND = wp ) :: eta_very_successful
      REAL ( KIND = wp ) :: eta_too_successful
      REAL ( KIND = wp ) :: weight_decrease_min
      REAL ( KIND = wp ) :: weight_decrease
      REAL ( KIND = wp ) :: weight_increase
      REAL ( KIND = wp ) :: weight_increase_max
      REAL ( KIND = wp ) :: reduce_gap
      REAL ( KIND = wp ) :: tiny_gap
      REAL ( KIND = wp ) :: large_root
      REAL ( KIND = wp ) :: switch_to_newton
      REAL ( KIND = wp ) :: cpu_time_limit
      REAL ( KIND = wp ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: subproblem_direct
      LOGICAL ( KIND = C_BOOL ) :: renormalize_weight
      LOGICAL ( KIND = C_BOOL ) :: magic_step
      LOGICAL ( KIND = C_BOOL ) :: print_obj
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
!!$      TYPE ( rqs_control_type ) :: rqs_control
!!$      TYPE ( glrt_control_type ) :: glrt_control
!!$      TYPE ( psls_control_type ) :: psls_control
!!$      TYPE ( bsc_control_type ) :: bsc_control
!!$      TYPE ( roots_control_type ) :: roots_control
    END TYPE nls_subproblem_control_type

    TYPE, EXTENDS( NLS_subproblem_control_type ), BIND( C ) :: nls_control_type
      TYPE ( nls_subproblem_control_type ) :: subproblem_control
    END TYPE nls_control_type

    TYPE, BIND( C ) :: nls_time_type
      REAL ( KIND = sp ) :: total
      REAL ( KIND = sp ) :: preprocess
      REAL ( KIND = sp ) :: analyse
      REAL ( KIND = sp ) :: factorize
      REAL ( KIND = sp ) :: solve
      REAL ( KIND = wp ) :: clock_total
      REAL ( KIND = wp ) :: clock_preprocess
      REAL ( KIND = wp ) :: clock_analyse
      REAL ( KIND = wp ) :: clock_factorize
      REAL ( KIND = wp ) :: clock_solve
    END TYPE nls_time_type

    TYPE, BIND( C ) :: nls_subproblem_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 13 ) :: bad_eval
      INTEGER ( KIND = C_INT ) :: iter
      INTEGER ( KIND = C_INT ) :: cg_iter
      INTEGER ( KIND = C_INT ) :: c_eval
      INTEGER ( KIND = C_INT ) :: j_eval
      INTEGER ( KIND = C_INT ) :: h_eval
      INTEGER ( KIND = C_INT ) :: factorization_max
      INTEGER ( KIND = C_INT ) :: factorization_status
      INTEGER ( KIND = C_INT ) :: max_entries_factors
      INTEGER ( KIND = C_INT ) :: factorization_integer
      INTEGER ( KIND = C_INT ) :: factorization_real
      REAL ( KIND = wp ) :: factorization_average
      REAL ( KIND = wp ) :: obj
      REAL ( KIND = wp ) :: norm_c
      REAL ( KIND = wp ) :: norm_g
      REAL ( KIND = wp ) :: weight
      TYPE ( nls_time_type ) :: time
!!$      TYPE ( rqs_inform_type ) :: rqs_inform
!!$      TYPE ( glrt_inform_type ) :: glrt_inform
!!$      TYPE ( psls_inform_type ) :: psls_inform
!!$      TYPE ( bsc_inform_type ) :: bsc_inform
!!$      TYPE ( roots_inform_type ) :: roots_inform
    END TYPE nls_subproblem_inform_type

    TYPE, EXTENDS( nls_subproblem_inform_type ), BIND( C ) :: nls_inform_type
      TYPE ( nls_subproblem_inform_type ) :: subproblem_inform
    END TYPE nls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_subproblem_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( nls_subproblem_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_nls_subproblem_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) )  f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%maxit = ccontrol%maxit
    fcontrol%alive_unit = ccontrol%alive_unit
    fcontrol%jacobian_available = ccontrol%jacobian_available
    fcontrol%hessian_available = ccontrol%hessian_available
    fcontrol%model = ccontrol%model
    fcontrol%norm = ccontrol%norm
    fcontrol%non_monotone = ccontrol%non_monotone
    fcontrol%weight_update_strategy = ccontrol%weight_update_strategy

    ! Reals
    fcontrol%stop_c_absolute = ccontrol%stop_c_absolute
    fcontrol%stop_c_relative = ccontrol%stop_c_relative
    fcontrol%stop_g_absolute = ccontrol%stop_g_absolute
    fcontrol%stop_g_relative = ccontrol%stop_g_relative
    fcontrol%stop_s = ccontrol%stop_s
    fcontrol%power = ccontrol%power
    fcontrol%initial_weight = ccontrol%initial_weight
    fcontrol%minimum_weight = ccontrol%minimum_weight
    fcontrol%initial_inner_weight = ccontrol%initial_inner_weight
    fcontrol%eta_successful = ccontrol%eta_successful
    fcontrol%eta_very_successful = ccontrol%eta_very_successful
    fcontrol%eta_too_successful = ccontrol%eta_too_successful
    fcontrol%weight_decrease_min = ccontrol%weight_decrease_min
    fcontrol%weight_decrease = ccontrol%weight_decrease
    fcontrol%weight_increase = ccontrol%weight_increase
    fcontrol%weight_increase_max = ccontrol%weight_increase_max
    fcontrol%reduce_gap = ccontrol%reduce_gap
    fcontrol%tiny_gap = ccontrol%tiny_gap
    fcontrol%large_root = ccontrol%large_root
    fcontrol%switch_to_newton = ccontrol%switch_to_newton
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%subproblem_direct = ccontrol%subproblem_direct
    fcontrol%renormalize_weight = ccontrol%renormalize_weight
    fcontrol%magic_step = ccontrol%magic_step
    fcontrol%print_obj = ccontrol%print_obj
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
!!$    CALL copy_rqs_control_in( ccontrol%rqs_control, fcontrol%rqs_control )
!!$    CALL copy_glrt_control_in( ccontrol%glrt_control, fcontrol%glrt_control )
!!$    CALL copy_psls_control_in( ccontrol%psls_control, fcontrol%psls_control )
!!$    CALL copy_bsc_control_in( ccontrol%bsc_control, fcontrol%bsc_control )
!!$    CALL copy_roots_control_in( ccontrol%roots_control, fcontrol%roots_control )

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%alive_file( i : i ) = ccontrol%alive_file( i )
    END DO
    DO i = 1, 31
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_subproblem_control_in

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( nls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_nls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    CALL copy_subproblem_control_in( ccontrol%nls_subproblem_control_type,     &
                                     fcontrol%f_nls_subproblem_control_type,   &
                                     f_indexing ) 
    CALL copy_subproblem_control_in( ccontrol%subproblem_control,              &
                                     fcontrol%subproblem_control,              &
                                     f_indexing ) 
    RETURN
    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_subproblem_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_nls_subproblem_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( nls_subproblem_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) )  ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%maxit = fcontrol%maxit
    ccontrol%alive_unit = fcontrol%alive_unit
    ccontrol%jacobian_available = fcontrol%jacobian_available
    ccontrol%hessian_available = fcontrol%hessian_available
    ccontrol%model = fcontrol%model
    ccontrol%norm = fcontrol%norm
    ccontrol%non_monotone = fcontrol%non_monotone
    ccontrol%weight_update_strategy = fcontrol%weight_update_strategy

    ! Reals
    ccontrol%stop_c_absolute = fcontrol%stop_c_absolute
    ccontrol%stop_c_relative = fcontrol%stop_c_relative
    ccontrol%stop_g_absolute = fcontrol%stop_g_absolute
    ccontrol%stop_g_relative = fcontrol%stop_g_relative
    ccontrol%stop_s = fcontrol%stop_s
    ccontrol%power = fcontrol%power
    ccontrol%initial_weight = fcontrol%initial_weight
    ccontrol%minimum_weight = fcontrol%minimum_weight
    ccontrol%initial_inner_weight = fcontrol%initial_inner_weight
    ccontrol%eta_successful = fcontrol%eta_successful
    ccontrol%eta_very_successful = fcontrol%eta_very_successful
    ccontrol%eta_too_successful = fcontrol%eta_too_successful
    ccontrol%weight_decrease_min = fcontrol%weight_decrease_min
    ccontrol%weight_decrease = fcontrol%weight_decrease
    ccontrol%weight_increase = fcontrol%weight_increase
    ccontrol%weight_increase_max = fcontrol%weight_increase_max
    ccontrol%reduce_gap = fcontrol%reduce_gap
    ccontrol%tiny_gap = fcontrol%tiny_gap
    ccontrol%large_root = fcontrol%large_root
    ccontrol%switch_to_newton = fcontrol%switch_to_newton
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%subproblem_direct = fcontrol%subproblem_direct
    ccontrol%renormalize_weight = fcontrol%renormalize_weight
    ccontrol%magic_step = fcontrol%magic_step
    ccontrol%print_obj = fcontrol%print_obj
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
!!$    CALL copy_rqs_control_out( fcontrol%rqs_control, ccontrol%rqs_control )
!!$    CALL copy_glrt_control_out( fcontrol%glrt_control, ccontrol%glrt_control )
!!$    CALL copy_psls_control_out( fcontrol%psls_control, ccontrol%psls_control )
!!$    CALL copy_bsc_control_out( fcontrol%bsc_control, ccontrol%bsc_control )
!!$    CALL copy_roots_control_out( fcontrol%roots_control, ccontrol%roots_control )

    ! Strings
    DO i = 1, LEN( fcontrol%alive_file )
      ccontrol%alive_file( i ) = fcontrol%alive_file( i : i )
    END DO
    ccontrol%alive_file( LEN( fcontrol%alive_file ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_subproblem_control_out

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_nls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( nls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    CALL copy_subproblem_control_out( fcontrol%f_nls_subproblem_control_type,  &
                                      ccontrol%nls_subproblem_control_type,    &
                                      f_indexing ) 
    CALL copy_subproblem_control_out( fcontrol%subproblem_control,             &
                                      ccontrol%subproblem_control, f_indexing ) 
    RETURN
    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime ) 
    TYPE ( nls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_nls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime ) 
    TYPE ( f_nls_time_type ), INTENT( IN ) :: ftime
    TYPE ( nls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_subproblem_inform_in( cinform, finform ) 
    TYPE ( nls_subproblem_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_nls_subproblem_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%cg_iter = cinform%cg_iter
    finform%c_eval = cinform%c_eval
    finform%j_eval = cinform%j_eval
    finform%h_eval = cinform%h_eval
    finform%factorization_max = cinform%factorization_max
    finform%factorization_status = cinform%factorization_status
    finform%max_entries_factors = cinform%max_entries_factors
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real

    ! Reals
    finform%factorization_average = cinform%factorization_average
    finform%obj = cinform%obj
    finform%norm_c = cinform%norm_c
    finform%norm_g = cinform%norm_g
    finform%weight = cinform%weight

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
!!$    CALL copy_rqs_inform_in( cinform%rqs_inform, finform%rqs_inform )
!!$    CALL copy_glrt_inform_in( cinform%glrt_inform, finform%glrt_inform )
!!$    CALL copy_psls_inform_in( cinform%psls_inform, finform%psls_inform )
!!$    CALL copy_bsc_inform_in( cinform%bsc_inform, finform%bsc_inform )
!!$    CALL copy_roots_inform_in( cinform%roots_inform, finform%roots_inform )

    ! Strings
    DO i = 1, 81
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    DO i = 1, 13
      IF ( cinform%bad_eval( i ) == C_NULL_CHAR ) EXIT
      finform%bad_eval( i : i ) = cinform%bad_eval( i )
    END DO
    RETURN

    END SUBROUTINE copy_subproblem_inform_in

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( nls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_nls_inform_type ), INTENT( OUT ) :: finform
    CALL copy_subproblem_inform_in( cinform%nls_subproblem_inform_type,        &
                                    finform%f_nls_subproblem_inform_type ) 
    CALL copy_subproblem_inform_in( cinform%subproblem_inform,                 &
                                    finform%subproblem_inform ) 
    RETURN
    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_subproblem_inform_out( finform, cinform ) 
    TYPE ( f_nls_subproblem_inform_type ), INTENT( IN ) :: finform
    TYPE ( nls_subproblem_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%cg_iter = finform%cg_iter
    cinform%c_eval = finform%c_eval
    cinform%j_eval = finform%j_eval
    cinform%h_eval = finform%h_eval
    cinform%factorization_max = finform%factorization_max
    cinform%factorization_status = finform%factorization_status
    cinform%max_entries_factors = finform%max_entries_factors
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real

    ! Reals
    cinform%factorization_average = finform%factorization_average
    cinform%obj = finform%obj
    cinform%norm_c = finform%norm_c
    cinform%norm_g = finform%norm_g
    cinform%weight = finform%weight

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
!!$    CALL copy_rqs_inform_out( finform%rqs_inform, cinform%rqs_inform )
!!$    CALL copy_glrt_inform_out( finform%glrt_inform, cinform%glrt_inform )
!!$    CALL copy_psls_inform_out( finform%psls_inform, cinform%psls_inform )
!!$    CALL copy_bsc_inform_out( finform%bsc_inform, cinform%bsc_inform )
!!$    CALL copy_roots_inform_out( finform%roots_inform, cinform%roots_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( LEN( finform%bad_alloc ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( finform%bad_eval )
      cinform%bad_eval( i ) = finform%bad_eval( i : i )
    END DO
    cinform%bad_eval( LEN( finform%bad_eval ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_subproblem_inform_out

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_nls_inform_type ), INTENT( IN ) :: finform
    TYPE ( nls_inform_type ), INTENT( OUT ) :: cinform
    CALL copy_subproblem_inform_in( finform%f_nls_subproblem_inform_type,      &
                                    cinform%nls_subproblem_inform_type )
    CALL copy_subproblem_inform_in( finform%subproblem_inform,                 &
                                    cinform%subproblem_inform )
    RETURN
    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_NLS_double_ciface

!  -------------------------------------
!  C interface to fortran nls_initialize
!  -------------------------------------

  SUBROUTINE nls_initialize( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( nls_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( nls_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_nls_full_data_type ), POINTER :: fdata
  TYPE ( f_nls_control_type ) :: fcontrol
  TYPE ( f_nls_inform_type ) :: finform
  LOGICAL :: f_indexing 

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_nls_initialize( fdata, fcontrol, finform )

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE nls_initialize

!  ----------------------------------------
!  C interface to fortran nls_read_specfile
!  ----------------------------------------

  SUBROUTINE nls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( nls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_nls_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = C_INT ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )
  
!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )
  
!  read control parameters from the specfile

  CALL f_nls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE nls_read_specfile

!  ---------------------------------
!  C interface to fortran nls_inport
!  ---------------------------------

  SUBROUTINE nls_import( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( nls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_nls_control_type ) :: fcontrol
  TYPE ( f_nls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN

!  import the problem data into the required NLS structure

    CALL f_nls_import( fcontrol, fdata, status )
  ELSE
    CALL f_nls_import( fcontrol, fdata, status )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE nls_import

!  --------------------------------------
!  C interface to fortran nls_information
!  --------------------------------------

  SUBROUTINE nls_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( nls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_nls_full_data_type ), pointer :: fdata
  TYPE ( f_nls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain NLS solution information

  CALL f_nls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE nls_information

!  ------------------------------------
!  C interface to fortran nls_terminate
!  ------------------------------------

  SUBROUTINE nls_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( nls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( nls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_nls_full_data_type ), pointer :: fdata
  TYPE ( f_nls_control_type ) :: fcontrol
  TYPE ( f_nls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_nls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE nls_terminate
