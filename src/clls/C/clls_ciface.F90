! THIS VERSION: GALAHAD 4.2 - 2023-12-20 AT 11:20 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  C L L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.2. October 6 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_CLLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_CLLS_precision, ONLY:                                          &
        f_clls_control_type   => CLLS_control_type,                            &
        f_clls_time_type      => CLLS_time_type,                               &
        f_clls_inform_type    => CLLS_inform_type,                             &
        f_clls_full_data_type => CLLS_full_data_type,                          &
        f_clls_initialize     => CLLS_initialize,                              &
        f_clls_read_specfile  => CLLS_read_specfile,                           &
        f_clls_import         => CLLS_import,                                  &
        f_clls_reset_control  => CLLS_reset_control,                           &
        f_clls_solve_clls     => CLLS_solve_clls,                              &
        f_clls_information    => CLLS_information,                             &
        f_clls_terminate      => CLLS_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in   => copy_inform_in,                                &
        copy_sls_inform_out  => copy_inform_out,                               &
        copy_sls_control_in  => copy_control_in,                               &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_FDC_precision_ciface, ONLY:                                    &
        fdc_inform_type,                                                       &
        fdc_control_type,                                                      &
        copy_fdc_inform_in   => copy_inform_in,                                &
        copy_fdc_inform_out  => copy_inform_out,                               &
        copy_fdc_control_in  => copy_control_in,                               &
        copy_fdc_control_out => copy_control_out

    USE GALAHAD_FIT_precision_ciface, ONLY:                                    &
        fit_inform_type,                                                       &
        fit_control_type,                                                      &
        copy_fit_inform_in   => copy_inform_in,                                &
        copy_fit_inform_out  => copy_inform_out,                               &
        copy_fit_control_in  => copy_control_in,                               &
        copy_fit_control_out => copy_control_out

    USE GALAHAD_ROOTS_precision_ciface, ONLY:                                  &
        roots_inform_type,                                                     &
        roots_control_type,                                                    &
        copy_roots_inform_in   => copy_inform_in,                              &
        copy_roots_inform_out  => copy_inform_out,                             &
        copy_roots_control_in  => copy_control_in,                             &
        copy_roots_control_out => copy_control_out

    USE GALAHAD_CRO_precision_ciface, ONLY:                                    &
        cro_inform_type,                                                       &
        cro_control_type,                                                      &
        copy_cro_inform_in   => copy_inform_in,                                &
        copy_cro_inform_out  => copy_inform_out,                               &
        copy_cro_control_in  => copy_control_in,                               &
        copy_cro_control_out => copy_control_out

    USE GALAHAD_RPD_precision_ciface, ONLY:                                    &
        rpd_inform_type,                                                       &
        rpd_control_type,                                                      &
        copy_rpd_inform_in   => copy_inform_in,                                &
        copy_rpd_inform_out  => copy_inform_out,                               &
        copy_rpd_control_in  => copy_control_in,                               &
        copy_rpd_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: clls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: infeas_max
      INTEGER ( KIND = ipc_ ) :: muzero_fixed
      INTEGER ( KIND = ipc_ ) :: restore_problem
      INTEGER ( KIND = ipc_ ) :: indicator_type
      INTEGER ( KIND = ipc_ ) :: arc
      INTEGER ( KIND = ipc_ ) :: series_order
      INTEGER ( KIND = ipc_ ) :: sif_file_device
      INTEGER ( KIND = ipc_ ) :: qplib_file_device
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: stop_abs_p
      REAL ( KIND = rpc_ ) :: stop_rel_p
      REAL ( KIND = rpc_ ) :: stop_abs_d
      REAL ( KIND = rpc_ ) :: stop_rel_d
      REAL ( KIND = rpc_ ) :: stop_abs_c
      REAL ( KIND = rpc_ ) :: stop_rel_c
      REAL ( KIND = rpc_ ) :: prfeas
      REAL ( KIND = rpc_ ) :: dufeas
      REAL ( KIND = rpc_ ) :: muzero
      REAL ( KIND = rpc_ ) :: tau
      REAL ( KIND = rpc_ ) :: gamma_c
      REAL ( KIND = rpc_ ) :: gamma_f
      REAL ( KIND = rpc_ ) :: reduce_infeas
      REAL ( KIND = rpc_ ) :: identical_bounds_tol
      REAL ( KIND = rpc_ ) :: mu_pounce
      REAL ( KIND = rpc_ ) :: indicator_tol_p
      REAL ( KIND = rpc_ ) :: indicator_tol_pd
      REAL ( KIND = rpc_ ) :: indicator_tol_tapia
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: remove_dependencies
      LOGICAL ( KIND = C_BOOL ) :: treat_zero_bounds_as_general
      LOGICAL ( KIND = C_BOOL ) :: treat_separable_as_general
      LOGICAL ( KIND = C_BOOL ) :: just_feasible
      LOGICAL ( KIND = C_BOOL ) :: getdua
      LOGICAL ( KIND = C_BOOL ) :: puiseux
      LOGICAL ( KIND = C_BOOL ) :: every_order
      LOGICAL ( KIND = C_BOOL ) :: feasol
      LOGICAL ( KIND = C_BOOL ) :: balance_initial_complentarity
      LOGICAL ( KIND = C_BOOL ) :: crossover
      LOGICAL ( KIND = C_BOOL ) :: reduced_pounce_system
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      LOGICAL ( KIND = C_BOOL ) :: generate_qplib_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: qplib_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( fdc_control_type ) :: fdc_control
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( sls_control_type ) :: sls_pounce_control
      TYPE ( fit_control_type ) :: fit_control
      TYPE ( roots_control_type ) :: roots_control
      TYPE ( cro_control_type ) :: cro_control
    END TYPE clls_control_type

    TYPE, BIND( C ) :: clls_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: preprocess
      REAL ( KIND = rpc_ ) :: find_dependent
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_preprocess
      REAL ( KIND = rpc_ ) :: clock_find_dependent
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE clls_time_type

    TYPE, BIND( C ) :: clls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      INTEGER ( KIND = ipc_ ) :: nfacts
      INTEGER ( KIND = ipc_ ) :: nbacts
      INTEGER ( KIND = ipc_ ) :: threads
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: primal_infeasibility
      REAL ( KIND = rpc_ ) :: dual_infeasibility
      REAL ( KIND = rpc_ ) :: complementary_slackness
      REAL ( KIND = rpc_ ) :: non_negligible_pivot
      LOGICAL ( KIND = C_BOOL ) :: feasible
      INTEGER ( KIND = ipc_ ), DIMENSION( 16 ) :: checkpointsIter
      REAL ( KIND = rpc_ ), DIMENSION( 16 ) :: checkpointsTime
      TYPE ( clls_time_type ) :: time
      TYPE ( fdc_inform_type ) :: fdc_inform
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( sls_inform_type ) :: sls_pounce_inform
      TYPE ( fit_inform_type ) :: fit_inform
      TYPE ( roots_inform_type ) :: roots_inform
      TYPE ( cro_inform_type ) :: cro_inform
      TYPE ( rpd_inform_type ) :: rpd_inform
    END TYPE clls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( clls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_clls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%maxit = ccontrol%maxit
    fcontrol%infeas_max = ccontrol%infeas_max
    fcontrol%muzero_fixed = ccontrol%muzero_fixed
    fcontrol%restore_problem = ccontrol%restore_problem
    fcontrol%indicator_type = ccontrol%indicator_type
    fcontrol%arc = ccontrol%arc
    fcontrol%series_order = ccontrol%series_order
    fcontrol%sif_file_device = ccontrol%sif_file_device
    fcontrol%qplib_file_device = ccontrol%qplib_file_device

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%stop_abs_p = ccontrol%stop_abs_p
    fcontrol%stop_rel_p = ccontrol%stop_rel_p
    fcontrol%stop_abs_d = ccontrol%stop_abs_d
    fcontrol%stop_rel_d = ccontrol%stop_rel_d
    fcontrol%stop_abs_c = ccontrol%stop_abs_c
    fcontrol%stop_rel_c = ccontrol%stop_rel_c
    fcontrol%prfeas = ccontrol%prfeas
    fcontrol%dufeas = ccontrol%dufeas
    fcontrol%muzero = ccontrol%muzero
    fcontrol%tau = ccontrol%tau
    fcontrol%gamma_c = ccontrol%gamma_c
    fcontrol%gamma_f = ccontrol%gamma_f
    fcontrol%reduce_infeas = ccontrol%reduce_infeas
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%mu_pounce = ccontrol%mu_pounce
    fcontrol%indicator_tol_p = ccontrol%indicator_tol_p
    fcontrol%indicator_tol_pd = ccontrol%indicator_tol_pd
    fcontrol%indicator_tol_tapia = ccontrol%indicator_tol_tapia
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%remove_dependencies = ccontrol%remove_dependencies
    fcontrol%treat_zero_bounds_as_general                                      &
      = ccontrol%treat_zero_bounds_as_general
    fcontrol%treat_separable_as_general = ccontrol%treat_separable_as_general
    fcontrol%just_feasible = ccontrol%just_feasible
    fcontrol%getdua = ccontrol%getdua
    fcontrol%puiseux = ccontrol%puiseux
    fcontrol%every_order = ccontrol%every_order
    fcontrol%feasol = ccontrol%feasol
    fcontrol%balance_initial_complentarity                                     &
      = ccontrol%balance_initial_complentarity
    fcontrol%crossover = ccontrol%crossover
    fcontrol%reduced_pounce_system = ccontrol%reduced_pounce_system
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file
    fcontrol%generate_qplib_file = ccontrol%generate_qplib_file

    ! Derived types
    CALL copy_fdc_control_in( ccontrol%fdc_control, fcontrol%fdc_control )
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_sls_control_in( ccontrol%sls_pounce_control,                     &
                              fcontrol%sls_pounce_control )
    CALL copy_fit_control_in( ccontrol%fit_control, fcontrol%fit_control )
    CALL copy_roots_control_in( ccontrol%roots_control, fcontrol%roots_control )
    CALL copy_cro_control_in( ccontrol%cro_control, fcontrol%cro_control )

    ! Strings
    DO i = 1, LEN( fcontrol%symmetric_linear_solver )
      IF ( ccontrol%symmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%symmetric_linear_solver( i : i )                                &
        = ccontrol%symmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%qplib_file_name )
      IF ( ccontrol%qplib_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%qplib_file_name( i : i ) = ccontrol%qplib_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_clls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( clls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%maxit = fcontrol%maxit
    ccontrol%infeas_max = fcontrol%infeas_max
    ccontrol%muzero_fixed = fcontrol%muzero_fixed
    ccontrol%restore_problem = fcontrol%restore_problem
    ccontrol%indicator_type = fcontrol%indicator_type
    ccontrol%arc = fcontrol%arc
    ccontrol%series_order = fcontrol%series_order
    ccontrol%sif_file_device = fcontrol%sif_file_device
    ccontrol%qplib_file_device = fcontrol%qplib_file_device

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%stop_abs_p = fcontrol%stop_abs_p
    ccontrol%stop_rel_p = fcontrol%stop_rel_p
    ccontrol%stop_abs_d = fcontrol%stop_abs_d
    ccontrol%stop_rel_d = fcontrol%stop_rel_d
    ccontrol%stop_abs_c = fcontrol%stop_abs_c
    ccontrol%stop_rel_c = fcontrol%stop_rel_c
    ccontrol%prfeas = fcontrol%prfeas
    ccontrol%dufeas = fcontrol%dufeas
    ccontrol%muzero = fcontrol%muzero
    ccontrol%tau = fcontrol%tau
    ccontrol%gamma_c = fcontrol%gamma_c
    ccontrol%gamma_f = fcontrol%gamma_f
    ccontrol%reduce_infeas = fcontrol%reduce_infeas
    ccontrol%mu_pounce = fcontrol%mu_pounce
    ccontrol%indicator_tol_p = fcontrol%indicator_tol_p
    ccontrol%indicator_tol_pd = fcontrol%indicator_tol_pd
    ccontrol%indicator_tol_tapia = fcontrol%indicator_tol_tapia
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%remove_dependencies = fcontrol%remove_dependencies
    ccontrol%treat_zero_bounds_as_general                                      &
      = fcontrol%treat_zero_bounds_as_general
    ccontrol%treat_separable_as_general = fcontrol%treat_separable_as_general
    ccontrol%just_feasible = fcontrol%just_feasible
    ccontrol%getdua = fcontrol%getdua
    ccontrol%puiseux = fcontrol%puiseux
    ccontrol%every_order = fcontrol%every_order
    ccontrol%feasol = fcontrol%feasol
    ccontrol%balance_initial_complentarity                                     &
      = fcontrol%balance_initial_complentarity
    ccontrol%crossover = fcontrol%crossover
    ccontrol%reduced_pounce_system = fcontrol%reduced_pounce_system
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file
    ccontrol%generate_qplib_file = fcontrol%generate_qplib_file

    ! Derived types
    CALL copy_fdc_control_out( fcontrol%fdc_control, ccontrol%fdc_control )
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_sls_control_out( fcontrol%sls_pounce_control,                    &
                               ccontrol%sls_pounce_control )
    CALL copy_fit_control_out( fcontrol%fit_control, ccontrol%fit_control )
    CALL copy_roots_control_out( fcontrol%roots_control, ccontrol%roots_control)
    CALL copy_cro_control_out( fcontrol%cro_control, ccontrol%cro_control )

    ! Strings
    l = LEN( fcontrol%symmetric_linear_solver )
    DO i = 1, l
      ccontrol%symmetric_linear_solver( i )                                    &
        = fcontrol%symmetric_linear_solver( i : i )
    END DO
    ccontrol%symmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%qplib_file_name )
    DO i = 1, l
      ccontrol%qplib_file_name( i ) = fcontrol%qplib_file_name( i : i )
    END DO
    ccontrol%qplib_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( clls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_clls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%find_dependent = ctime%find_dependent
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_find_dependent = ctime%clock_find_dependent
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_clls_time_type ), INTENT( IN ) :: ftime
    TYPE ( clls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%find_dependent = ftime%find_dependent
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_find_dependent = ftime%clock_find_dependent
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( clls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_clls_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%factorization_status = cinform%factorization_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%nfacts = cinform%nfacts
    finform%nbacts = cinform%nbacts
    finform%threads = cinform%threads
    finform%checkpointsIter = cinform%checkpointsIter

    ! Reals
    finform%obj = cinform%obj
    finform%primal_infeasibility = cinform%primal_infeasibility
    finform%dual_infeasibility = cinform%dual_infeasibility
    finform%complementary_slackness = cinform%complementary_slackness
    finform%non_negligible_pivot = cinform%non_negligible_pivot
    finform%checkpointsTime = cinform%checkpointsTime

    ! Logicals
    finform%feasible = cinform%feasible

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_fdc_inform_in( cinform%fdc_inform, finform%fdc_inform )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_sls_inform_in( cinform%sls_pounce_inform,                        &
                             finform%sls_pounce_inform )
    CALL copy_fit_inform_in( cinform%fit_inform, finform%fit_inform )
    CALL copy_roots_inform_in( cinform%roots_inform, finform%roots_inform )
    CALL copy_cro_inform_in( cinform%cro_inform, finform%cro_inform )
    CALL copy_rpd_inform_in( cinform%rpd_inform, finform%rpd_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_clls_inform_type ), INTENT( IN ) :: finform
    TYPE ( clls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%factorization_status = finform%factorization_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%nfacts = finform%nfacts
    cinform%nbacts = finform%nbacts
    cinform%threads = finform%threads
    cinform%checkpointsIter = finform%checkpointsIter

    ! Reals
    cinform%obj = finform%obj
    cinform%primal_infeasibility = finform%primal_infeasibility
    cinform%dual_infeasibility = finform%dual_infeasibility
    cinform%complementary_slackness = finform%complementary_slackness
    cinform%non_negligible_pivot = finform%non_negligible_pivot
    cinform%checkpointsTime = finform%checkpointsTime

    ! Logicals
    cinform%feasible = finform%feasible

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_fdc_inform_out( finform%fdc_inform, cinform%fdc_inform )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_sls_inform_out( finform%sls_pounce_inform,                       &
                              cinform%sls_pounce_inform )
    CALL copy_fit_inform_out( finform%fit_inform, cinform%fit_inform )
    CALL copy_roots_inform_out( finform%roots_inform, cinform%roots_inform )
    CALL copy_cro_inform_out( finform%cro_inform, cinform%cro_inform )
    CALL copy_rpd_inform_out( finform%rpd_inform, cinform%rpd_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_CLLS_precision_ciface

!  -------------------------------------
!  C interface to fortran clls_initialize
!  -------------------------------------

  SUBROUTINE clls_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( clls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_clls_full_data_type ), POINTER :: fdata
  TYPE ( f_clls_control_type ) :: fcontrol
  TYPE ( f_clls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_clls_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE clls_initialize

!  ----------------------------------------
!  C interface to fortran clls_read_specfile
!  ----------------------------------------

  SUBROUTINE clls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( clls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_clls_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_clls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE clls_read_specfile

!  ---------------------------------
!  C interface to fortran clls_inport
!  ---------------------------------

  SUBROUTINE clls_import( ccontrol, cdata, status, n, o, m,                    &
                          caotype, aone, aorow, aocol, aoptrne, aoptr,         &
                          catype, ane, arow, acol, aptrne, aptr ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( clls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, o, m
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: aone, aoptrne, ane, aptrne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aone ), OPTIONAL :: aorow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aone ), OPTIONAL :: aocol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aoptrne ), OPTIONAL :: aoptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: caotype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aptrne ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( caotype ) ) :: faotype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_clls_control_type ) :: fcontrol
  TYPE ( f_clls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  faotype = cstr_to_fchar( caotype )
  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required CLLS structure

  CALL f_clls_import( fcontrol, fdata, status, n, o, m,                        &
                      faotype, aone, aorow, aocol, aoptr,                      &
                      fatype, ane, arow, acol, aptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE clls_import

!  ----------------------------------------
!  C interface to fortran clls_reset_control
!  ----------------------------------------

  SUBROUTINE clls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( clls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_clls_control_type ) :: fcontrol
  TYPE ( f_clls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_clls_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE clls_reset_control

!  ------------------------------------
!  C interface to fortran clls_solve_clls
!  ------------------------------------

  SUBROUTINE clls_solve_clls( cdata, status, n, o, m, aone, aoval, b,          &
                              regularization_weight, ane, aval, cl, cu,        &
                              xl, xu, x, r, c, y, z, xstat, cstat, w ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, o, m, aone, ane
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( aone ) :: aoval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( o ) :: b
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: regularization_weight
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( o ) :: r
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( m ) :: cstat
  REAL ( KIND = rpc_ ), INTENT( IN ), OPTIONAL, DIMENSION( o ) :: w
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_clls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_clls_solve_clls( fdata, status, aoval, b, aval, cl, cu, xl, xu, x,    &
                          r, c, y, z, xstat, cstat, regularization_weight, w )
  RETURN

  END SUBROUTINE clls_solve_clls

!  --------------------------------------
!  C interface to fortran clls_information
!  --------------------------------------

  SUBROUTINE clls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( clls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_clls_full_data_type ), pointer :: fdata
  TYPE ( f_clls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain CLLS solution information

  CALL f_clls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE clls_information

!  ------------------------------------
!  C interface to fortran clls_terminate
!  ------------------------------------

  SUBROUTINE clls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_CLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( clls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( clls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_clls_full_data_type ), pointer :: fdata
  TYPE ( f_clls_control_type ) :: fcontrol
  TYPE ( f_clls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_clls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE clls_terminate
