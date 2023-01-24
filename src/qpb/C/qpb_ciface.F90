! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  Q P B    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 7th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_QPB_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_QPB_precision, ONLY:                                           &
        f_qpb_control_type   => QPB_control_type,                              &
        f_qpb_time_type      => QPB_time_type,                                 &
        f_qpb_inform_type    => QPB_inform_type,                               &
        f_qpb_full_data_type => QPB_full_data_type,                            &
        f_qpb_initialize     => QPB_initialize,                                &
        f_qpb_read_specfile  => QPB_read_specfile,                             &
        f_qpb_import         => QPB_import,                                    &
        f_qpb_reset_control  => QPB_reset_control,                             &
        f_qpb_solve_qp       => QPB_solve_qp,                                  &
        f_qpb_information    => QPB_information,                               &
        f_qpb_terminate      => QPB_terminate

    USE GALAHAD_LSQP_precision_ciface, ONLY:                                   &
        lsqp_inform_type,                                                      &
        lsqp_control_type,                                                     &
        copy_lsqp_inform_in   => copy_inform_in,                               &
        copy_lsqp_inform_out  => copy_inform_out,                              &
        copy_lsqp_control_in  => copy_control_in,                              &
        copy_lsqp_control_out => copy_control_out

    USE GALAHAD_FDC_precision_ciface, ONLY:                                    &
        fdc_inform_type,                                                       &
        fdc_control_type,                                                      &
        copy_fdc_inform_in   => copy_inform_in,                                &
        copy_fdc_inform_out  => copy_inform_out,                               &
        copy_fdc_control_in  => copy_control_in,                               &
        copy_fdc_control_out => copy_control_out

    USE GALAHAD_SBLS_precision_ciface, ONLY:                                   &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in   => copy_inform_in,                               &
        copy_sbls_inform_out  => copy_inform_out,                              &
        copy_sbls_control_in  => copy_control_in,                              &
        copy_sbls_control_out => copy_control_out

    USE GALAHAD_GLTR_precision_ciface, ONLY:                                   &
        gltr_inform_type,                                                      &
        gltr_control_type,                                                     &
        copy_gltr_inform_in   => copy_inform_in,                               &
        copy_gltr_inform_out  => copy_inform_out,                              &
        copy_gltr_control_in  => copy_control_in,                              &
        copy_gltr_control_out => copy_control_out

    USE GALAHAD_FIT_precision_ciface, ONLY:                                    &
        fit_inform_type,                                                       &
        fit_control_type,                                                      &
        copy_fit_inform_in => copy_inform_in,                                  &
        copy_fit_inform_out => copy_inform_out,                                &
        copy_fit_control_in => copy_control_in,                                &
        copy_fit_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: qpb_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: itref_max
      INTEGER ( KIND = ipc_ ) :: cg_maxit
      INTEGER ( KIND = ipc_ ) :: indicator_type
      INTEGER ( KIND = ipc_ ) :: restore_problem
      INTEGER ( KIND = ipc_ ) :: extrapolate
      INTEGER ( KIND = ipc_ ) :: path_history
      INTEGER ( KIND = ipc_ ) :: factor
      INTEGER ( KIND = ipc_ ) :: max_col
      INTEGER ( KIND = ipc_ ) :: indmin
      INTEGER ( KIND = ipc_ ) :: valmin
      INTEGER ( KIND = ipc_ ) :: infeas_max
      INTEGER ( KIND = ipc_ ) :: precon
      INTEGER ( KIND = ipc_ ) :: nsemib
      INTEGER ( KIND = ipc_ ) :: path_derivatives
      INTEGER ( KIND = ipc_ ) :: fit_order
      INTEGER ( KIND = ipc_ ) :: sif_file_device
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: stop_p
      REAL ( KIND = rpc_ ) :: stop_d
      REAL ( KIND = rpc_ ) :: stop_c
      REAL ( KIND = rpc_ ) :: theta_d
      REAL ( KIND = rpc_ ) :: theta_c
      REAL ( KIND = rpc_ ) :: beta
      REAL ( KIND = rpc_ ) :: prfeas
      REAL ( KIND = rpc_ ) :: dufeas
      REAL ( KIND = rpc_ ) :: muzero
      REAL ( KIND = rpc_ ) :: reduce_infeas
      REAL ( KIND = rpc_ ) :: obj_unbounded
      REAL ( KIND = rpc_ ) :: pivot_tol
      REAL ( KIND = rpc_ ) :: pivot_tol_for_dependencies
      REAL ( KIND = rpc_ ) :: zero_pivot
      REAL ( KIND = rpc_ ) :: identical_bounds_tol
      REAL ( KIND = rpc_ ) :: inner_stop_relative
      REAL ( KIND = rpc_ ) :: inner_stop_absolute
      REAL ( KIND = rpc_ ) :: initial_radius
      REAL ( KIND = rpc_ ) :: mu_min
      REAL ( KIND = rpc_ ) :: inner_fraction_opt
      REAL ( KIND = rpc_ ) :: indicator_tol_p
      REAL ( KIND = rpc_ ) :: indicator_tol_pd
      REAL ( KIND = rpc_ ) :: indicator_tol_tapia
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: remove_dependencies
      LOGICAL ( KIND = C_BOOL ) :: treat_zero_bounds_as_general
      LOGICAL ( KIND = C_BOOL ) :: center
      LOGICAL ( KIND = C_BOOL ) :: primal
      LOGICAL ( KIND = C_BOOL ) :: puiseux
      LOGICAL ( KIND = C_BOOL ) :: feasol
      LOGICAL ( KIND = C_BOOL ) :: array_syntax_worse_than_do_loop
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( lsqp_control_type ) :: lsqp_control
      TYPE ( fdc_control_type ) :: fdc_control
      TYPE ( sbls_control_type ) :: sbls_control
      TYPE ( gltr_control_type ) :: gltr_control
      TYPE ( fit_control_type ) :: fit_control
    END TYPE qpb_control_type

    TYPE, BIND( C ) :: qpb_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: preprocess
      REAL ( KIND = rpc_ ) :: find_dependent
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: phase1_total
      REAL ( KIND = rpc_ ) :: phase1_analyse
      REAL ( KIND = rpc_ ) :: phase1_factorize
      REAL ( KIND = rpc_ ) :: phase1_solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_preprocess
      REAL ( KIND = rpc_ ) :: clock_find_dependent
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
      REAL ( KIND = rpc_ ) :: clock_phase1_total
      REAL ( KIND = rpc_ ) :: clock_phase1_analyse
      REAL ( KIND = rpc_ ) :: clock_phase1_factorize
      REAL ( KIND = rpc_ ) :: clock_phase1_solve
    END TYPE qpb_time_type

    TYPE, BIND( C ) :: qpb_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: cg_iter
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      INTEGER ( KIND = ipc_ ) :: nfacts
      INTEGER ( KIND = ipc_ ) :: nbacts
      INTEGER ( KIND = ipc_ ) :: nmods
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: non_negligible_pivot
      LOGICAL ( KIND = C_BOOL ) :: feasible
      TYPE ( qpb_time_type ) :: time
      TYPE ( lsqp_inform_type ) :: lsqp_inform
      TYPE ( fdc_inform_type ) :: fdc_inform
      TYPE ( sbls_inform_type ) :: sbls_inform
      TYPE ( gltr_inform_type ) :: gltr_inform
      TYPE ( fit_inform_type ) :: fit_inform
    END TYPE qpb_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( qpb_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_qpb_control_type ), INTENT( OUT ) :: fcontrol
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
    fcontrol%itref_max = ccontrol%itref_max
    fcontrol%cg_maxit = ccontrol%cg_maxit
    fcontrol%indicator_type = ccontrol%indicator_type
    fcontrol%restore_problem = ccontrol%restore_problem
    fcontrol%extrapolate = ccontrol%extrapolate
    fcontrol%path_history = ccontrol%path_history
    fcontrol%factor = ccontrol%factor
    fcontrol%max_col = ccontrol%max_col
    fcontrol%indmin = ccontrol%indmin
    fcontrol%valmin = ccontrol%valmin
    fcontrol%infeas_max = ccontrol%infeas_max
    fcontrol%precon = ccontrol%precon
    fcontrol%nsemib = ccontrol%nsemib
    fcontrol%path_derivatives = ccontrol%path_derivatives
    fcontrol%fit_order = ccontrol%fit_order
    fcontrol%sif_file_device = ccontrol%sif_file_device

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%stop_p = ccontrol%stop_p
    fcontrol%stop_d = ccontrol%stop_d
    fcontrol%stop_c = ccontrol%stop_c
    fcontrol%theta_d = ccontrol%theta_d
    fcontrol%theta_c = ccontrol%theta_c
    fcontrol%beta = ccontrol%beta
    fcontrol%prfeas = ccontrol%prfeas
    fcontrol%dufeas = ccontrol%dufeas
    fcontrol%muzero = ccontrol%muzero
    fcontrol%reduce_infeas = ccontrol%reduce_infeas
    fcontrol%obj_unbounded = ccontrol%obj_unbounded
    fcontrol%pivot_tol = ccontrol%pivot_tol
    fcontrol%pivot_tol_for_dependencies = ccontrol%pivot_tol_for_dependencies
    fcontrol%zero_pivot = ccontrol%zero_pivot
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%inner_stop_relative = ccontrol%inner_stop_relative
    fcontrol%inner_stop_absolute = ccontrol%inner_stop_absolute
    fcontrol%initial_radius = ccontrol%initial_radius
    fcontrol%mu_min = ccontrol%mu_min
    fcontrol%inner_fraction_opt = ccontrol%inner_fraction_opt
    fcontrol%indicator_tol_p = ccontrol%indicator_tol_p
    fcontrol%indicator_tol_pd = ccontrol%indicator_tol_pd
    fcontrol%indicator_tol_tapia = ccontrol%indicator_tol_tapia
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%remove_dependencies = ccontrol%remove_dependencies
    fcontrol%treat_zero_bounds_as_general                                      &
      = ccontrol%treat_zero_bounds_as_general
    fcontrol%center = ccontrol%center
    fcontrol%primal = ccontrol%primal
    fcontrol%puiseux = ccontrol%puiseux
    fcontrol%feasol = ccontrol%feasol
    fcontrol%array_syntax_worse_than_do_loop                                   &
      = ccontrol%array_syntax_worse_than_do_loop
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file

    ! Derived types
    CALL copy_lsqp_control_in( ccontrol%lsqp_control, fcontrol%lsqp_control )
    CALL copy_fdc_control_in( ccontrol%fdc_control, fcontrol%fdc_control )
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
    CALL copy_gltr_control_in( ccontrol%gltr_control, fcontrol%gltr_control )
    CALL copy_fit_control_in( ccontrol%fit_control, fcontrol%fit_control )

    ! Strings
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_qpb_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( qpb_control_type ), INTENT( OUT ) :: ccontrol
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
    ccontrol%itref_max = fcontrol%itref_max
    ccontrol%cg_maxit = fcontrol%cg_maxit
    ccontrol%indicator_type = fcontrol%indicator_type
    ccontrol%restore_problem = fcontrol%restore_problem
    ccontrol%extrapolate = fcontrol%extrapolate
    ccontrol%path_history = fcontrol%path_history
    ccontrol%factor = fcontrol%factor
    ccontrol%max_col = fcontrol%max_col
    ccontrol%indmin = fcontrol%indmin
    ccontrol%valmin = fcontrol%valmin
    ccontrol%infeas_max = fcontrol%infeas_max
    ccontrol%precon = fcontrol%precon
    ccontrol%nsemib = fcontrol%nsemib
    ccontrol%path_derivatives = fcontrol%path_derivatives
    ccontrol%fit_order = fcontrol%fit_order
    ccontrol%sif_file_device = fcontrol%sif_file_device

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%stop_p = fcontrol%stop_p
    ccontrol%stop_d = fcontrol%stop_d
    ccontrol%stop_c = fcontrol%stop_c
    ccontrol%theta_d = fcontrol%theta_d
    ccontrol%theta_c = fcontrol%theta_c
    ccontrol%beta = fcontrol%beta
    ccontrol%prfeas = fcontrol%prfeas
    ccontrol%dufeas = fcontrol%dufeas
    ccontrol%muzero = fcontrol%muzero
    ccontrol%reduce_infeas = fcontrol%reduce_infeas
    ccontrol%obj_unbounded = fcontrol%obj_unbounded
    ccontrol%pivot_tol = fcontrol%pivot_tol
    ccontrol%pivot_tol_for_dependencies = fcontrol%pivot_tol_for_dependencies
    ccontrol%zero_pivot = fcontrol%zero_pivot
    ccontrol%identical_bounds_tol = fcontrol%identical_bounds_tol
    ccontrol%inner_stop_relative = fcontrol%inner_stop_relative
    ccontrol%inner_stop_absolute = fcontrol%inner_stop_absolute
    ccontrol%initial_radius = fcontrol%initial_radius
    ccontrol%mu_min = fcontrol%mu_min
    ccontrol%inner_fraction_opt = fcontrol%inner_fraction_opt
    ccontrol%indicator_tol_p = fcontrol%indicator_tol_p
    ccontrol%indicator_tol_pd = fcontrol%indicator_tol_pd
    ccontrol%indicator_tol_tapia = fcontrol%indicator_tol_tapia
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%remove_dependencies = fcontrol%remove_dependencies
    ccontrol%treat_zero_bounds_as_general                                      &
      = fcontrol%treat_zero_bounds_as_general
    ccontrol%center = fcontrol%center
    ccontrol%primal = fcontrol%primal
    ccontrol%puiseux = fcontrol%puiseux
    ccontrol%feasol = fcontrol%feasol
    ccontrol%array_syntax_worse_than_do_loop                                   &
      = fcontrol%array_syntax_worse_than_do_loop
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file

    ! Derived types
    CALL copy_lsqp_control_out( fcontrol%lsqp_control, ccontrol%lsqp_control )
    CALL copy_fdc_control_out( fcontrol%fdc_control, ccontrol%fdc_control )
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
    CALL copy_gltr_control_out( fcontrol%gltr_control, ccontrol%gltr_control )
    CALL copy_fit_control_out( fcontrol%fit_control, ccontrol%fit_control )

    ! Strings
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( qpb_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_qpb_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%find_dependent = ctime%find_dependent
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%phase1_total = ctime%phase1_total
    ftime%phase1_analyse = ctime%phase1_analyse
    ftime%phase1_factorize = ctime%phase1_factorize
    ftime%phase1_solve = ctime%phase1_solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_find_dependent = ctime%clock_find_dependent
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%clock_phase1_total = ctime%clock_phase1_total
    ftime%clock_phase1_analyse = ctime%clock_phase1_analyse
    ftime%clock_phase1_factorize = ctime%clock_phase1_factorize
    ftime%clock_phase1_solve = ctime%clock_phase1_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_qpb_time_type ), INTENT( IN ) :: ftime
    TYPE ( qpb_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%find_dependent = ftime%find_dependent
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%phase1_total = ftime%phase1_total
    ctime%phase1_analyse = ftime%phase1_analyse
    ctime%phase1_factorize = ftime%phase1_factorize
    ctime%phase1_solve = ftime%phase1_solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_find_dependent = ftime%clock_find_dependent
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%clock_phase1_total = ftime%clock_phase1_total
    ctime%clock_phase1_analyse = ftime%clock_phase1_analyse
    ctime%clock_phase1_factorize = ftime%clock_phase1_factorize
    ctime%clock_phase1_solve = ftime%clock_phase1_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( qpb_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_qpb_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%cg_iter = cinform%cg_iter
    finform%factorization_status = cinform%factorization_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%nfacts = cinform%nfacts
    finform%nbacts = cinform%nbacts
    finform%nmods = cinform%nmods

    ! Reals
    finform%obj = cinform%obj
    finform%non_negligible_pivot = cinform%non_negligible_pivot

    ! Logicals
    finform%feasible = cinform%feasible

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_lsqp_inform_in( cinform%lsqp_inform, finform%lsqp_inform )
    CALL copy_fdc_inform_in( cinform%fdc_inform, finform%fdc_inform )
    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )
    CALL copy_gltr_inform_in( cinform%gltr_inform, finform%gltr_inform )
    CALL copy_fit_inform_in( cinform%fit_inform, finform%fit_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_qpb_inform_type ), INTENT( IN ) :: finform
    TYPE ( qpb_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%cg_iter = finform%cg_iter
    cinform%factorization_status = finform%factorization_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%nfacts = finform%nfacts
    cinform%nbacts = finform%nbacts
    cinform%nmods = finform%nmods

    ! Reals
    cinform%obj = finform%obj
    cinform%non_negligible_pivot = finform%non_negligible_pivot

    ! Logicals
    cinform%feasible = finform%feasible

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_lsqp_inform_out( finform%lsqp_inform, cinform%lsqp_inform )
    CALL copy_fdc_inform_out( finform%fdc_inform, cinform%fdc_inform )
    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )
    CALL copy_gltr_inform_out( finform%gltr_inform, cinform%gltr_inform )
    CALL copy_fit_inform_out( finform%fit_inform, cinform%fit_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_QPB_precision_ciface

!  -------------------------------------
!  C interface to fortran qpb_initialize
!  -------------------------------------

  SUBROUTINE qpb_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( qpb_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_qpb_full_data_type ), POINTER :: fdata
  TYPE ( f_qpb_control_type ) :: fcontrol
  TYPE ( f_qpb_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_qpb_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE qpb_initialize

!  ----------------------------------------
!  C interface to fortran qpb_read_specfile
!  ----------------------------------------

  SUBROUTINE qpb_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( qpb_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_qpb_control_type ) :: fcontrol
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

  CALL f_qpb_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE qpb_read_specfile

!  ---------------------------------
!  C interface to fortran qpb_inport
!  ---------------------------------

  SUBROUTINE qpb_import( ccontrol, cdata, status, n, m,                        &
                         chtype, hne, hrow, hcol, hptr,                        &
                         catype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( qpb_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, ane
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_qpb_control_type ) :: fcontrol
  TYPE ( f_qpb_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )
  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required QPB structure

  CALL f_qpb_import( fcontrol, fdata, status, n, m,                            &
                     fhtype, hne, hrow, hcol, hptr,                            &
                     fatype, ane, arow, acol, aptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE qpb_import

!  ----------------------------------------
!  C interface to fortran qpb_reset_control
!  ----------------------------------------

  SUBROUTINE qpb_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( qpb_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_qpb_control_type ) :: fcontrol
  TYPE ( f_qpb_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_qpb_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE qpb_reset_control

!  -----------------------------------
!  C interface to fortran qpb_solve_qp
!  -----------------------------------

  SUBROUTINE qpb_solve_qp( cdata, status, n, m, hne, hval, g, f, ane, aval,    &
                           cl, cu, xl, xu, x, c, y, z, xstat, cstat ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, ane, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_qpb_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_qpb_solve_qp( fdata, status, hval, g, f, aval, cl, cu, xl, xu,        &
                       x, c, y, z, xstat, cstat )
  RETURN

  END SUBROUTINE qpb_solve_qp

!  --------------------------------------
!  C interface to fortran qpb_information
!  --------------------------------------

  SUBROUTINE qpb_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( qpb_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_qpb_full_data_type ), pointer :: fdata
  TYPE ( f_qpb_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain QPB solution information

  CALL f_qpb_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE qpb_information

!  ------------------------------------
!  C interface to fortran qpb_terminate
!  ------------------------------------

  SUBROUTINE qpb_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_QPB_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( qpb_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( qpb_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_qpb_full_data_type ), pointer :: fdata
  TYPE ( f_qpb_control_type ) :: fcontrol
  TYPE ( f_qpb_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_qpb_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE qpb_terminate
