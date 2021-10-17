! THIS VERSION: GALAHAD 3.3 - 03/09/2021 AT 09:11 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  C Q P    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. September 3rd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_CQP_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_CQP_double, ONLY: &
        f_cqp_control_type   => CQP_control_type,                              &
        f_cqp_time_type      => CQP_time_type,                                 &
        f_cqp_inform_type    => CQP_inform_type,                               &
        f_cqp_full_data_type => CQP_full_data_type,                            &
        f_cqp_initialize     => CQP_initialize,                                &
        f_cqp_read_specfile  => CQP_read_specfile,                             &
        f_cqp_import         => CQP_import,                                    &
        f_cqp_reset_control  => CQP_reset_control,                             &
        f_cqp_information    => CQP_information,                               &
        f_cqp_terminate      => CQP_terminate

!!$    USE GALAHAD_FDC_double_ciface, ONLY: &
!!$        fdc_inform_type, &
!!$        fdc_control_type, &
!!$        copy_fdc_inform_in => copy_inform_in, &
!!$        copy_fdc_inform_out => copy_inform_out, &
!!$        copy_fdc_control_in => copy_control_in, &
!!$        copy_fdc_control_out => copy_control_out
!!$
!!$    USE GALAHAD_SBLS_double_ciface, ONLY: &
!!$        sbls_inform_type, &
!!$        sbls_control_type, &
!!$        copy_sbls_inform_in => copy_inform_in, &
!!$        copy_sbls_inform_out => copy_inform_out, &
!!$        copy_sbls_control_in => copy_control_in, &
!!$        copy_sbls_control_out => copy_control_out
!!$
!!$    USE GALAHAD_FIT_double_ciface, ONLY: &
!!$        fit_inform_type, &
!!$        fit_control_type, &
!!$        copy_fit_inform_in => copy_inform_in, &
!!$        copy_fit_inform_out => copy_inform_out, &
!!$        copy_fit_control_in => copy_control_in, &
!!$        copy_fit_control_out => copy_control_out
!!$
!!$    USE GALAHAD_ROOTS_double_ciface, ONLY: &
!!$        roots_inform_type, &
!!$        roots_control_type, &
!!$        copy_roots_inform_in => copy_inform_in, &
!!$        copy_roots_inform_out => copy_inform_out, &
!!$        copy_roots_control_in => copy_control_in, &
!!$        copy_roots_control_out => copy_control_out
!!$
!!$    USE GALAHAD_CRO_double_ciface, ONLY: &
!!$        cro_inform_type, &
!!$        cro_control_type, &
!!$        copy_cro_inform_in => copy_inform_in, &
!!$        copy_cro_inform_out => copy_inform_out, &
!!$        copy_cro_control_in => copy_control_in, &
!!$        copy_cro_control_out => copy_control_out
!!$
!!$    USE GALAHAD_RPD_double_ciface, ONLY: &
!!$        rpd_inform_type, &
!!$        rpd_control_type, &
!!$        copy_rpd_inform_in => copy_inform_in, &
!!$        copy_rpd_inform_out => copy_inform_out, &
!!$        copy_rpd_control_in => copy_control_in, &
!!$        copy_rpd_control_out => copy_control_out

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: cqp_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: start_print
      INTEGER ( KIND = C_INT ) :: stop_print
      INTEGER ( KIND = C_INT ) :: maxit
      INTEGER ( KIND = C_INT ) :: infeas_max
      INTEGER ( KIND = C_INT ) :: muzero_fixed
      INTEGER ( KIND = C_INT ) :: restore_problem
      INTEGER ( KIND = C_INT ) :: indicator_type
      INTEGER ( KIND = C_INT ) :: arc
      INTEGER ( KIND = C_INT ) :: series_order
      INTEGER ( KIND = C_INT ) :: sif_file_device
      INTEGER ( KIND = C_INT ) :: qplib_file_device
      REAL ( KIND = wp ) :: infinity
      REAL ( KIND = wp ) :: stop_abs_p
      REAL ( KIND = wp ) :: stop_rel_p
      REAL ( KIND = wp ) :: stop_abs_d
      REAL ( KIND = wp ) :: stop_rel_d
      REAL ( KIND = wp ) :: stop_abs_c
      REAL ( KIND = wp ) :: stop_rel_c
      REAL ( KIND = wp ) :: perturb_h
      REAL ( KIND = wp ) :: prfeas
      REAL ( KIND = wp ) :: dufeas
      REAL ( KIND = wp ) :: muzero
      REAL ( KIND = wp ) :: tau
      REAL ( KIND = wp ) :: gamma_c
      REAL ( KIND = wp ) :: gamma_f
      REAL ( KIND = wp ) :: reduce_infeas
      REAL ( KIND = wp ) :: obj_unbounded
      REAL ( KIND = wp ) :: potential_unbounded
      REAL ( KIND = wp ) :: identical_bounds_tol
      REAL ( KIND = wp ) :: mu_lunge
      REAL ( KIND = wp ) :: indicator_tol_p
      REAL ( KIND = wp ) :: indicator_tol_pd
      REAL ( KIND = wp ) :: indicator_tol_tapia
      REAL ( KIND = wp ) :: cpu_time_limit
      REAL ( KIND = wp ) :: clock_time_limit
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
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      LOGICAL ( KIND = C_BOOL ) :: generate_qplib_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: qplib_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
!!$      TYPE ( fdc_control_type ) :: fdc_control
!!$      TYPE ( sbls_control_type ) :: sbls_control
!!$      TYPE ( fit_control_type ) :: fit_control
!!$      TYPE ( roots_control_type ) :: roots_control
!!$      TYPE ( cro_control_type ) :: cro_control
    END TYPE cqp_control_type

    TYPE, BIND( C ) :: cqp_time_type
      REAL ( KIND = wp ) :: total
      REAL ( KIND = wp ) :: preprocess
      REAL ( KIND = wp ) :: find_dependent
      REAL ( KIND = wp ) :: analyse
      REAL ( KIND = wp ) :: factorize
      REAL ( KIND = wp ) :: solve
      REAL ( KIND = wp ) :: clock_total
      REAL ( KIND = wp ) :: clock_preprocess
      REAL ( KIND = wp ) :: clock_find_dependent
      REAL ( KIND = wp ) :: clock_analyse
      REAL ( KIND = wp ) :: clock_factorize
      REAL ( KIND = wp ) :: clock_solve
    END TYPE cqp_time_type

    TYPE, BIND( C ) :: cqp_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = C_INT ) :: iter
      INTEGER ( KIND = C_INT ) :: factorization_status
      INTEGER ( KIND = C_INT ) :: factorization_integer
      INTEGER ( KIND = C_INT ) :: factorization_real
      INTEGER ( KIND = C_INT ) :: nfacts
      INTEGER ( KIND = C_INT ) :: nbacts
      INTEGER ( KIND = C_INT ) :: threads
      REAL ( KIND = wp ) :: obj
      REAL ( KIND = wp ) :: primal_infeasibility
      REAL ( KIND = wp ) :: dual_infeasibility
      REAL ( KIND = wp ) :: complementary_slackness
      REAL ( KIND = wp ) :: init_primal_infeasibility
      REAL ( KIND = wp ) :: init_dual_infeasibility
      REAL ( KIND = wp ) :: init_complementary_slackness
      REAL ( KIND = wp ) :: potential
      REAL ( KIND = wp ) :: non_negligible_pivot
      LOGICAL ( KIND = C_BOOL ) :: feasible
      INTEGER ( KIND = C_INT ) :: checkpointsIter
      REAL ( KIND = wp ) :: checkpointsTime
      TYPE ( cqp_time_type ) :: time
!!$      TYPE ( fdc_inform_type ) :: fdc_inform
!!$      TYPE ( sbls_inform_type ) :: sbls_inform
!!$      TYPE ( fit_inform_type ) :: fit_inform
!!$      TYPE ( roots_inform_type ) :: roots_inform
!!$      TYPE ( cro_inform_type ) :: cro_inform
!!$      TYPE ( rpd_inform_type ) :: rpd_inform
    END TYPE cqp_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( cqp_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_cqp_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
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
    fcontrol%perturb_h = ccontrol%perturb_h
    fcontrol%prfeas = ccontrol%prfeas
    fcontrol%dufeas = ccontrol%dufeas
    fcontrol%muzero = ccontrol%muzero
    fcontrol%tau = ccontrol%tau
    fcontrol%gamma_c = ccontrol%gamma_c
    fcontrol%gamma_f = ccontrol%gamma_f
    fcontrol%reduce_infeas = ccontrol%reduce_infeas
    fcontrol%obj_unbounded = ccontrol%obj_unbounded
    fcontrol%potential_unbounded = ccontrol%potential_unbounded
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%mu_lunge = ccontrol%mu_lunge
    fcontrol%indicator_tol_p = ccontrol%indicator_tol_p
    fcontrol%indicator_tol_pd = ccontrol%indicator_tol_pd
    fcontrol%indicator_tol_tapia = ccontrol%indicator_tol_tapia
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%remove_dependencies = ccontrol%remove_dependencies
    fcontrol%treat_zero_bounds_as_general = ccontrol%treat_zero_bounds_as_general
    fcontrol%treat_separable_as_general = ccontrol%treat_separable_as_general
    fcontrol%just_feasible = ccontrol%just_feasible
    fcontrol%getdua = ccontrol%getdua
    fcontrol%puiseux = ccontrol%puiseux
    fcontrol%every_order = ccontrol%every_order
    fcontrol%feasol = ccontrol%feasol
    fcontrol%balance_initial_complentarity = ccontrol%balance_initial_complentarity
    fcontrol%crossover = ccontrol%crossover
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file
    fcontrol%generate_qplib_file = ccontrol%generate_qplib_file

    ! Derived types
!!$    CALL copy_fdc_control_in( ccontrol%fdc_control, fcontrol%fdc_control )
!!$    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
!!$    CALL copy_fit_control_in( ccontrol%fit_control, fcontrol%fit_control )
!!$    CALL copy_roots_control_in( ccontrol%roots_control, fcontrol%roots_control )
!!$    CALL copy_cro_control_in( ccontrol%cro_control, fcontrol%cro_control )

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, 31
      IF ( ccontrol%qplib_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%qplib_file_name( i : i ) = ccontrol%qplib_file_name( i )
    END DO
    DO i = 1, 31
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_cqp_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( cqp_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i
    
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
    ccontrol%perturb_h = fcontrol%perturb_h
    ccontrol%prfeas = fcontrol%prfeas
    ccontrol%dufeas = fcontrol%dufeas
    ccontrol%muzero = fcontrol%muzero
    ccontrol%tau = fcontrol%tau
    ccontrol%gamma_c = fcontrol%gamma_c
    ccontrol%gamma_f = fcontrol%gamma_f
    ccontrol%reduce_infeas = fcontrol%reduce_infeas
    ccontrol%obj_unbounded = fcontrol%obj_unbounded
    ccontrol%potential_unbounded = fcontrol%potential_unbounded
    ccontrol%identical_bounds_tol = fcontrol%identical_bounds_tol
    ccontrol%mu_lunge = fcontrol%mu_lunge
    ccontrol%indicator_tol_p = fcontrol%indicator_tol_p
    ccontrol%indicator_tol_pd = fcontrol%indicator_tol_pd
    ccontrol%indicator_tol_tapia = fcontrol%indicator_tol_tapia
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%remove_dependencies = fcontrol%remove_dependencies
    ccontrol%treat_zero_bounds_as_general = fcontrol%treat_zero_bounds_as_general
    ccontrol%treat_separable_as_general = fcontrol%treat_separable_as_general
    ccontrol%just_feasible = fcontrol%just_feasible
    ccontrol%getdua = fcontrol%getdua
    ccontrol%puiseux = fcontrol%puiseux
    ccontrol%every_order = fcontrol%every_order
    ccontrol%feasol = fcontrol%feasol
    ccontrol%balance_initial_complentarity = fcontrol%balance_initial_complentarity
    ccontrol%crossover = fcontrol%crossover
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file
    ccontrol%generate_qplib_file = fcontrol%generate_qplib_file

    ! Derived types
!!$    CALL copy_fdc_control_out( fcontrol%fdc_control, ccontrol%fdc_control )
!!$    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
!!$    CALL copy_fit_control_out( fcontrol%fit_control, ccontrol%fit_control )
!!$    CALL copy_roots_control_out( fcontrol%roots_control, ccontrol%roots_control )
!!$    CALL copy_cro_control_out( fcontrol%cro_control, ccontrol%cro_control )

    ! Strings
    DO i = 1, LEN( fcontrol%sif_file_name )
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( LEN( fcontrol%sif_file_name ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( fcontrol%qplib_file_name )
      ccontrol%qplib_file_name( i ) = fcontrol%qplib_file_name( i : i )
    END DO
    ccontrol%qplib_file_name( LEN( fcontrol%qplib_file_name ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime ) 
    TYPE ( cqp_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_cqp_time_type ), INTENT( OUT ) :: ftime

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
    TYPE ( f_cqp_time_type ), INTENT( IN ) :: ftime
    TYPE ( cqp_time_type ), INTENT( OUT ) :: ctime

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
    TYPE ( cqp_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_cqp_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

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
    finform%init_primal_infeasibility = cinform%init_primal_infeasibility
    finform%init_dual_infeasibility = cinform%init_dual_infeasibility
    finform%init_complementary_slackness = cinform%init_complementary_slackness
    finform%potential = cinform%potential
    finform%non_negligible_pivot = cinform%non_negligible_pivot
    finform%checkpointsTime = cinform%checkpointsTime

    ! Logicals
    finform%feasible = cinform%feasible

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
!!$    CALL copy_fdc_inform_in( cinform%fdc_inform, finform%fdc_inform )
!!$    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )
!!$    CALL copy_fit_inform_in( cinform%fit_inform, finform%fit_inform )
!!$    CALL copy_roots_inform_in( cinform%roots_inform, finform%roots_inform )
!!$    CALL copy_cro_inform_in( cinform%cro_inform, finform%cro_inform )
!!$    CALL copy_rpd_inform_in( cinform%rpd_inform, finform%rpd_inform )

    ! Strings
    DO i = 1, 81
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_cqp_inform_type ), INTENT( IN ) :: finform
    TYPE ( cqp_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i

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
    cinform%init_primal_infeasibility = finform%init_primal_infeasibility
    cinform%init_dual_infeasibility = finform%init_dual_infeasibility
    cinform%init_complementary_slackness = finform%init_complementary_slackness
    cinform%potential = finform%potential
    cinform%non_negligible_pivot = finform%non_negligible_pivot
    cinform%checkpointsTime = finform%checkpointsTime

    ! Logicals
    cinform%feasible = finform%feasible

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
!!$    CALL copy_fdc_inform_out( finform%fdc_inform, cinform%fdc_inform )
!!$    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )
!!$    CALL copy_fit_inform_out( finform%fit_inform, cinform%fit_inform )
!!$    CALL copy_roots_inform_out( finform%roots_inform, cinform%roots_inform )
!!$    CALL copy_cro_inform_out( finform%cro_inform, cinform%cro_inform )
!!$    CALL copy_rpd_inform_out( finform%rpd_inform, cinform%rpd_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( LEN( finform%bad_alloc ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_CQP_double_ciface

!  -------------------------------------
!  C interface to fortran cqp_initialize
!  -------------------------------------

  SUBROUTINE cqp_initialize( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_CQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( cqp_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( cqp_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_cqp_full_data_type ), POINTER :: fdata
  TYPE ( f_cqp_control_type ) :: fcontrol
  TYPE ( f_cqp_inform_type ) :: finform
  LOGICAL :: f_indexing 

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_cqp_initialize( fdata, fcontrol, finform )

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE cqp_initialize

!  ----------------------------------------
!  C interface to fortran cqp_read_specfile
!  ----------------------------------------

  SUBROUTINE cqp_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_CQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( cqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_cqp_control_type ) :: fcontrol
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

  CALL f_cqp_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE cqp_read_specfile

!  ---------------------------------
!  C interface to fortran cqp_inport
!  ---------------------------------

  SUBROUTINE cqp_import( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_CQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( cqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_cqp_control_type ) :: fcontrol
  TYPE ( f_cqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN

!  import the problem data into the required CQP structure

    CALL f_cqp_import( fcontrol, fdata, status )
  ELSE
    CALL f_cqp_import( fcontrol, fdata, status )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE cqp_import

!  ----------------------------------------
!  C interface to fortran cqp_reset_control
!  ----------------------------------------

  SUBROUTINE cqp_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_CQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( cqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_cqp_control_type ) :: fcontrol
  TYPE ( f_cqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_cqp_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE cqp_reset_control

!  --------------------------------------
!  C interface to fortran cqp_information
!  --------------------------------------

  SUBROUTINE cqp_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_CQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( cqp_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_cqp_full_data_type ), pointer :: fdata
  TYPE ( f_cqp_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain CQP solution information

  CALL f_cqp_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE cqp_information

!  ------------------------------------
!  C interface to fortran cqp_terminate
!  ------------------------------------

  SUBROUTINE cqp_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_CQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( cqp_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( cqp_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_cqp_full_data_type ), pointer :: fdata
  TYPE ( f_cqp_control_type ) :: fcontrol
  TYPE ( f_cqp_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_cqp_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE cqp_terminate
