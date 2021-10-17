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
        f_nls_subproblem_control_type   => NLS_subproblem_control_type,        &
        f_nls_control_type              => NLS_control_type,                   &
        f_nls_time_type                 => NLS_time_type,                      &
        f_nls_subproblem_inform_type    => NLS_subproblem_inform_type,         &
        f_nls_inform_type               => NLS_inform_type,                    &
        f_nls_full_data_type            => NLS_full_data_type,                 &
        f_nls_initialize                => NLS_initialize,                     &
        f_nls_read_specfile             => NLS_read_specfile,                  &
        f_nls_import                    => NLS_import,                         &
        f_nls_reset_control             => NLS_reset_control,                  &
        f_nls_solve_with_mat            => NLS_solve_with_mat,                 &
        f_nls_solve_without_mat         => NLS_solve_without_mat,              &
        f_nls_solve_reverse_with_mat    => NLS_solve_reverse_with_mat,         &
        f_nls_solve_reverse_without_mat => NLS_solve_reverse_without_mat,      &
        f_nls_information               => NLS_information,                    &
        f_nls_terminate                 => NLS_terminate
    USE GALAHAD_USERDATA_double, only:                                         &
        f_galahad_userdata_type => GALAHAD_userdata_type

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

    TYPE, BIND( C ) :: nls_control_type
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
      INTEGER ( KIND = C_LONG ) :: max_entries_factors
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

    TYPE, BIND( C ) :: nls_inform_type
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
      TYPE ( nls_subproblem_inform_type ) :: subproblem_inform
    END TYPE nls_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_C( n, m, x, c, userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m
        REAL ( KIND = wp ), DIMENSION( n ),INTENT( IN ) :: x
        REAL ( KIND = wp ), DIMENSION( m ),INTENT( OUT ) :: c
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_C
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_J( n, m, jne, x, jval, userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, jne
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = wp ), DIMENSION( jne ),INTENT( OUT ) :: jval
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_J
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_H( n, m, hne, x, y, hval,                                  &
                       userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, hne
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = wp ), DIMENSION( m ), INTENT( IN ) :: y
        REAL ( KIND = wp ), DIMENSION( hne ), INTENT( OUT ) :: hval
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_H
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_JPROD( n, m, x, transpose, u, v, got_j,                    &
                           userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ), VALUE :: transpose
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = wp ), DIMENSION( MAX( n, m ) ), INTENT( INOUT ) :: u
        REAL ( KIND = wp ), DIMENSION( MAX( n, m ) ), INTENT( IN ) :: v
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ) :: got_j
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_JPROD
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_HPROD( n, m, x, y, u, v, got_h,                            &
                           userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = wp ), DIMENSION( m ), INTENT( IN ) :: y
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( INOUT ) :: u
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: v
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ) :: got_h
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_HPROD
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_HPRODS( n, m, pne, x, v, pval, got_h,                      &
                            userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, pne
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = wp ), DIMENSION( m ), INTENT( IN ) :: v
        REAL ( KIND = wp ), DIMENSION( pne ), INTENT( INOUT ) :: pval
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ) :: got_h
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_HPRODS
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_SCALE( n, m, x, u, v,                                      &
                           userdata ) RESULT( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( IN ) :: x, v
        REAL ( KIND = wp ), DIMENSION( n ), INTENT( OUT ) :: u
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_SCALE
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( nls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_nls_control_type ), INTENT( OUT ) :: fcontrol
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
!!$    CALL copy_roots_control_in( ccontrol%roots_control,                     &
!!$                               fcontrol%roots_control )

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%alive_file( i : i ) = ccontrol%alive_file( i )
    END DO
    DO i = 1, 31
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO

    ! subproblem_control parameters

    ! Integers
    fcontrol%subproblem_control%error = &
      ccontrol%subproblem_control%error
    fcontrol%subproblem_control%out = &
      ccontrol%subproblem_control%out
    fcontrol%subproblem_control%print_level = &
      ccontrol%subproblem_control%print_level
    fcontrol%subproblem_control%start_print = &
      ccontrol%subproblem_control%start_print
    fcontrol%subproblem_control%stop_print = &
      ccontrol%subproblem_control%stop_print
    fcontrol%subproblem_control%print_gap = &
      ccontrol%subproblem_control%print_gap
    fcontrol%subproblem_control%maxit = &
      ccontrol%subproblem_control%maxit
    fcontrol%subproblem_control%alive_unit = &
      ccontrol%subproblem_control%alive_unit
    fcontrol%subproblem_control%jacobian_available = &
      ccontrol%subproblem_control%jacobian_available
    fcontrol%subproblem_control%hessian_available = &
      ccontrol%subproblem_control%hessian_available
    fcontrol%subproblem_control%model = &
      ccontrol%subproblem_control%model
    fcontrol%subproblem_control%norm = &
      ccontrol%subproblem_control%norm
    fcontrol%subproblem_control%non_monotone = &
      ccontrol%subproblem_control%non_monotone
    fcontrol%subproblem_control%weight_update_strategy = &
      ccontrol%subproblem_control%weight_update_strategy

    ! Reals
    fcontrol%subproblem_control%stop_c_absolute = &
      ccontrol%subproblem_control%stop_c_absolute
    fcontrol%subproblem_control%stop_c_relative = &
      ccontrol%subproblem_control%stop_c_relative
    fcontrol%subproblem_control%stop_g_absolute = &
      ccontrol%subproblem_control%stop_g_absolute
    fcontrol%subproblem_control%stop_g_relative = &
      ccontrol%subproblem_control%stop_g_relative
    fcontrol%subproblem_control%stop_s = &
      ccontrol%subproblem_control%stop_s
    fcontrol%subproblem_control%power = &
      ccontrol%subproblem_control%power
    fcontrol%subproblem_control%initial_weight = &
      ccontrol%subproblem_control%initial_weight
    fcontrol%subproblem_control%minimum_weight = &
      ccontrol%subproblem_control%minimum_weight
    fcontrol%subproblem_control%initial_inner_weight = &
      ccontrol%subproblem_control%initial_inner_weight
    fcontrol%subproblem_control%eta_successful = &
      ccontrol%subproblem_control%eta_successful
    fcontrol%subproblem_control%eta_very_successful = &
      ccontrol%subproblem_control%eta_very_successful
    fcontrol%subproblem_control%eta_too_successful = &
      ccontrol%subproblem_control%eta_too_successful
    fcontrol%subproblem_control%weight_decrease_min = &
      ccontrol%subproblem_control%weight_decrease_min
    fcontrol%subproblem_control%weight_decrease = &
      ccontrol%subproblem_control%weight_decrease
    fcontrol%subproblem_control%weight_increase = &
      ccontrol%subproblem_control%weight_increase
    fcontrol%subproblem_control%weight_increase_max = &
      ccontrol%subproblem_control%weight_increase_max
    fcontrol%subproblem_control%reduce_gap = &
      ccontrol%subproblem_control%reduce_gap
    fcontrol%subproblem_control%tiny_gap = &
      ccontrol%subproblem_control%tiny_gap
    fcontrol%subproblem_control%large_root = &
      ccontrol%subproblem_control%large_root
    fcontrol%subproblem_control%switch_to_newton = &
      ccontrol%subproblem_control%switch_to_newton
    fcontrol%subproblem_control%cpu_time_limit = &
      ccontrol%subproblem_control%cpu_time_limit
    fcontrol%subproblem_control%clock_time_limit = &
      ccontrol%subproblem_control%clock_time_limit

    ! Logicals
    fcontrol%subproblem_control%subproblem_direct = &
      ccontrol%subproblem_control%subproblem_direct
    fcontrol%subproblem_control%renormalize_weight = &
      ccontrol%subproblem_control%renormalize_weight
    fcontrol%subproblem_control%magic_step = &
      ccontrol%subproblem_control%magic_step
    fcontrol%subproblem_control%print_obj = &
      ccontrol%subproblem_control%print_obj
    fcontrol%subproblem_control%space_critical = &
      ccontrol%subproblem_control%space_critical
    fcontrol%subproblem_control%deallocate_error_fatal = &
      ccontrol%subproblem_control%deallocate_error_fatal

    ! Derived types
!!$    CALL copy_rqs_control_in( ccontrol%subproblem_control%rqs_control,      &
!!&                              fcontrol%subproblem_control%rqs_control )
!!$    CALL copy_glrt_control_in( ccontrol%subproblem_control%glrt_control,    &
!!&                              fcontrol%subproblem_control%glrt_control )
!!$    CALL copy_psls_control_in( ccontrol%subproblem_control%psls_control,    &
!!&                              fcontrol%subproblem_control%psls_control )
!!$    CALL copy_bsc_control_in( ccontrol%subproblem_control%bsc_control,      &
!!&                              fcontrol%subproblem_control%bsc_control )
!!$    CALL copy_roots_control_in( ccontrol%subproblem_control%roots_control,  &
!!&                                fcontrol%subproblem_control%roots_control )

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%subproblem_control%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%subproblem_control%alive_file( i : i ) =                        &
        ccontrol%subproblem_control%alive_file( i )
    END DO
    DO i = 1, 31
      IF ( ccontrol%subproblem_control%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%subproblem_control%prefix( i : i ) =                            &
        ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_nls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( nls_control_type ), INTENT( OUT ) :: ccontrol
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
!!$    CALL copy_rqs_control_out( fcontrol%rqs_control,                        &
!!&                                ccontrol%rqs_control )
!!$    CALL copy_glrt_control_out( fcontrol%glrt_control,                      &
!!&                                ccontrol%glrt_control )
!!$    CALL copy_psls_control_out( fcontrol%psls_control,                      &
!!&                                ccontrol%psls_control )
!!$    CALL copy_bsc_control_out( fcontrol%bsc_control,                        &
!!&                               ccontrol%bsc_control )
!!$    CALL copy_roots_control_out( fcontrol%roots_control,                    &
!!&                                 ccontrol%roots_control )

    ! Strings
    DO i = 1, LEN( fcontrol%alive_file )
      ccontrol%alive_file( i ) = fcontrol%alive_file( i : i )
    END DO
    ccontrol%alive_file( LEN( fcontrol%alive_file ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR

    ! subproblem_control parameters

    ! Integers
    ccontrol%subproblem_control%error = &
      fcontrol%subproblem_control%error
    ccontrol%subproblem_control%out = &
      fcontrol%subproblem_control%out
    ccontrol%subproblem_control%print_level = &
      fcontrol%subproblem_control%print_level
    ccontrol%subproblem_control%start_print = &
      fcontrol%subproblem_control%start_print
    ccontrol%subproblem_control%stop_print = &
      fcontrol%subproblem_control%stop_print
    ccontrol%subproblem_control%print_gap = &
      fcontrol%subproblem_control%print_gap
    ccontrol%subproblem_control%maxit = &
      fcontrol%subproblem_control%maxit
    ccontrol%subproblem_control%alive_unit = &
      fcontrol%subproblem_control%alive_unit
    ccontrol%subproblem_control%jacobian_available = &
      fcontrol%subproblem_control%jacobian_available
    ccontrol%subproblem_control%hessian_available = &
      fcontrol%subproblem_control%hessian_available
    ccontrol%subproblem_control%model = &
      fcontrol%subproblem_control%model
    ccontrol%subproblem_control%norm = &
      fcontrol%subproblem_control%norm
    ccontrol%subproblem_control%non_monotone = &
      fcontrol%subproblem_control%non_monotone
    ccontrol%subproblem_control%weight_update_strategy = &
      fcontrol%subproblem_control%weight_update_strategy

    ! Reals
    ccontrol%subproblem_control%stop_c_absolute = &
      fcontrol%subproblem_control%stop_c_absolute
    ccontrol%subproblem_control%stop_c_relative = &
      fcontrol%subproblem_control%stop_c_relative
    ccontrol%subproblem_control%stop_g_absolute = &
      fcontrol%subproblem_control%stop_g_absolute
    ccontrol%subproblem_control%stop_g_relative = &
      fcontrol%subproblem_control%stop_g_relative
    ccontrol%subproblem_control%stop_s = &
      fcontrol%subproblem_control%stop_s
    ccontrol%subproblem_control%power = &
      fcontrol%subproblem_control%power
    ccontrol%subproblem_control%initial_weight = &
      fcontrol%subproblem_control%initial_weight
    ccontrol%subproblem_control%minimum_weight = &
      fcontrol%subproblem_control%minimum_weight
    ccontrol%subproblem_control%initial_inner_weight = &
      fcontrol%subproblem_control%initial_inner_weight
    ccontrol%subproblem_control%eta_successful = &
      fcontrol%subproblem_control%eta_successful
    ccontrol%subproblem_control%eta_very_successful = &
      fcontrol%subproblem_control%eta_very_successful
    ccontrol%subproblem_control%eta_too_successful = &
      fcontrol%subproblem_control%eta_too_successful
    ccontrol%subproblem_control%weight_decrease_min = &
      fcontrol%subproblem_control%weight_decrease_min
    ccontrol%subproblem_control%weight_decrease = &
      fcontrol%subproblem_control%weight_decrease
    ccontrol%subproblem_control%weight_increase = &
      fcontrol%subproblem_control%weight_increase
    ccontrol%subproblem_control%weight_increase_max = &
      fcontrol%subproblem_control%weight_increase_max
    ccontrol%subproblem_control%reduce_gap = &
      fcontrol%subproblem_control%reduce_gap
    ccontrol%subproblem_control%tiny_gap = &
      fcontrol%subproblem_control%tiny_gap
    ccontrol%subproblem_control%large_root = &
      fcontrol%subproblem_control%large_root
    ccontrol%subproblem_control%switch_to_newton = &
      fcontrol%subproblem_control%switch_to_newton
    ccontrol%subproblem_control%cpu_time_limit = &
      fcontrol%subproblem_control%cpu_time_limit
    ccontrol%subproblem_control%clock_time_limit = &
      fcontrol%subproblem_control%clock_time_limit

    ! Logicals
    ccontrol%subproblem_control%subproblem_direct = &
      fcontrol%subproblem_control%subproblem_direct
    ccontrol%subproblem_control%renormalize_weight = &
      fcontrol%subproblem_control%renormalize_weight
    ccontrol%subproblem_control%magic_step = &
      fcontrol%subproblem_control%magic_step
    ccontrol%subproblem_control%print_obj = &
      fcontrol%subproblem_control%print_obj
    ccontrol%subproblem_control%space_critical = &
      fcontrol%subproblem_control%space_critical
    ccontrol%subproblem_control%deallocate_error_fatal = &
      fcontrol%subproblem_control%deallocate_error_fatal

    ! Derived types
!!$    CALL copy_rqs_control_out( fcontrol%subproblem_control%rqs_controll,    &
!!&                               ccontrol%subproblem_control%rqs_control )
!!$    CALL copy_glrt_control_out( fcontrol%subproblem_control,                &
!!&                                ccontrol%subproblem_control%glrt_control )
!!$    CALL copy_psls_control_out( fcontrol%subproblem_control%psls_control,   &
!!&                                ccontrol%subproblem_control%psls_control )
!!$    CALL copy_bsc_control_out( fcontrol%subproblem_control%bsc_control,     &
!!&                               ccontrol%subproblem_control%bsc_control )
!!$    CALL copy_roots_control_out( fcontrol%subproblem_control%roots_control, &
!!&                                 ccontrol%subproblem_control%roots_control )
    ! Strings
    DO i = 1, LEN( fcontrol%subproblem_control%alive_file )
      ccontrol%subproblem_control%alive_file( i ) =                            &
        fcontrol%subproblem_control%alive_file( i : i )
    END DO
    ccontrol%subproblem_control%alive_file(                                    &
      LEN( fcontrol%subproblem_control%alive_file ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( fcontrol%subproblem_control%prefix )
      ccontrol%subproblem_control%prefix( i ) =                                &
        fcontrol%subproblem_control%prefix( i : i )
    END DO
    ccontrol%subproblem_control%prefix(                                        &
      LEN( fcontrol%subproblem_control%prefix ) + 1 ) = C_NULL_CHAR
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

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( nls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_nls_inform_type ), INTENT( OUT ) :: finform
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

    ! subproblem_inform parameters

    ! Integers
    finform%subproblem_inform%status = &
      cinform%subproblem_inform%status
    finform%subproblem_inform%alloc_status = &
      cinform%subproblem_inform%alloc_status
    finform%subproblem_inform%iter = &
      cinform%subproblem_inform%iter
    finform%subproblem_inform%cg_iter = &
      cinform%subproblem_inform%cg_iter
    finform%subproblem_inform%c_eval = &
      cinform%subproblem_inform%c_eval
    finform%subproblem_inform%j_eval = &
      cinform%subproblem_inform%j_eval
    finform%subproblem_inform%h_eval = &
      cinform%subproblem_inform%h_eval
    finform%subproblem_inform%factorization_max = &
      cinform%subproblem_inform%factorization_max
    finform%subproblem_inform%factorization_status = &
      cinform%subproblem_inform%factorization_status
    finform%subproblem_inform%max_entries_factors = &
      cinform%subproblem_inform%max_entries_factors
    finform%subproblem_inform%factorization_integer = &
      cinform%subproblem_inform%factorization_integer
    finform%subproblem_inform%factorization_real = &
      cinform%subproblem_inform%factorization_real

    ! Reals
    finform%subproblem_inform%factorization_average = &
      cinform%subproblem_inform%factorization_average
    finform%subproblem_inform%obj = &
      cinform%subproblem_inform%obj
    finform%subproblem_inform%norm_c = &
      cinform%subproblem_inform%norm_c
    finform%subproblem_inform%norm_g = &
      cinform%subproblem_inform%norm_g
    finform%subproblem_inform%weight = &
      cinform%subproblem_inform%weight

    ! Derived types
    CALL copy_time_in( cinform%subproblem_inform%time,                         &
                       finform%subproblem_inform%time )
!!$    CALL copy_rqs_inform_in( cinform%subproblem_inform%rqs_inform,          &
!!&                             finform%subproblem_inform%rqs_inform )
!!$    CALL copy_glrt_inform_in( cinform%subproblem_inform%glrt_inform,        &
!!&                              finform%subproblem_inform%glrt_inform )
!!$    CALL copy_psls_inform_in( cinform%subproblem_inform%psls_inform,        &
!!&                              finform%subproblem_inform%psls_inform )
!!$    CALL copy_bsc_inform_in( cinform%subproblem_inform%bsc_inform,          &
!!&                             finform%subproblem_inform%bsc_inform )
!!$    CALL copy_roots_inform_in( cinform%subproblem_inform%roots_inform,      &
!!&                               finform%subproblem_inform%roots_inform )

    ! Strings
    DO i = 1, 81
      IF ( cinform%subproblem_inform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%subproblem_inform%bad_alloc( i : i ) =                           &
        cinform%subproblem_inform%bad_alloc( i )
    END DO
    DO i = 1, 13
      IF ( cinform%subproblem_inform%bad_eval( i ) == C_NULL_CHAR ) EXIT
      finform%subproblem_inform%bad_eval( i : i ) =                            &
        cinform%subproblem_inform%bad_eval( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_nls_inform_type ), INTENT( IN ) :: finform
    TYPE ( nls_inform_type ), INTENT( OUT ) :: cinform
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

    ! subproblem_inform parameters

    ! Integers
    cinform%subproblem_inform%status = &
      finform%subproblem_inform%status
    cinform%subproblem_inform%alloc_status = &
      finform%subproblem_inform%alloc_status
    cinform%subproblem_inform%iter = &
      finform%subproblem_inform%iter
    cinform%subproblem_inform%cg_iter = &
      finform%subproblem_inform%cg_iter
    cinform%subproblem_inform%c_eval = &
      finform%subproblem_inform%c_eval
    cinform%subproblem_inform%j_eval = &
      finform%subproblem_inform%j_eval
    cinform%subproblem_inform%h_eval = &
      finform%subproblem_inform%h_eval
    cinform%subproblem_inform%factorization_max = &
      finform%subproblem_inform%factorization_max
    cinform%subproblem_inform%factorization_status = &
      finform%subproblem_inform%factorization_status
    cinform%subproblem_inform%max_entries_factors = &
      finform%subproblem_inform%max_entries_factors
    cinform%subproblem_inform%factorization_integer = &
      finform%subproblem_inform%factorization_integer
    cinform%subproblem_inform%factorization_real = &
      finform%subproblem_inform%factorization_real

    ! Reals
    cinform%subproblem_inform%factorization_average = &
      finform%subproblem_inform%factorization_average
    cinform%subproblem_inform%obj = &
      finform%subproblem_inform%obj
    cinform%subproblem_inform%norm_c = &
      finform%subproblem_inform%norm_c
    cinform%subproblem_inform%norm_g = &
      finform%subproblem_inform%norm_g
    cinform%subproblem_inform%weight = &
      finform%subproblem_inform%weight

    ! Derived types
    CALL copy_time_out( finform%subproblem_inform%time,                        &
                        cinform%subproblem_inform%time )
!!$    CALL copy_rqs_inform_out( finform%subproblem_inform%rqs_inform,         &
!!$                              cinform%subproblem_inform%rqs_inform )
!!$    CALL copy_glrt_inform_out( finform%subproblem_inform%glrt_inform,       &
!!$                               cinform%subproblem_inform%glrt_inform )
!!$    CALL copy_psls_inform_out( finform%subproblem_inform%psls_inform,       &
!!$                               cinform%subproblem_inform%psls_inform )
!!$    CALL copy_bsc_inform_out( finform%subproblem_inform%bsc_inform,         &
!!$                               cinform%subproblem_inform%bsc_inform )
!!$    CALL copy_roots_inform_out( finform%subproblem_inform%roots_inform,     &
!!$                                cinform%subproblem_inform%roots_inform )

    ! Strings
    DO i = 1, LEN( finform%subproblem_inform%bad_alloc )
      cinform%subproblem_inform%bad_alloc( i ) =                               &
        finform%subproblem_inform%bad_alloc( i : i )
    END DO
    cinform%subproblem_inform%bad_alloc(                                       &
      LEN( finform%subproblem_inform%bad_alloc ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( finform%subproblem_inform%bad_eval )
      cinform%subproblem_inform%bad_eval( i ) =                                &
        finform%subproblem_inform%bad_eval( i : i )
    END DO
    cinform%subproblem_inform%bad_eval(                                        &
      LEN( finform%subproblem_inform%bad_eval ) + 1 ) = C_NULL_CHAR
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

  SUBROUTINE nls_import( ccontrol, cdata, status, n, m,                        &
                         cjtype, jne, jrow, jcol, jptr,                        &
                         chtype, hne, hrow, hcol, hptr,                        &
                         cptype, pne, prow, pcol, pptr, w ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( nls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, jne, hne, pne
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: jrow
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: jcol
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: jptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cjtype
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: hrow
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: hcol
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: prow
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: pcol
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: pptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cptype
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ), OPTIONAL :: w

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cjtype ) ) :: fjtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cptype ) ) :: fptype
  TYPE ( f_nls_control_type ) :: fcontrol
  TYPE ( f_nls_full_data_type ), POINTER :: fdata
  INTEGER, DIMENSION( : ), ALLOCATABLE :: jrow_find, jcol_find, jptr_find
  INTEGER, DIMENSION( : ), ALLOCATABLE :: hrow_find, hcol_find, hptr_find
  INTEGER, DIMENSION( : ), ALLOCATABLE :: prow_find, pcol_find, pptr_find
  LOGICAL :: f_indexing

! IF ( PRESENT( w ) ) WRITE( 6, * ) ' w ', w

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fjtype = cstr_to_fchar( cjtype )
  fhtype = cstr_to_fchar( chtype )
  fptype = cstr_to_fchar( cptype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN
    IF ( PRESENT( jrow ) ) THEN
      ALLOCATE( jrow_find( jne ) )
      jrow_find = jrow + 1
    END IF
    IF ( PRESENT( jcol ) ) THEN
      ALLOCATE( jcol_find( jne ) )
      jcol_find = jcol + 1
    END IF
    IF ( PRESENT( jptr ) ) THEN
      ALLOCATE( jptr_find( m + 1 ) )
      jptr_find = jptr + 1
    END IF

    IF ( PRESENT( hrow ) ) THEN
      ALLOCATE( hrow_find( hne ) )
      hrow_find = hrow + 1
    END IF
    IF ( PRESENT( hcol ) ) THEN
      ALLOCATE( hcol_find( hne ) )
      hcol_find = hcol + 1
    END IF
    IF ( PRESENT( hptr ) ) THEN
      ALLOCATE( hptr_find( n + 1 ) )
      hptr_find = hptr + 1
    END IF

    IF ( PRESENT( prow ) ) THEN
      ALLOCATE( prow_find( pne ) )
      prow_find = prow + 1
    END IF
    IF ( PRESENT( pcol ) ) THEN
      ALLOCATE( pcol_find( pne ) )
      pcol_find = pcol + 1
    END IF
    IF ( PRESENT( pptr ) ) THEN
      ALLOCATE( pptr_find( m + 1 ) )
      pptr_find = pptr + 1
    END IF

!  import the problem data into the required NLS structure

    CALL f_nls_import( fcontrol, fdata, status, n, m,                          &
                       fjtype, jne, jrow_find, jcol_find, jptr_find,           &
                       fhtype, hne, hrow_find, hcol_find, hptr_find,           &
                       fptype, pne, prow_find, pcol_find, pptr_find, w )
  ELSE
    CALL f_nls_import( fcontrol, fdata, status, n, m,                          &
                       fjtype, jne, jrow, jcol, jptr,                          &
                       fhtype, hne, hrow, hcol, hptr,                          &
                       fptype, pne, prow, pcol, pptr, w )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE nls_import

!  ----------------------------------------
!  C interface to fortran nls_reset_control
!  ----------------------------------------

  SUBROUTINE nls_reset_control( ccontrol, cdata, status ) BIND( C )
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

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_nls_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE nls_reset_control

!  -----------------------------------------
!  C interface to fortran nls_solve_with_mat
!  -----------------------------------------

  SUBROUTINE nls_solve_with_mat( cdata, cuserdata, status, n, m, x, c, g,      &
                                 ceval_c, jne, ceval_j, hne, ceval_h,          &
                                 pne, ceval_hprods ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, jne, hne, pne
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: c
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: g 
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_c, ceval_j
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_h, ceval_hprods

!  local variables

  TYPE ( f_nls_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_c ), POINTER :: feval_c
  PROCEDURE( eval_j ), POINTER :: feval_j
  PROCEDURE( eval_h ), POINTER :: feval_h
  PROCEDURE( eval_hprods ), POINTER :: feval_hprods

!  ignore Fortran userdata type (not interoperable)

  TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_c, feval_c )
  CALL C_F_PROCPOINTER( ceval_j, feval_j )
  IF ( C_ASSOCIATED( ceval_h ) ) THEN 
    CALL C_F_PROCPOINTER( ceval_h, feval_h )
  ELSE
    NULLIFY( feval_h )
  END IF
  IF ( C_ASSOCIATED( ceval_hprods ) ) THEN 
    CALL C_F_PROCPOINTER( ceval_hprods, feval_hprods )
  ELSE
    NULLIFY( feval_hprods )
  END IF

!  solve the problem when the Hessian is explicitly available

  CALL f_nls_solve_with_mat( fdata, fuserdata, status, x, c, g, wrap_eval_c,   &
                              wrap_eval_j, wrap_eval_h, wrap_eval_hprods )

  RETURN

!  wrappers

  CONTAINS

!  eval_c wrapper

    SUBROUTINE wrap_eval_c( status, x, userdata, c )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: c

!  call C interoperable eval_c

!   write(6, "( ' X in wrap_eval_c = ', 2ES12.4 )" ) x
    status = feval_c( n, m, x, c, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_c

!  eval_j wrapper

    SUBROUTINE wrap_eval_j( status, x, userdata, jval )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: jval

!  Call C interoperable eval_j
    status = feval_j( n, m, jne, x, jval, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_j

!  eval_H wrapper

    SUBROUTINE wrap_eval_h( status, x, y, userdata, hval )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x, y
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: hval

!  Call C interoperable eval_h
    status = feval_h( n, m, hne, x, y, hval, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_h

!  eval_hprods wrapper

    SUBROUTINE wrap_eval_hprods( status, x, v, userdata, pval, fgot_h )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x, v
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: pval
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

!  call C interoperable eval_hprods

    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .FALSE.
    END IF

    status = feval_hprods( n, m, pne, x, v, pval, cgot_h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hprods

  END SUBROUTINE nls_solve_with_mat

!  --------------------------------------------
!  C interface to fortran nls_solve_without_mat
!  --------------------------------------------

  SUBROUTINE nls_solve_without_mat( cdata, cuserdata, status, n, m, x, c, g,   &
                                    ceval_c, ceval_jprod, ceval_hprod,         &
                                    pne, ceval_hprods ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, pne
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: c
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: g 
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_c, ceval_jprod
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_hprod, ceval_hprods

!  local variables

  TYPE ( f_nls_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_c ), POINTER :: feval_c
  PROCEDURE( eval_jprod ), POINTER :: feval_jprod
  PROCEDURE( eval_hprod ), POINTER :: feval_hprod
  PROCEDURE( eval_hprods ), POINTER :: feval_hprods

!  ignore Fortran userdata type (not interoperable)

  TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )

!  associate data pointer

  CALL C_F_POINTER(cdata, fdata)

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_c, feval_c ) 
  CALL C_F_PROCPOINTER( ceval_jprod, feval_jprod )
  IF ( C_ASSOCIATED( ceval_hprod ) ) THEN 
    CALL C_F_PROCPOINTER( ceval_hprod, feval_hprod )
  ELSE
    NULLIFY( feval_hprod )
  END IF
  IF ( C_ASSOCIATED( ceval_hprods ) ) THEN 
    CALL C_F_PROCPOINTER( ceval_hprods, feval_hprods )
  ELSE
    NULLIFY( feval_hprods )
  END IF

!  solve the problem when the Hessian is only available via products

  CALL f_nls_solve_without_mat( fdata, fuserdata, status, x, c, g,             &
                                wrap_eval_c, wrap_eval_jprod,                  &
                                wrap_eval_hprod, wrap_eval_hprods )
  RETURN

!  wrappers

  CONTAINS

!  eval_c wrapper

    SUBROUTINE wrap_eval_c( status, x, userdata, c )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: c

!  call C interoperable eval_c

    status = feval_c( n, m, x, c, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_c

!  eval_jprod wrapper

    SUBROUTINE wrap_eval_jprod( status, x, userdata, ftranspose, u, v, fgot_j )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: u
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: v
    LOGICAL, INTENT( IN ) :: ftranspose
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_j
    LOGICAL ( KIND = C_BOOL ) :: cgot_j, ctranspose

!  call C interoperable eval_jprod

    ctranspose = ftranspose
!   IF ( ftranspose ) THEN
!     t = 1
!     ctranspose = .TRUE.
!   ELSE
!     t = 0
!     ctranspose = .FALSE.
!   END IF
!   write(6,*) 'ctranspose, ftranspose', ctranspose, ftranspose
    IF ( PRESENT( fgot_j ) ) THEN
      cgot_j = fgot_j
    ELSE
      cgot_j = .FALSE.
    END IF
    status = feval_jprod( n, m, x, ctranspose, u, v, cgot_j, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_jprod

!  eval_hprod wrapper

    SUBROUTINE wrap_eval_hprod( status, x, y, userdata, u, v, fgot_h )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x, y
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: u
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: v
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

!  call C interoperable eval_hprod

    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .FALSE.
    END IF
    status = feval_hprod( n, m, x, y, u, v, cgot_h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hprod

!  eval_hprods wrapper

    SUBROUTINE wrap_eval_hprods( status, x, v, userdata, pval, fgot_h )
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: x, v
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: pval
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

!  call C interoperable eval_hprods

    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .FALSE.
    END IF

    status = feval_hprods( n, m, pne, x, v, pval, cgot_h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hprods

  END SUBROUTINE nls_solve_without_mat

!  -------------------------------------------------
!  C interface to fortran nls_solve_reverse_with_mat
!  -------------------------------------------------

  SUBROUTINE nls_solve_reverse_with_mat( cdata, status, eval_status,           &
                                         n, m, x, c, g, jne, jval, y,          &
                                         hne, hval, v, pne, pval ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, jne, hne, pne
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status, eval_status
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x, g 
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: c
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( jne ) :: jval
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( hne ) :: hval
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( pne ) :: pval
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: y
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: v
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_nls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when the Hessian is available by reverse communication

  CALL f_nls_solve_reverse_with_mat( fdata, status, eval_status, x, c, g,      &
                                     jval, y, hval, v, pval )
  RETURN
    
  END SUBROUTINE nls_solve_reverse_with_mat

!  ----------------------------------------------------
!  C interface to fortran nls_solve_reverse_without_mat
!  ----------------------------------------------------

  SUBROUTINE nls_solve_reverse_without_mat( cdata, status, eval_status,        &
                                            n, m, x, c, g, ctranspose, u, v,   &
                                            y, pne, pval ) BIND( C )
  USE GALAHAD_NLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, pne
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status, eval_status
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x, g
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: c, y
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( MAX( n, m ) ) :: u, v
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( pne ) :: pval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  LOGICAL ( KIND = C_BOOL ), INTENT( INOUT )  :: ctranspose
  
!  local variables

  TYPE ( f_nls_full_data_type ), POINTER :: fdata
  LOGICAL :: ftranspose

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when Hessian products are available by reverse 
!  communication

  CALL f_nls_solve_reverse_without_mat( fdata, status, eval_status, x, c, g,  &
                                        ftranspose, u, v, y, pval )
  IF ( status == 5 ) ctranspose = ftranspose
  RETURN

  END SUBROUTINE nls_solve_reverse_without_mat

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
