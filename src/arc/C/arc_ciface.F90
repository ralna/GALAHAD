! THIS VERSION: GALAHAD 5.2 - 2025-01-12 AT 09:20 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ A R C   C   I N T E R F A C E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. July 28th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module to GALAHAD_ARC types and interfaces

  MODULE GALAHAD_ARC_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_ARC_precision, ONLY:                                           &
        f_arc_time_type                 => ARC_time_type,                      &
        f_arc_inform_type               => ARC_inform_type,                    &
        f_arc_control_type              => ARC_control_type,                   &
        f_arc_full_data_type            => ARC_full_data_type,                 &
        f_arc_initialize                => ARC_initialize,                     &
        f_arc_read_specfile             => ARC_read_specfile,                  &
        f_arc_import                    => ARC_import,                         &
        f_arc_reset_control             => ARC_reset_control,                  &
        f_arc_solve_with_mat            => ARC_solve_with_mat,                 &
        f_arc_solve_without_mat         => ARC_solve_without_mat,              &
        f_arc_solve_reverse_with_mat    => ARC_solve_reverse_with_mat,         &
        f_arc_solve_reverse_without_mat => ARC_solve_reverse_without_mat,      &
        f_arc_information               => ARC_information,                    &
        f_arc_terminate                 => ARC_terminate

    USE GALAHAD_USERDATA_precision, ONLY:                                      &
        f_galahad_userdata_type => GALAHAD_userdata_type

    USE GALAHAD_RQS_precision_ciface, ONLY:                                    &
        rqs_inform_type,                                                       &
        rqs_control_type,                                                      &
        copy_rqs_inform_in   => copy_inform_in,                                &
        copy_rqs_inform_out  => copy_inform_out,                               &
        copy_rqs_control_in  => copy_control_in,                               &
        copy_rqs_control_out => copy_control_out

    USE GALAHAD_GLRT_precision_ciface, ONLY:                                   &
        glrt_inform_type,                                                      &
        glrt_control_type,                                                     &
        copy_glrt_inform_in   => copy_inform_in,                               &
        copy_glrt_inform_out  => copy_inform_out,                              &
        copy_glrt_control_in  => copy_control_in,                              &
        copy_glrt_control_out => copy_control_out

    USE GALAHAD_PSLS_precision_ciface, ONLY:                                   &
        psls_inform_type,                                                      &
        psls_control_type,                                                     &
        copy_psls_inform_in   => copy_inform_in,                               &
        copy_psls_inform_out  => copy_inform_out,                              &
        copy_psls_control_in  => copy_control_in,                              &
        copy_psls_control_out => copy_control_out

     USE GALAHAD_DPS_precision_ciface, ONLY:                                   &
         dps_inform_type,                                                      &
         dps_control_type,                                                     &
         copy_dps_inform_in   => copy_inform_in,                               &
         copy_dps_inform_out  => copy_inform_out,                              &
         copy_dps_control_in  => copy_control_in,                              &
         copy_dps_control_out => copy_control_out

    USE GALAHAD_LMS_precision_ciface, ONLY:                                    &
        lms_inform_type,                                                       &
        lms_control_type,                                                      &
        copy_lms_inform_in   => copy_inform_in,                                &
        copy_lms_inform_out  => copy_inform_out,                               &
        copy_lms_control_in  => copy_control_in,                               &
        copy_lms_control_out => copy_control_out

    USE GALAHAD_SHA_precision_ciface, ONLY:                                    &
        sha_inform_type,                                                       &
        sha_control_type,                                                      &
        copy_sha_inform_in   => copy_inform_in,                                &
        copy_sha_inform_out  => copy_inform_out,                               &
        copy_sha_control_in  => copy_control_in,                               &
        copy_sha_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: arc_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: alive_unit
      CHARACTER( KIND = C_CHAR ), DIMENSION( 31 ) :: alive_file
      INTEGER ( KIND = ipc_ ) :: non_monotone
      INTEGER ( KIND = ipc_ ) :: model
      INTEGER ( KIND = ipc_ ) :: norm
      INTEGER ( KIND = ipc_ ) :: semi_bandwidth
      INTEGER ( KIND = ipc_ ) :: lbfgs_vectors
      INTEGER ( KIND = ipc_ ) :: max_dxg
      INTEGER ( KIND = ipc_ ) :: icfs_vectors
      INTEGER ( KIND = ipc_ ) :: mi28_lsize
      INTEGER ( KIND = ipc_ ) :: mi28_rsize
      INTEGER ( KIND = ipc_ ) :: advanced_start
      REAL ( KIND = rpc_ ) :: stop_g_absolute
      REAL ( KIND = rpc_ ) :: stop_g_relative
      REAL ( KIND = rpc_ ) :: stop_s
      REAL ( KIND = rpc_ ) :: initial_weight
      REAL ( KIND = rpc_ ) :: minimum_weight
      REAL ( KIND = rpc_ ) :: reduce_gap
      REAL ( KIND = rpc_ ) :: tiny_gap
      REAL ( KIND = rpc_ ) :: large_root
      REAL ( KIND = rpc_ ) :: eta_successful
      REAL ( KIND = rpc_ ) :: eta_very_successful
      REAL ( KIND = rpc_ ) :: eta_too_successful
      REAL ( KIND = rpc_ ) :: weight_decrease_min
      REAL ( KIND = rpc_ ) :: weight_decrease
      REAL ( KIND = rpc_ ) :: weight_increase
      REAL ( KIND = rpc_ ) :: weight_increase_max
      REAL ( KIND = rpc_ ) :: obj_unbounded
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: hessian_available
      LOGICAL ( KIND = C_BOOL ) :: subproblem_direct
      LOGICAL ( KIND = C_BOOL ) :: renormalize_weight
      LOGICAL ( KIND = C_BOOL ) :: quadratic_ratio_test
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( rqs_control_type ) :: rqs_control
      TYPE ( glrt_control_type ) :: glrt_control
      TYPE ( dps_control_type ) :: dps_control
      TYPE ( psls_control_type ) :: psls_control
      TYPE ( lms_control_type ) :: lms_control
      TYPE ( lms_control_type ) :: lms_control_prec
      TYPE ( sha_control_type ) :: sha_control
    END TYPE arc_control_type

    TYPE, BIND( C ) :: arc_time_type
      REAL ( KIND = spc_ ) :: total
      REAL ( KIND = spc_ ) :: preprocess
      REAL ( KIND = spc_ ) :: analyse
      REAL ( KIND = spc_ ) :: factorize
      REAL ( KIND = spc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_preprocess
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE arc_time_type

    TYPE, BIND( C ) :: arc_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: cg_iter
      INTEGER ( KIND = ipc_ ) :: f_eval
      INTEGER ( KIND = ipc_ ) :: g_eval
      INTEGER ( KIND = ipc_ ) :: h_eval
      INTEGER ( KIND = ipc_ ) :: factorization_max
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = longc_ ) :: max_entries_factors
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      REAL ( KIND = rpc_ ) :: factorization_average
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: norm_g
      REAL ( KIND = rpc_ ) :: weight
      TYPE ( arc_time_type ) :: time
      TYPE ( rqs_inform_type ) :: rqs_inform
      TYPE ( glrt_inform_type ) :: glrt_inform
      TYPE ( dps_inform_type ) :: dps_inform
      TYPE ( psls_inform_type ) :: psls_inform
      TYPE ( lms_inform_type ) :: lms_inform
      TYPE ( lms_inform_type ) :: lms_inform_prec
      TYPE ( sha_inform_type ) :: sha_inform
    END TYPE arc_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_f( n, x, f, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), value :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), INTENT( OUT ) :: f
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_f
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_g( n, x, g, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: g
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_g
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_h( n, ne, x, hval, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: ne
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( ne ), INTENT( OUT ) :: hval
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_h
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_hprod( n, x, u, v, got_h, userdata ) RESULT( status )      &
                                                         BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( INOUT ) :: u
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: v
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ), VALUE :: got_h
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_hprod
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_prec( n, x, u, v, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: u
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: v
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_prec
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( arc_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_arc_control_type ), INTENT( OUT ) :: fcontrol
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
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%maxit = ccontrol%maxit
    fcontrol%alive_unit = ccontrol%alive_unit
    fcontrol%non_monotone = ccontrol%non_monotone
    fcontrol%model = ccontrol%model
    fcontrol%norm = ccontrol%norm
    fcontrol%semi_bandwidth = ccontrol%semi_bandwidth
    fcontrol%lbfgs_vectors = ccontrol%lbfgs_vectors
    fcontrol%max_dxg = ccontrol%max_dxg
    fcontrol%icfs_vectors = ccontrol%icfs_vectors
    fcontrol%mi28_lsize = ccontrol%mi28_lsize
    fcontrol%mi28_rsize = ccontrol%mi28_rsize
    fcontrol%advanced_start = ccontrol%advanced_start

    ! Reals
    fcontrol%stop_g_absolute = ccontrol%stop_g_absolute
    fcontrol%stop_g_relative = ccontrol%stop_g_relative
    fcontrol%stop_s = ccontrol%stop_s
    fcontrol%initial_weight = ccontrol%initial_weight
    fcontrol%minimum_weight = ccontrol%minimum_weight

    fcontrol%reduce_gap = ccontrol%reduce_gap
    fcontrol%tiny_gap = ccontrol%tiny_gap
    fcontrol%large_root = ccontrol%large_root

    fcontrol%eta_successful = ccontrol%eta_successful
    fcontrol%eta_very_successful = ccontrol%eta_very_successful
    fcontrol%eta_too_successful = ccontrol%eta_too_successful

    fcontrol%weight_decrease_min = ccontrol%weight_decrease_min
    fcontrol%weight_decrease  = ccontrol%weight_decrease
    fcontrol%weight_increase = ccontrol%weight_increase
    fcontrol%weight_increase_max = ccontrol%weight_increase_max

    fcontrol%obj_unbounded = ccontrol%obj_unbounded
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%hessian_available = ccontrol%hessian_available
    fcontrol%subproblem_direct = ccontrol%subproblem_direct
    fcontrol%renormalize_weight = ccontrol%renormalize_weight
    fcontrol%quadratic_ratio_test = ccontrol%quadratic_ratio_test
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_rqs_control_in( ccontrol%rqs_control,fcontrol%rqs_control )
    CALL copy_glrt_control_in( ccontrol%glrt_control,fcontrol%glrt_control )
    CALL copy_dps_control_in( ccontrol%dps_control,fcontrol%dps_control )
    CALL copy_psls_control_in( ccontrol%psls_control,fcontrol%psls_control )
    CALL copy_lms_control_in( ccontrol%lms_control,fcontrol%lms_control )
    CALL copy_lms_control_in( ccontrol%lms_control_prec,                       &
                              fcontrol%lms_control_prec)
    CALL copy_sha_control_in( ccontrol%sha_control,fcontrol%sha_control )

    ! Strings
    DO i = 1, LEN( fcontrol%alive_file )
      IF ( ccontrol%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%alive_file( i : i ) = ccontrol%alive_file( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN
    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_arc_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( arc_control_type ), INTENT( OUT ) :: ccontrol
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
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%maxit = fcontrol%maxit
    ccontrol%alive_unit = fcontrol%alive_unit
    ccontrol%non_monotone = fcontrol%non_monotone
    ccontrol%model = fcontrol%model
    ccontrol%norm = fcontrol%norm
    ccontrol%semi_bandwidth = fcontrol%semi_bandwidth
    ccontrol%lbfgs_vectors = fcontrol%lbfgs_vectors
    ccontrol%max_dxg = fcontrol%max_dxg
    ccontrol%icfs_vectors = fcontrol%icfs_vectors
    ccontrol%mi28_lsize = fcontrol%mi28_lsize
    ccontrol%mi28_rsize = fcontrol%mi28_rsize
    ccontrol%advanced_start = fcontrol%advanced_start

    ! Reals
    ccontrol%stop_g_absolute = fcontrol%stop_g_absolute
    ccontrol%stop_g_relative = fcontrol%stop_g_relative
    ccontrol%stop_s = fcontrol%stop_s
    ccontrol%initial_weight = fcontrol%initial_weight
    ccontrol%minimum_weight = fcontrol%minimum_weight
    ccontrol%reduce_gap = fcontrol%reduce_gap
    ccontrol%tiny_gap = fcontrol%tiny_gap
    ccontrol%large_root = fcontrol%large_root
    ccontrol%eta_successful = fcontrol%eta_successful
    ccontrol%eta_very_successful = fcontrol%eta_very_successful
    ccontrol%eta_too_successful = fcontrol%eta_too_successful
    ccontrol%weight_decrease_min = fcontrol%weight_decrease_min
    ccontrol%weight_decrease  = fcontrol%weight_decrease
    ccontrol%weight_increase = fcontrol%weight_increase
    ccontrol%weight_increase_max = fcontrol%weight_increase_max
    ccontrol%obj_unbounded = fcontrol%obj_unbounded
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%hessian_available = fcontrol%hessian_available
    ccontrol%subproblem_direct = fcontrol%subproblem_direct
    ccontrol%renormalize_weight = fcontrol%renormalize_weight
    ccontrol%quadratic_ratio_test = fcontrol%quadratic_ratio_test
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_rqs_control_out(fcontrol%rqs_control,ccontrol%rqs_control)
    CALL copy_glrt_control_out(fcontrol%glrt_control,ccontrol%glrt_control)
    CALL copy_dps_control_out(fcontrol%dps_control,ccontrol%dps_control)
    CALL copy_psls_control_out(fcontrol%psls_control,ccontrol%psls_control)
    CALL copy_lms_control_out(fcontrol%lms_control,ccontrol%lms_control)
    CALL copy_lms_control_out(fcontrol%lms_control_prec,                       &
                              ccontrol%lms_control_prec)
    CALL copy_sha_control_out(fcontrol%sha_control,ccontrol%sha_control)

    ! Strings
    l = LEN( fcontrol%alive_file )
    DO i = 1, l
      ccontrol%alive_file( i ) = fcontrol%alive_file( i : i )
    END DO
    ccontrol%alive_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C times to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( arc_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_arc_time_type ), INTENT( OUT ) :: ftime

    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran times to C

    SUBROUTINE copy_time_out( ftime,  ctime )
    TYPE ( f_arc_time_type ), INTENT( IN ) :: ftime
    TYPE ( arc_time_type ), INTENT( OUT ) :: ctime

    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C information parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( arc_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_arc_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%cg_iter = cinform%cg_iter
    finform%f_eval = cinform%f_eval
    finform%g_eval = cinform%g_eval
    finform%h_eval = cinform%h_eval
    finform%factorization_max = cinform%factorization_max
    finform%factorization_status = cinform%factorization_status
    finform%max_entries_factors = cinform%max_entries_factors
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real

    ! Reals
    finform%factorization_average = cinform%factorization_average
    finform%obj = cinform%obj
    finform%norm_g = cinform%norm_g
    finform%weight = cinform%weight

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_rqs_inform_in( cinform%rqs_inform, finform%rqs_inform )
    CALL copy_glrt_inform_in( cinform%glrt_inform, finform%glrt_inform )
    CALL copy_dps_inform_in( cinform%dps_inform, finform%dps_inform )
    CALL copy_psls_inform_in( cinform%psls_inform, finform%psls_inform )
    CALL copy_lms_inform_in( cinform%lms_inform, finform%lms_inform )
    CALL copy_lms_inform_in( cinform%lms_inform_prec,                          &
                             finform%lms_inform_prec )
    CALL copy_sha_inform_in( cinform%sha_inform, finform%sha_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_arc_inform_type ), INTENT( IN ) :: finform
    TYPE ( arc_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%cg_iter = finform%cg_iter
    cinform%f_eval = finform%f_eval
    cinform%g_eval = finform%g_eval
    cinform%h_eval = finform%h_eval
    cinform%factorization_max = finform%factorization_max
    cinform%factorization_status = finform%factorization_status
    cinform%max_entries_factors = finform%max_entries_factors
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real

    ! Reals
    cinform%factorization_average = finform%factorization_average
    cinform%obj = finform%obj
    cinform%norm_g = finform%norm_g
    cinform%weight = finform%weight

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_rqs_inform_out( finform%rqs_inform, cinform%rqs_inform )
    CALL copy_glrt_inform_out( finform%glrt_inform, cinform%glrt_inform )
    CALL copy_dps_inform_out( finform%dps_inform, cinform%dps_inform )
    CALL copy_psls_inform_out( finform%psls_inform, cinform%psls_inform )
    CALL copy_lms_inform_out( finform%lms_inform, cinform%lms_inform )
    CALL copy_lms_inform_out( finform%lms_inform_prec,                         &
                              cinform%lms_inform_prec )
    CALL copy_sha_inform_out( finform%sha_inform, cinform%sha_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    end do
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_ARC_precision_ciface

!  -------------------------------------
!  C interface to fortran arc_initialize
!  -------------------------------------

  SUBROUTINE arc_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( arc_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_arc_full_data_type ), POINTER :: fdata
  TYPE ( f_arc_control_type ) :: fcontrol
  TYPE ( f_arc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC(fdata)

!  initialize required fortran types

  CALL f_arc_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE arc_initialize

!  ----------------------------------------
!  C interface to fortran arc_read_specfile
!  ----------------------------------------

  SUBROUTINE arc_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( arc_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_arc_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  open( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_arc_read_specfile( fcontrol, device )

!  close specfile

  close( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE arc_read_specfile

!  ---------------------------------
!  C interface to fortran arc_inport
!  ---------------------------------

  SUBROUTINE arc_import( ccontrol, cdata, status, n, ctype,                    &
                         ne, row, col, ptr ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ), OPTIONAL :: row, col
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: ptr

!  local variables

  TYPE ( C_PTR ), INTENT( IN ), VALUE :: ctype
  TYPE ( arc_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( ctype ) ) :: ftype
  TYPE ( f_arc_control_type ) :: fcontrol
  TYPE ( f_arc_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  ftype = cstr_to_fchar( ctype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required ARC structure

  CALL f_arc_import( fcontrol, fdata, status, n, ftype, ne,                    &
                     row, col, ptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE arc_import

!  ----------------------------------------
!  C interface to fortran arc_reset_control
!  ----------------------------------------

  SUBROUTINE arc_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( arc_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_arc_control_type ) :: fcontrol
  TYPE ( f_arc_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_arc_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE arc_reset_control

!  -----------------------------------------
!  C interface to fortran arc_solve_with_mat
!  -----------------------------------------

  SUBROUTINE arc_solve_with_mat( cdata, cuserdata, status, n, x, g, ne,        &
                                 ceval_f, ceval_g, ceval_h,                    &
                                 ceval_prec ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_f, ceval_g
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_h, ceval_prec

!  local variables

  TYPE ( f_arc_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_f ), POINTER :: feval_f
  PROCEDURE( eval_g ), POINTER :: feval_g
  PROCEDURE( eval_h ), POINTER :: feval_h
  PROCEDURE( eval_prec ), POINTER :: feval_prec

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_galahad_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_f, feval_f )
  CALL C_F_PROCPOINTER( ceval_g, feval_g )
  CALL C_F_PROCPOINTER( ceval_h, feval_h )
  IF ( C_ASSOCIATED( ceval_prec ) ) THEN
    CALL C_F_PROCPOINTER( ceval_prec, feval_prec )
  ELSE
    NULLIFY( feval_prec )
  END IF

!  solve the problem when the Hessian is explicitly available

  IF ( ASSOCIATED( feval_prec ) ) THEN
    CALL f_arc_solve_with_mat( fdata, fuserdata, status, x, g,                 &
                               wrap_eval_f, wrap_eval_g, wrap_eval_h,          &
                               eval_prec = wrap_eval_prec )
  ELSE
    CALL f_arc_solve_with_mat( fdata, fuserdata, status, x, g,                 &
                               wrap_eval_f, wrap_eval_g, wrap_eval_h )
  END IF

  RETURN

!  wrappers

  CONTAINS

!  eval_F wrapper

    SUBROUTINE wrap_eval_f( status, x, userdata, f )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), INTENT( OUT ) :: f

!  call C interoperable eval_f

    status = feval_f( n, x, f, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_f

!  eval_G wrapper

    SUBROUTINE wrap_eval_g( status, x, userdata, g )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: g

!  Call C interoperable eval_g
    status = feval_g( n, x, g, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_g

!  eval_H wrapper

    SUBROUTINE wrap_eval_h( status, x, userdata, hval )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: hval

!  Call C interoperable eval_h
    status = feval_h( n, ne, x, hval, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_h

!  eval_PREC wrapper

    SUBROUTINE wrap_eval_prec( status, x, userdata, u, v )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v

!  Call C interoperable eval_prec
    status = feval_prec( n, x, u, v, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_prec

  END SUBROUTINE arc_solve_with_mat

!  --------------------------------------------
!  C interface to fortran arc_solve_without_mat
!  --------------------------------------------

  SUBROUTINE arc_solve_without_mat( cdata, cuserdata, status, n, x, g,         &
                                    ceval_f, ceval_g, ceval_hprod,             &
                                    ceval_prec ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_f, ceval_g
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_hprod, ceval_prec

!  local variables

  TYPE ( f_arc_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_f ), POINTER :: feval_f
  PROCEDURE( eval_g ), POINTER :: feval_g
  PROCEDURE( eval_hprod ), POINTER :: feval_hprod
  PROCEDURE( eval_prec ), POINTER :: feval_prec

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_galahad_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER(cdata, fdata)

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_f, feval_f )
  CALL C_F_PROCPOINTER( ceval_g, feval_g )
  CALL C_F_PROCPOINTER( ceval_hprod, feval_hprod )
  IF ( C_ASSOCIATED( ceval_prec ) ) THEN
    CALL C_F_PROCPOINTER( ceval_prec, feval_prec )
  ELSE
    NULLIFY( feval_prec )
  END IF

!  solve the problem when the Hessian is only available via products

  IF ( ASSOCIATED( feval_prec ) ) THEN
    CALL f_arc_solve_without_mat( fdata, fuserdata, status, x, g,              &
                                  wrap_eval_f, wrap_eval_g, wrap_eval_hprod,   &
                                  eval_prec = wrap_eval_prec )
  ELSE
    CALL f_arc_solve_without_mat( fdata, fuserdata, status, x, g,              &
                                  wrap_eval_f, wrap_eval_g, wrap_eval_hprod )
  END IF

  RETURN

!  wrappers

  CONTAINS

!  eval_F wrapper

    SUBROUTINE wrap_eval_f( status, x, userdata, f )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), INTENT( OUT ) :: f

!  call C interoperable eval_f
    status = feval_f( n, x, f, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_f

!  eval_G wrapper

    SUBROUTINE wrap_eval_g( status, x, userdata, g )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: g

!  call C interoperable eval_g

    status = feval_g( n, x, g, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_g

!  eval_HPROD wrapper

    SUBROUTINE wrap_eval_hprod( status, x, userdata, u, v, fgot_h )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( INOUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

!  call C interoperable eval_hprod

    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .false.
    END IF
    status = feval_hprod( n, x, u, v, cgot_h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hprod

!  eval_PREC wrapper

    SUBROUTINE wrap_eval_prec( status, x, userdata, u, v )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v

!  call C interoperable eval_prec

    status = feval_prec( n, x, u, v, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_prec

  END SUBROUTINE arc_solve_without_mat

!  -------------------------------------------------
!  C interface to fortran arc_solve_reverse_with_mat
!  -------------------------------------------------

  SUBROUTINE arc_solve_reverse_with_mat( cdata, status, eval_status,           &
                                         n, x, f, g, ne, val, u, v ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( ne ) :: val
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: u
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: v
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_arc_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when the Hessian is available by reverse communication

  CALL f_arc_solve_reverse_with_mat( fdata, status, eval_status, x, f, g, val, &
                                      u, v )
  RETURN

  END SUBROUTINE arc_solve_reverse_with_mat

!  ----------------------------------------------------
!  C interface to fortran arc_solve_reverse_without_mat
!  ----------------------------------------------------

  SUBROUTINE arc_solve_reverse_without_mat( cdata, status, eval_status,        &
                                            n, x, f, g, u, v ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g, u, v
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_arc_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when Hessian products are available by reverse
!  communication

  CALL f_arc_solve_reverse_without_mat( fdata, status, eval_status, x, f, g,   &
                                         u, v )
  RETURN

  END SUBROUTINE arc_solve_reverse_without_mat

!  --------------------------------------
!  C interface to fortran arc_information
!  --------------------------------------

  SUBROUTINE arc_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( arc_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_arc_full_data_type ), pointer :: fdata
  TYPE ( f_arc_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain ARC solution information

  CALL f_arc_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE arc_information

!  ------------------------------------
!  C interface to fortran arc_terminate
!  ------------------------------------

  SUBROUTINE arc_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_ARC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( arc_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( arc_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_arc_full_data_type ), pointer :: fdata
  TYPE ( f_arc_control_type ) :: fcontrol
  TYPE ( f_arc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_arc_terminate( fdata, fcontrol,finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE arc_terminate
