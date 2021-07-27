! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-  GALAHAD_TRB C INTERFACE  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Jaroslav Fowkes

!  History -
!   currently in development

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

module GALAHAD_TRB_double_ciface
    use iso_c_binding
    use GALAHAD_TRB_double, only:                                     &
        f_trb_time_type               => TRB_time_type,               &
        f_trb_inform_type             => TRB_inform_type,             &
        f_trb_control_type            => TRB_control_type,            &
        f_trb_full_data_type          => TRB_full_data_type,          &
        f_trb_initialize              => TRB_initialize,              &
        f_trb_read_specfile           => TRB_read_specfile,           &
        f_trb_import                  => TRB_import,                  &
        f_trb_solve_with_h            => TRB_solve_with_h,            &
        f_trb_solve_without_h         => TRB_solve_without_h,         &
        f_trb_solve_reverse_with_h    => TRB_solve_reverse_with_h,    &
        f_trb_solve_reverse_without_h => TRB_solve_reverse_without_h, &
        f_trb_terminate               => TRB_terminate,               &
        f_trb_projection              => TRB_projection
    use GALAHAD_NLPT_double, only:                                    &
        f_nlpt_userdata_type          => NLPT_userdata_type

    implicit none

    integer, parameter :: wp = C_DOUBLE ! double precision
    integer, parameter :: sp = C_FLOAT  ! single precision

    type, bind(C) :: trb_time_type
        real(sp) :: total
        real(sp) :: preprocess
        real(sp) :: analyse
        real(sp) :: factorize
        real(sp) :: solve
        real(wp) :: clock_total
        real(wp) :: clock_preprocess
        real(wp) :: clock_analyse
        real(wp) :: clock_factorize
        real(wp) :: clock_solve
    end type trb_time_type

    type, bind(C) :: trb_inform_type
        integer(C_INT) :: status
        integer(C_INT) :: alloc_status
        character(C_CHAR), dimension(81) :: bad_alloc
        integer(C_INT) :: iter
        integer(C_INT) :: cg_iter
        integer(C_INT) :: cg_maxit
        integer(C_INT) :: f_eval
        integer(C_INT) :: g_eval
        integer(C_INT) :: h_eval
        integer(C_INT) :: n_free
        integer(C_INT) :: factorization_max
        integer(C_INT) :: factorization_status
        integer(C_LONG) :: max_entries_factors
        integer(C_INT) :: factorization_integer
        integer(C_INT) :: factorization_real
        real(wp) :: obj
        real(wp) :: norm_pg
        real(wp) :: radius
        type(trb_time_type) :: time
        !type(trs_inform_type) :: trs_inform
        !type(gltr_inform_type) :: gltr_inform
        !type(psls_inform_type) :: psls_inform
        !type(lms_inform_type) :: lms_inform
        !type(lms_inform_type) :: lms_inform_prec
        !type(sha_inform_type) :: sha_inform
    end type trb_inform_type

    type, bind(C) :: trb_control_type
        logical(C_BOOL) :: f_indexing
        integer(C_INT) :: error
        integer(C_INT) :: out
        integer(C_INT) :: print_level
        integer(C_INT) :: start_print
        integer(C_INT) :: stop_print
        integer(C_INT) :: print_gap
        integer(C_INT) :: maxit
        integer(C_INT) :: alive_unit 
        character(C_CHAR), dimension(31) :: alive_file
        integer(C_INT) :: more_toraldo
        integer(C_INT) :: non_monotone
        integer(C_INT) :: model
        integer(C_INT) :: norm
        integer(C_INT) :: semi_bandwidth
        integer(C_INT) :: lbfgs_vectors
        integer(C_INT) :: max_dxg
        integer(C_INT) :: icfs_vectors
        integer(C_INT) :: mi28_lsize
        integer(C_INT) :: mi28_rsize
        real(wp) :: stop_pg_absolute
        real(wp) :: stop_pg_relative
        real(wp) :: stop_s
        integer(C_INT) :: advanced_start
        real(wp) :: infinity
        real(wp) :: initial_radius
        real(wp) :: maximum_radius
        real(wp) :: stop_rel_cg        
        real(wp) :: eta_successful
        real(wp) :: eta_very_successful
        real(wp) :: eta_too_successful
        real(wp) :: radius_increase
        real(wp) :: radius_reduce
        real(wp) :: radius_reduce_max
        real(wp) :: obj_unbounded
        real(wp) :: cpu_time_limit 
        real(wp) :: clock_time_limit 
        logical(C_BOOL) :: hessian_available
        logical(C_BOOL) :: subproblem_direct
        logical(C_BOOL) :: retrospective_trust_region
        logical(C_BOOL) :: renormalize_radius
        logical(C_BOOL) :: two_norm_tr
        logical(C_BOOL) :: exact_gcp
        logical(C_BOOL) :: accurate_bqp
        logical(C_BOOL) :: space_critical
        logical(C_BOOL) :: deallocate_error_fatal
        character(C_CHAR), dimension(31) :: prefix 
        !type(trs_control_type) :: trs_control
        !type(gltr_control_type) :: gltr_control
        !type(psls_control_type) :: psls_control
        !type(lms_control_type) :: lms_control
        !type(lms_control_type) :: lms_control_prec
        !type(sha_control_type) :: sha_control
    end type trb_control_type

    interface
        integer(C_SIZE_T) pure function strlen(cstr) bind(C)
            use iso_c_binding
            implicit none
            type(C_PTR), intent(in), value :: cstr
        end function strlen 
    end interface 

    abstract interface
        function eval_f(n, x, f, userdata) result(status)
            use iso_c_binding
            import :: wp
            
            integer(C_INT), intent(in), value :: n
            real(wp), dimension(n), intent(in) :: x
            real(wp), intent(out) :: f
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_f
    end interface

    abstract interface
        function eval_g(n, x, g, userdata) result(status)
            use iso_c_binding
            import :: wp
            
            integer(C_INT), intent(in), value :: n
            real(wp), dimension(n), intent(in) :: x
            real(wp), dimension(n), intent(out) :: g
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_g
    end interface

    abstract interface
        function eval_h(n, ne, x, hval, userdata) result(status)
            use iso_c_binding
            import :: wp
            
            integer(C_INT), intent(in), value :: n
            integer(C_INT), intent(in), value :: ne
            real(wp), dimension(n), intent(in) :: x
            real(wp), dimension(ne), intent(out) :: hval
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_h
    end interface

    abstract interface
        function eval_hprod(n, x, u, v, got_h, userdata) result(status)
            use iso_c_binding
            import :: wp
            
            integer(C_INT), intent(in), value :: n
            real(wp), dimension(n), intent(in) :: x
            real(wp), dimension(n), intent(inout) :: u
            real(wp), dimension(n), intent(in) :: v
            logical(C_BOOL), intent(in), value :: got_h
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_hprod
    end interface

    abstract interface
        function eval_shprod(n, x, nnz_v, index_nz_v, v, nnz_u, index_nz_u, u, got_h, userdata) result(status)
            use iso_c_binding
            import :: wp

            integer(C_INT), intent(in), value :: n
            real(wp), dimension(n), intent(in) :: x
            integer(C_INT), intent(in), value :: nnz_v
            integer(C_INT), dimension(n), intent(in) :: index_nz_v
            real(wp), dimension(n), intent(in) :: v
            integer(C_INT), intent(out) :: nnz_u 
            integer(C_INT), dimension(n), intent(out) :: index_nz_u
            real(wp), dimension(n), intent(out) :: u
            logical(C_BOOL), intent(in), value :: got_h
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_shprod
    end interface

    abstract interface
        function eval_prec(n, x, u, v, userdata) result(status)
            use iso_c_binding
            import :: wp

            integer(C_INT), intent(in), value :: n
            real(wp), dimension(n), intent(in) :: x
            real(wp), dimension(n), intent(out) :: u
            real(wp), dimension(n), intent(in) :: v
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_prec
    end interface

contains

    ! optional string length
    pure function opt_strlen(cstr) result(len)
        type(C_PTR), intent(in), value :: cstr
        integer(C_SIZE_T) :: len    

        if(C_ASSOCIATED(cstr)) then
            len = strlen(cstr)
        else
            len = 0
        end if
    end function opt_strlen

    function cstr_to_fchar(cstr) result(fchar)
        type(C_PTR) :: cstr
        character(kind=C_CHAR, len=strlen(cstr)) :: fchar
        
        integer :: i
        character(C_CHAR), dimension(:) , pointer :: temp

        call c_f_pointer(cstr, temp, shape = (/ strlen(cstr) /) )

        do i = 1, size(temp) 
            fchar(i:i) = temp(i)
        end do
    end function cstr_to_fchar

    subroutine copy_time_in(ctime, ftime) 
        type(trb_time_type), intent(in) :: ctime
        type(f_trb_time_type), intent(out) :: ftime

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
    end subroutine copy_time_in

    subroutine copy_time_out(ftime, ctime)
        type(f_trb_time_type), intent(in) :: ftime
        type(trb_time_type), intent(out) :: ctime

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
    end subroutine copy_time_out

    subroutine copy_inform_in(cinform, finform) 
        type(trb_inform_type), intent(in) :: cinform
        type(f_trb_inform_type), intent(out) :: finform
        integer :: i

        ! Integers
        finform%status = cinform%status
        finform%alloc_status = cinform%alloc_status
        finform%iter = cinform%iter
        finform%cg_iter = cinform%cg_iter
        finform%cg_maxit = cinform%cg_maxit
        finform%f_eval = cinform%f_eval
        finform%g_eval = cinform%g_eval
        finform%h_eval = cinform%h_eval
        finform%n_free = cinform%n_free
        finform%factorization_max = cinform%factorization_max
        finform%factorization_status = cinform%factorization_status
        finform%max_entries_factors = cinform%max_entries_factors
        finform%factorization_integer = cinform%factorization_integer
        finform%factorization_real = cinform%factorization_real

        ! Reals
        finform%obj = cinform%obj
        finform%norm_pg = cinform%norm_pg
        finform%radius = cinform%radius

        ! Derived types
        call copy_time_in(cinform%time,finform%time)
        !call copy_trs_inform_in(cinform%trs_inform,finform%trs_inform)
        !call copy_gltr_inform_in(cinform%gltr_inform,finform%gltr_inform)
        !call copy_psls_inform_in(cinform%psls_inform,finform%psls_inform)
        !call copy_lms_inform_in(cinform%lms_inform,finform%lms_inform)
        !call copy_lms_inform_prec_in(cinform%lms_inform_prec,finform%lms_inform_prec)
        !call copy_sha_inform_in(cinform%sha_inform,finform%sha_inform)

        ! Strings
        do i = 1, 81
            if (cinform%bad_alloc(i) == C_NULL_CHAR) exit
            finform%bad_alloc(i:i) = cinform%bad_alloc(i)
        end do
    end subroutine copy_inform_in

    subroutine copy_inform_out(finform, cinform)
        type(f_trb_inform_type), intent(in) :: finform 
        type(trb_inform_type), intent(out) :: cinform
        integer :: i

        ! Integers
        cinform%status = finform%status
        cinform%alloc_status = finform%alloc_status
        cinform%iter = finform%iter
        cinform%cg_iter = finform%cg_iter
        cinform%cg_maxit = finform%cg_maxit
        cinform%f_eval = finform%f_eval
        cinform%g_eval = finform%g_eval
        cinform%h_eval = finform%h_eval
        cinform%n_free = finform%n_free
        cinform%factorization_max = finform%factorization_max
        cinform%factorization_status = finform%factorization_status
        cinform%max_entries_factors = finform%max_entries_factors
        cinform%factorization_integer = finform%factorization_integer
        cinform%factorization_real = finform%factorization_real

        ! Reals
        cinform%obj = finform%obj
        cinform%norm_pg = finform%norm_pg
        cinform%radius = finform%radius

        ! Derived types
        call copy_time_out(finform%time,cinform%time)
        !call copy_trs_inform_out(finform%trs_inform,cinform%trs_inform)
        !call copy_gltr_inform_out(finform%gltr_inform,cinform%gltr_inform)
        !call copy_psls_inform_out(finform%psls_inform,cinform%psls_inform)
        !call copy_lms_inform_out(finform%lms_inform,cinform%lms_inform)
        !call copy_lms_inform_prec_out(finform%lms_inform_prec,cinform%lms_inform_prec)
        !call copy_sha_inform_out(finform%sha_inform,cinform%sha_inform)

        ! Strings
        do i = 1,len(finform%bad_alloc)
            cinform%bad_alloc(i) = finform%bad_alloc(i:i)
        end do
        cinform%bad_alloc(len(finform%bad_alloc) + 1) = C_NULL_CHAR
    end subroutine copy_inform_out

    subroutine copy_control_in(ccontrol, fcontrol, f_indexing) 
        type(trb_control_type), intent(in) :: ccontrol
        type(f_trb_control_type), intent(out) :: fcontrol
        logical, optional, intent(out) :: f_indexing
        integer :: i
        
        ! C or Fortran sparse matrix indexing
        if(present(f_indexing)) f_indexing = ccontrol%f_indexing

        ! Integers
        fcontrol%error = ccontrol%error
        fcontrol%out = ccontrol%out
        fcontrol%print_level = ccontrol%print_level
        fcontrol%start_print = ccontrol%start_print
        fcontrol%stop_print = ccontrol%stop_print
        fcontrol%print_gap = ccontrol%print_gap
        fcontrol%maxit = ccontrol%maxit
        fcontrol%alive_unit = ccontrol%alive_unit
        fcontrol%more_toraldo = ccontrol%more_toraldo
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

        ! Doubles
        fcontrol%stop_pg_absolute = ccontrol%stop_pg_absolute
        fcontrol%stop_pg_relative = ccontrol%stop_pg_relative
        fcontrol%stop_s = ccontrol%stop_s
        fcontrol%infinity = ccontrol%infinity
        fcontrol%initial_radius = ccontrol%initial_radius
        fcontrol%maximum_radius = ccontrol%maximum_radius
        fcontrol%stop_rel_cg = ccontrol%stop_rel_cg
        fcontrol%eta_successful = ccontrol%eta_successful
        fcontrol%eta_very_successful = ccontrol%eta_very_successful
        fcontrol%eta_too_successful = ccontrol%eta_too_successful
        fcontrol%radius_increase = ccontrol%radius_increase
        fcontrol%radius_reduce = ccontrol%radius_reduce
        fcontrol%radius_reduce_max = ccontrol%radius_reduce_max
        fcontrol%obj_unbounded = ccontrol%obj_unbounded
        fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
        fcontrol%clock_time_limit = ccontrol%clock_time_limit 

        ! Logicals
        fcontrol%hessian_available = ccontrol%hessian_available
        fcontrol%subproblem_direct = ccontrol%subproblem_direct
        fcontrol%retrospective_trust_region = ccontrol%retrospective_trust_region
        fcontrol%renormalize_radius = ccontrol%renormalize_radius
        fcontrol%two_norm_tr = ccontrol%two_norm_tr
        fcontrol%exact_gcp = ccontrol%exact_gcp
        fcontrol%accurate_bqp = ccontrol%accurate_bqp
        fcontrol%space_critical = ccontrol%space_critical
        fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

        ! Derived types
        !call copy_trs_control_in(ccontrol%trs_control,fcontrol%trs_control)
        !call copy_gltr_control_in(ccontrol%gltr_control,fcontrol%gltr_control)
        !call copy_psls_control_in(ccontrol%psls_control,fcontrol%psls_control)
        !call copy_lms_control_in(ccontrol%lms_control,fcontrol%lms_control)
        !call copy_lms_control_prec_in(ccontrol%lms_control_prec,fcontrol%lms_control_prec)
        !call copy_sha_control_in(ccontrol%sha_control,fcontrol%sha_control)

        ! Strings
        do i = 1, 31
            if (ccontrol%alive_file(i) == C_NULL_CHAR) exit
            fcontrol%alive_file(i:i) = ccontrol%alive_file(i)
        end do
        do i = 1, 31
            if (ccontrol%prefix(i) == C_NULL_CHAR) exit
            fcontrol%prefix(i:i) = ccontrol%prefix(i)
        end do
    end subroutine copy_control_in

    subroutine copy_control_out(fcontrol, ccontrol, f_indexing)
        type(f_trb_control_type), intent(in) :: fcontrol
        type(trb_control_type), intent(out) :: ccontrol
        logical, optional, intent(in) :: f_indexing
        integer :: i
        
        ! C or Fortran sparse matrix indexing
        if(present(f_indexing)) ccontrol%f_indexing = f_indexing

        ! Integers
        ccontrol%error = fcontrol%error
        ccontrol%out = fcontrol%out
        ccontrol%print_level = fcontrol%print_level
        ccontrol%start_print = fcontrol%start_print
        ccontrol%stop_print = fcontrol%stop_print
        ccontrol%print_gap = fcontrol%print_gap
        ccontrol%maxit = fcontrol%maxit
        ccontrol%alive_unit = fcontrol%alive_unit
        ccontrol%more_toraldo = fcontrol%more_toraldo
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

        ! Doubles
        ccontrol%stop_pg_absolute = fcontrol%stop_pg_absolute
        ccontrol%stop_pg_relative = fcontrol%stop_pg_relative
        ccontrol%stop_s = fcontrol%stop_s
        ccontrol%infinity = fcontrol%infinity
        ccontrol%initial_radius = fcontrol%initial_radius
        ccontrol%maximum_radius = fcontrol%maximum_radius
        ccontrol%stop_rel_cg = fcontrol%stop_rel_cg
        ccontrol%eta_successful = fcontrol%eta_successful
        ccontrol%eta_very_successful = fcontrol%eta_very_successful
        ccontrol%eta_too_successful = fcontrol%eta_too_successful
        ccontrol%radius_increase = fcontrol%radius_increase
        ccontrol%radius_reduce = fcontrol%radius_reduce
        ccontrol%radius_reduce_max = fcontrol%radius_reduce_max
        ccontrol%obj_unbounded = fcontrol%obj_unbounded
        ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
        ccontrol%clock_time_limit = fcontrol%clock_time_limit 

        ! Logicals
        ccontrol%hessian_available = fcontrol%hessian_available
        ccontrol%subproblem_direct = fcontrol%subproblem_direct
        ccontrol%retrospective_trust_region = fcontrol%retrospective_trust_region
        ccontrol%renormalize_radius = fcontrol%renormalize_radius
        ccontrol%two_norm_tr = fcontrol%two_norm_tr
        ccontrol%exact_gcp = fcontrol%exact_gcp
        ccontrol%accurate_bqp = fcontrol%accurate_bqp
        ccontrol%space_critical = fcontrol%space_critical
        ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

        ! Derived types
        !call copy_trs_control_out(fcontrol%trs_control,ccontrol%trs_control)
        !call copy_gltr_control_out(fcontrol%gltr_control,ccontrol%gltr_control)
        !call copy_psls_control_out(fcontrol%psls_control,ccontrol%psls_control)
        !call copy_lms_control_out(fcontrol%lms_control,ccontrol%lms_control)
        !call copy_lms_control_prec_out(fcontrol%lms_control_prec,ccontrol%lms_control_prec)
        !call copy_sha_control_out(fcontrol%sha_control,ccontrol%sha_control)

        ! Strings
        do i = 1,len(fcontrol%alive_file)
            ccontrol%alive_file(i) = fcontrol%alive_file(i:i)
        end do
        ccontrol%alive_file(len(fcontrol%alive_file) + 1) = C_NULL_CHAR
        do i = 1,len(fcontrol%prefix)
            ccontrol%prefix(i) = fcontrol%prefix(i:i)
        end do
        ccontrol%prefix(len(fcontrol%prefix) + 1) = C_NULL_CHAR
    end subroutine copy_control_out

end module GALAHAD_TRB_double_ciface

subroutine trb_initialize(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_TRB_double_ciface
    implicit none

    type(C_PTR), intent(out) :: cdata ! data is a black-box
    type(trb_control_type), intent(out) :: ccontrol
    type(trb_inform_type), intent(out) :: cinform

    type(f_trb_full_data_type), pointer :: fdata
    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    logical :: f_indexing 

    ! Allocate fdata
    allocate(fdata); cdata = C_LOC(fdata)

    ! Call TRB_initialize
    call f_trb_initialize(fdata, fcontrol, finform)

    ! C sparse matrix indexing by default
    f_indexing = .false.

    ! Copy control out 
    call copy_control_out(fcontrol, ccontrol, f_indexing)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine trb_initialize

subroutine trb_read_specfile(ccontrol, cspecfile) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    type(trb_control_type), intent(inout) :: ccontrol
    type(C_PTR), intent(in), value :: cspecfile

    type(f_trb_control_type) :: fcontrol
    character(kind=C_CHAR, len=strlen(cspecfile)) :: fspecfile
    logical :: f_indexing

    ! Device unit number for specfile
    integer(C_INT), parameter :: device = 10

    ! Convert C string to Fortran string
    fspecfile = cstr_to_fchar(cspecfile)

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    
    ! Open specfile for reading
    open(unit=device, file=fspecfile)
    
    ! Call TRB_read_specfile
    call f_trb_read_specfile(fcontrol, device)

    ! Close specfile
    close(device)

    ! Copy control out
    call copy_control_out(fcontrol, ccontrol, f_indexing)
end subroutine trb_read_specfile

subroutine trb_import(ccontrol, cinform, cdata, n, xl, xu, ctype, ne, row, col, ptr) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, ne
    real(wp), intent(in), dimension(n) :: xl, xu
    integer(C_INT), intent(in), dimension(ne), optional :: row, col
    integer(C_INT), intent(in), dimension(n+1), optional :: ptr

    type(C_PTR), intent(in), value :: ctype
    type(trb_control_type), intent(inout) :: ccontrol
    type(trb_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    character(kind=C_CHAR, len=opt_strlen(ctype)) :: ftype
    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    type(f_trb_full_data_type), pointer :: fdata
    integer, dimension(:), allocatable :: row_find, col_find, ptr_find
    logical :: f_indexing
    

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Convert C string to Fortran string
    ftype = cstr_to_fchar(ctype)

    ! Handle C sparse matrix indexing
    if(.not.f_indexing) then
        if(present(row)) then
            allocate(row_find(ne))
            row_find = row + 1
        end if
        if(present(col)) then
            allocate(col_find(ne))
            col_find = col + 1
        end if
        if(present(ptr)) then
            allocate(ptr_find(n+1))
            ptr_find = ptr + 1
        end if
    end if

    ! Call TRB_import
    if(f_indexing) then
        call f_trb_import(fcontrol, finform, fdata, n, xl, xu, ftype, ne, row, col, ptr)
    else ! handle C sparse matrix indexing
        call f_trb_import(fcontrol, finform, fdata, n, xl, xu, ftype, ne, row_find, col_find, ptr_find)
    end if

    ! Copy control out 
    call copy_control_out(fcontrol, ccontrol, f_indexing)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine trb_import

subroutine trb_solve_with_h(ccontrol, cinform, cdata, cuserdata, n, x, g, ne, ceval_f, ceval_g, ceval_h, ceval_prec) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, ne
    real(wp), intent(inout), dimension(n) :: x, g 

    type(trb_control_type), intent(in) :: ccontrol
    type(trb_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata
    type(C_PTR), intent(in), value :: cuserdata
    type(C_FUNPTR), intent(in), value :: ceval_f, ceval_g, ceval_h, ceval_prec

    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    type(f_trb_full_data_type), pointer :: fdata
    procedure(eval_f), pointer :: feval_f
    procedure(eval_g), pointer :: feval_g
    procedure(eval_h), pointer :: feval_h
    procedure(eval_prec), pointer :: feval_prec
    logical :: f_indexing

    ! Ignore Fortran userdata type (not interoperable)
    type(f_nlpt_userdata_type), pointer :: fuserdata => null()

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Associate procedure pointers
    call C_F_PROCPOINTER(ceval_f, feval_f)
    call C_F_PROCPOINTER(ceval_g, feval_g)
    call C_F_PROCPOINTER(ceval_h, feval_h)
    if(C_ASSOCIATED(ceval_prec)) then 
        call C_F_PROCPOINTER(ceval_prec, feval_prec)
    else
        nullify(feval_prec)
    endif

    ! Call TRB_solve_with_h
    call f_trb_solve_with_h(fcontrol, finform, fdata, fuserdata, x, g, &
                            wrap_eval_f, wrap_eval_g, wrap_eval_h, wrap_eval_prec)

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    contains

    ! eval_F wrapper
    subroutine wrap_eval_f(status, x, userdata, f)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), intent(out) :: f

        ! Call C interoperable eval_f
        status = feval_f(n, x, f, cuserdata)
    end subroutine wrap_eval_f

    ! eval_G wrapper
    subroutine wrap_eval_g(status, x, userdata, g)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), dimension(:), intent(out) :: g

        ! Call C interoperable eval_g
        status = feval_g(n, x, g, cuserdata)
    end subroutine wrap_eval_g

    ! eval_H wrapper
    subroutine wrap_eval_h(status, x, userdata, hval)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), dimension(:), intent(out) :: hval

        ! Call C interoperable eval_h
        status = feval_h(n, ne, x, hval, cuserdata)
    end subroutine wrap_eval_h

    ! eval_PREC wrapper
    subroutine wrap_eval_prec(status, x, userdata, u, v)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), dimension(:), intent(out) :: u
        real(wp), dimension(:), intent(in) :: v

        ! Call C interoperable eval_prec
        status = feval_prec(n, x, u, v, cuserdata)
    end subroutine wrap_eval_prec

end subroutine trb_solve_with_h

subroutine trb_solve_without_h(ccontrol, cinform, cdata, cuserdata, n, x, g, &
                               ceval_f, ceval_g, ceval_hprod, ceval_shprod, ceval_prec) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n
    real(wp), intent(inout), dimension(n) :: x, g 

    type(trb_control_type), intent(in) :: ccontrol
    type(trb_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata
    type(C_PTR), intent(in), value :: cuserdata
    type(C_FUNPTR), intent(in), value :: ceval_f, ceval_g, ceval_hprod, ceval_shprod, ceval_prec

    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    type(f_trb_full_data_type), pointer :: fdata
    procedure(eval_f), pointer :: feval_f
    procedure(eval_g), pointer :: feval_g
    procedure(eval_hprod), pointer :: feval_hprod
    procedure(eval_shprod), pointer :: feval_shprod
    procedure(eval_prec), pointer :: feval_prec
    logical :: f_indexing

    ! Ignore Fortran userdata type (not interoperable)
    type(f_nlpt_userdata_type), pointer :: fuserdata => null()

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Associate procedure pointers
    call C_F_PROCPOINTER(ceval_f, feval_f) 
    call C_F_PROCPOINTER(ceval_g, feval_g)
    call C_F_PROCPOINTER(ceval_hprod, feval_hprod)
    call C_F_PROCPOINTER(ceval_shprod, feval_shprod)
    if(C_ASSOCIATED(ceval_prec)) then 
        call C_F_PROCPOINTER(ceval_prec, feval_prec)
    else
        nullify(feval_prec)
    endif

    ! Call TRB_solve_without_h
    call f_trb_solve_without_h(fcontrol, finform, fdata, fuserdata, x, g, &
                               wrap_eval_f, wrap_eval_g, wrap_eval_hprod, wrap_eval_shprod, wrap_eval_prec)

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    contains

    ! eval_F wrapper
    subroutine wrap_eval_f(status, x, userdata, f)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), intent(out) :: f

        ! Call C interoperable eval_f
        status = feval_f(n, x, f, cuserdata)
    end subroutine wrap_eval_f

    ! eval_G wrapper
    subroutine wrap_eval_g(status, x, userdata, g)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), dimension(:), intent(out) :: g

        ! Call C interoperable eval_g
        status = feval_g(n, x, g, cuserdata)
    end subroutine wrap_eval_g

    ! eval_HPROD wrapper 
    subroutine wrap_eval_hprod(status, x, userdata, u, v, fgot_h)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), dimension(:), intent(inout) :: u
        real(wp), dimension(:), intent(in) :: v
        logical, optional, intent(in) :: fgot_h
        logical(C_BOOL) :: cgot_h

        ! Call C interoperable eval_hprod
        if(present(fgot_h)) then
            cgot_h = fgot_h
        else
            cgot_h = .false.
        end if
        status = feval_hprod(n, x, u, v, cgot_h, cuserdata)
    end subroutine wrap_eval_hprod

    ! eval_SHPROD wrapper
    subroutine wrap_eval_shprod(status, x, userdata, nnz_v, index_nz_v, v, &
        nnz_u, index_nz_u, u, fgot_h)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        integer(C_INT), intent(in) :: nnz_v
        integer(C_INT), dimension(:), intent(in) :: index_nz_v
        real(wp), dimension(:), intent(in) :: v
        integer(C_INT), intent(out) :: nnz_u 
        integer(C_INT), dimension(:), intent(out) :: index_nz_u
        real(wp), dimension(:), intent(out) :: u
        logical, optional, intent(in) :: fgot_h
        logical(C_BOOL) :: cgot_h

        ! Call C interoperable eval_shprod
        if(present(fgot_h)) then
            cgot_h = fgot_h 
        else
            cgot_h = .false.
        end if
        if(f_indexing) then
            status = feval_shprod(n, x, nnz_v, index_nz_v, v, nnz_u, index_nz_u, u, cgot_h, cuserdata)
        else ! handle C sparse matrix indexing
            status = feval_shprod(n, x, nnz_v, index_nz_v-1, v, nnz_u, index_nz_u, u, cgot_h, cuserdata)
            index_nz_u = index_nz_u + 1
        endif
    end subroutine wrap_eval_shprod

    ! eval_PREC wrapper
    subroutine wrap_eval_prec(status, x, userdata, u, v)
        integer(C_INT), intent(out) :: status
        real(wp), dimension(:), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), dimension(:), intent(out) :: u
        real(wp), dimension(:), intent(in) :: v

        ! Call C interoperable eval_prec
        status = feval_prec(n, x, u, v, cuserdata)
    end subroutine wrap_eval_prec

end subroutine trb_solve_without_h

subroutine trb_solve_reverse_with_h(ccontrol, cinform, cdata, eval_status, n, x, f, g, ne, val, u, v) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, ne
    integer(C_INT), intent(inout) :: eval_status
    real(wp), intent(in), value :: f
    real(wp), intent(inout), dimension(n) :: x, g 
    real(wp), intent(inout), dimension(ne) :: val
    real(wp), intent(in), dimension(n) :: u
    real(wp), intent(out), dimension(n) :: v

    type(trb_control_type), intent(in) :: ccontrol
    type(trb_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    type(f_trb_full_data_type), pointer :: fdata
    logical :: f_indexing

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Call TRB_solve_reverse_with_h
    call f_trb_solve_reverse_with_h(fcontrol, finform, fdata, eval_status, x, f, g, val, u, v)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
    
end subroutine trb_solve_reverse_with_h

subroutine trb_solve_reverse_without_h(ccontrol, cinform, cdata, eval_status, n, x, f, g, u, v, &
                                       index_nz_v, nnz_v, index_nz_u, nnz_u) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, nnz_u
    integer(C_INT), intent(inout) :: eval_status
    integer(C_INT), intent(out) :: nnz_v
    real(wp), intent(in), value :: f
    real(wp), intent(inout), dimension(n) :: x, g, u, v
    integer(C_INT), intent(in), dimension(n) :: index_nz_u
    integer(C_INT), intent(out), dimension(n) :: index_nz_v

    type(trb_control_type), intent(in) :: ccontrol
    type(trb_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    type(f_trb_full_data_type), pointer :: fdata
    logical :: f_indexing

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Call TRB_solve_reverse_without_h
    if(f_indexing) then
        call f_trb_solve_reverse_without_h(fcontrol, finform, fdata, eval_status, x, f, g, u, v, &
                                           index_nz_v, nnz_v, index_nz_u, nnz_u)
    else
        call f_trb_solve_reverse_without_h(fcontrol, finform, fdata, eval_status, x, f, g, u, v, &
                                           index_nz_v, nnz_v, index_nz_u+1, nnz_u) 
    end if

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    ! Convert to C indexing if required
    if(.not.f_indexing) then
        index_nz_v = index_nz_v - 1
    endif

end subroutine trb_solve_reverse_without_h

subroutine trb_terminate(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_TRB_double_ciface
    implicit none

    type(C_PTR), intent(inout) :: cdata
    type(trb_control_type), intent(in) :: ccontrol
    type(trb_inform_type), intent(inout) :: cinform

    type(f_trb_full_data_type), pointer :: fdata
    type(f_trb_control_type) :: fcontrol
    type(f_trb_inform_type) :: finform
    logical :: f_indexing

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol, f_indexing)

    ! Copy inform in
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Call TRB_terminate
    call f_trb_terminate(fdata,fcontrol,finform)

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    ! Deallocate data
    deallocate(fdata); cdata = C_NULL_PTR 
end subroutine trb_terminate

subroutine trb_projection(n, x, x_l, x_u, projection) bind(C)
    use GALAHAD_TRB_double_ciface
    implicit none

    integer(C_INT), value, intent(in) :: n
    real(wp), intent(in), dimension(n) :: x, x_l, x_u
    real(wp), intent(out), dimension(n) :: projection

    ! Call TRB_projection
    projection = f_trb_projection(n, x, x_l, x_u)
end subroutine trb_projection
