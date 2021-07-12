! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-  GALAHAD_BGO C INTERFACE  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Jaroslav Fowkes

!  History -
!   currently in development

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

module GALAHAD_BGO_double_ciface
    use iso_c_binding
    use GALAHAD_BGO_double, only:                                     &
        f_bgo_time_type               => BGO_time_type,               &
        f_bgo_inform_type             => BGO_inform_type,             &
        f_bgo_control_type            => BGO_control_type,            &
        f_bgo_full_data_type          => BGO_full_data_type,          &
        f_bgo_initialize              => BGO_initialize,              &
        f_bgo_read_specfile           => BGO_read_specfile,           &
        f_bgo_import                  => BGO_import,                  &
        f_bgo_solve_with_h            => BGO_solve_with_h,            &
        f_bgo_solve_without_h         => BGO_solve_without_h,         &
        f_bgo_solve_reverse_with_h    => BGO_solve_reverse_with_h,    &
        f_bgo_solve_reverse_without_h => BGO_solve_reverse_without_h, &
        f_bgo_terminate               => BGO_terminate
    use GALAHAD_NLPT_double, only:                                    &
        f_nlpt_userdata_type          => NLPT_userdata_type

    use GALAHAD_TRB_double_ciface, only:                              &
        trb_inform_type,                                              &
        trb_control_type,                                             &
        copy_trb_inform_in            => copy_inform_in,              &
        copy_trb_inform_out           => copy_inform_out,             &
        copy_trb_control_in           => copy_control_in,             &
        copy_trb_control_out          => copy_control_out
    use GALAHAD_UGO_double_ciface, only:                              &
        ugo_inform_type,                                              &
        ugo_control_type,                                             &
        copy_ugo_inform_in            => copy_inform_in,              &
        copy_ugo_inform_out           => copy_inform_out,             &
        copy_ugo_control_in           => copy_control_in,             &
        copy_ugo_control_out          => copy_control_out
    use GALAHAD_LHS_double_ciface, only:                              &
        lhs_inform_type,                                              &
        lhs_control_type,                                             &
        copy_lhs_inform_in            => copy_inform_in,              &
        copy_lhs_inform_out           => copy_inform_out,             &
        copy_lhs_control_in           => copy_control_in,             &
        copy_lhs_control_out          => copy_control_out

    implicit none

    integer, parameter :: wp = C_DOUBLE ! double precision
    integer, parameter :: sp = C_FLOAT  ! single precision

    type, bind(C) :: bgo_time_type
        real(sp) :: total
        real(sp) :: univariate_global
        real(sp) :: multivariate_local
        real(wp) :: clock_total
        real(wp) :: clock_univariate_global
        real(wp) :: clock_multivariate_local
    end type bgo_time_type

    type, bind(C) :: bgo_inform_type
        integer(C_INT) :: status
        integer(C_INT) :: alloc_status
        character(C_CHAR), dimension(81) :: bad_alloc
        integer(C_INT) :: f_eval
        integer(C_INT) :: g_eval
        integer(C_INT) :: h_eval
        real(wp) :: obj
        real(wp) :: norm_pg
        type(bgo_time_type) :: time
        type(trb_inform_type) :: trb_inform
        type(ugo_inform_type) :: ugo_inform
        type(lhs_inform_type) :: lhs_inform
    end type bgo_inform_type

    type, bind(C) :: bgo_control_type
        logical(C_BOOL) :: f_indexing
        integer(C_INT) :: error
        integer(C_INT) :: out
        integer(C_INT) :: print_level
        integer(C_INT) :: attempts_max
        integer(C_INT) :: max_evals
        integer(C_INT) :: sampling_strategy
        integer(C_INT) :: hypercube_discretization
        integer(C_INT) :: alive_unit 
        character(C_CHAR), dimension(31) :: alive_file
        real(wp) :: infinity
        real(wp) :: obj_unbounded
        real(wp) :: cpu_time_limit 
        real(wp) :: clock_time_limit 
        logical(C_BOOL) :: random_multistart
        logical(C_BOOL) :: hessian_available
        logical(C_BOOL) :: space_critical
        logical(C_BOOL) :: deallocate_error_fatal
        character(C_CHAR), dimension(31) :: prefix 
        type(trb_control_type) :: trb_control
        type(ugo_control_type) :: ugo_control
        type(lhs_control_type) :: lhs_control
    end type bgo_control_type

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
        type(bgo_time_type), intent(in) :: ctime
        type(f_bgo_time_type), intent(out) :: ftime

        ftime%total = ctime%total
        ftime%univariate_global = ctime%univariate_global
        ftime%multivariate_local = ctime%multivariate_local
        ftime%clock_total = ctime%clock_total
        ftime%clock_univariate_global = ctime%clock_univariate_global
        ftime%clock_multivariate_local = ctime%clock_multivariate_local
    end subroutine copy_time_in

    subroutine copy_time_out(ftime, ctime)
        type(f_bgo_time_type), intent(in) :: ftime
        type(bgo_time_type), intent(out) :: ctime

        ctime%total = ftime%total
        ctime%univariate_global = ftime%univariate_global
        ctime%multivariate_local = ftime%multivariate_local
        ctime%clock_total = ftime%clock_total
        ctime%clock_univariate_global = ftime%clock_univariate_global
        ctime%clock_multivariate_local = ftime%clock_multivariate_local
    end subroutine copy_time_out

    subroutine copy_inform_in(cinform, finform) 
        type(bgo_inform_type), intent(in) :: cinform
        type(f_bgo_inform_type), intent(out) :: finform
        integer :: i

        ! Integers
        finform%status = cinform%status
        finform%alloc_status = cinform%alloc_status
        finform%f_eval = cinform%f_eval
        finform%g_eval = cinform%g_eval
        finform%h_eval = cinform%h_eval

        ! Reals
        finform%obj = cinform%obj
        finform%norm_pg = cinform%norm_pg

        ! Derived types
        call copy_time_in(cinform%time,finform%time)
        call copy_trb_inform_in(cinform%trb_inform,finform%trb_inform)
        call copy_ugo_inform_in(cinform%ugo_inform,finform%ugo_inform)
        call copy_lhs_inform_in(cinform%lhs_inform,finform%lhs_inform)

        ! Strings
        do i = 1, 81
            if (cinform%bad_alloc(i) == C_NULL_CHAR) exit
            finform%bad_alloc(i:i) = cinform%bad_alloc(i)
        end do
    end subroutine copy_inform_in

    subroutine copy_inform_out(finform, cinform)
        type(f_bgo_inform_type), intent(in) :: finform 
        type(bgo_inform_type), intent(out) :: cinform
        integer :: i

        ! Integers
        cinform%status = finform%status
        cinform%alloc_status = finform%alloc_status
        cinform%f_eval = finform%f_eval
        cinform%g_eval = finform%g_eval
        cinform%h_eval = finform%h_eval

        ! Reals
        cinform%obj = finform%obj
        cinform%norm_pg = finform%norm_pg

        ! Derived types
        call copy_time_out(finform%time,cinform%time)
        call copy_trb_inform_out(finform%trb_inform,cinform%trb_inform)
        call copy_ugo_inform_out(finform%ugo_inform,cinform%ugo_inform)
        call copy_lhs_inform_out(finform%lhs_inform,cinform%lhs_inform)

        ! Strings
        do i = 1,len(finform%bad_alloc)
            cinform%bad_alloc(i) = finform%bad_alloc(i:i)
        end do
        cinform%bad_alloc(len(finform%bad_alloc) + 1) = C_NULL_CHAR
    end subroutine copy_inform_out

    subroutine copy_control_in(ccontrol, fcontrol, f_indexing) 
        type(bgo_control_type), intent(in) :: ccontrol
        type(f_bgo_control_type), intent(out) :: fcontrol
        logical, optional, intent(out) :: f_indexing
        integer :: i
        
        ! C or Fortran sparse matrix indexing
        if(present(f_indexing)) f_indexing = ccontrol%f_indexing

        ! Integers
        fcontrol%error = ccontrol%error
        fcontrol%out = ccontrol%out
        fcontrol%print_level = ccontrol%print_level
        fcontrol%attempts_max = ccontrol%attempts_max
        fcontrol%max_evals = ccontrol%max_evals
        fcontrol%sampling_strategy = ccontrol%sampling_strategy
        fcontrol%hypercube_discretization = ccontrol%hypercube_discretization
        fcontrol%alive_unit = ccontrol%alive_unit

        ! Doubles
        fcontrol%infinity = ccontrol%infinity
        fcontrol%obj_unbounded = ccontrol%obj_unbounded
        fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
        fcontrol%clock_time_limit = ccontrol%clock_time_limit 

        ! Logicals
        fcontrol%random_multistart = ccontrol%random_multistart
        fcontrol%hessian_available = ccontrol%hessian_available
        fcontrol%space_critical = ccontrol%space_critical
        fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

        ! Derived types
        call copy_trb_control_in(ccontrol%trb_control,fcontrol%trb_control)
        call copy_ugo_control_in(ccontrol%ugo_control,fcontrol%ugo_control)
        call copy_lhs_control_in(ccontrol%lhs_control,fcontrol%lhs_control)

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
        type(f_bgo_control_type), intent(in) :: fcontrol
        type(bgo_control_type), intent(out) :: ccontrol
        logical, optional, intent(in) :: f_indexing
        integer :: i
        
        ! C or Fortran sparse matrix indexing
        if(present(f_indexing)) ccontrol%f_indexing = f_indexing

        ! Integers
        ccontrol%error = fcontrol%error
        ccontrol%out = fcontrol%out
        ccontrol%print_level = fcontrol%print_level
        ccontrol%attempts_max = fcontrol%attempts_max
        ccontrol%max_evals = fcontrol%max_evals
        ccontrol%sampling_strategy = fcontrol%sampling_strategy
        ccontrol%hypercube_discretization = fcontrol%hypercube_discretization
        ccontrol%alive_unit = fcontrol%alive_unit

        ! Doubles
        ccontrol%infinity = fcontrol%infinity
        ccontrol%obj_unbounded = fcontrol%obj_unbounded
        ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
        ccontrol%clock_time_limit = fcontrol%clock_time_limit 

        ! Logicals
        ccontrol%random_multistart = fcontrol%random_multistart
        ccontrol%hessian_available = fcontrol%hessian_available
        ccontrol%space_critical = fcontrol%space_critical
        ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

        ! Derived types
        call copy_trb_control_out(fcontrol%trb_control,ccontrol%trb_control)
        call copy_ugo_control_out(fcontrol%ugo_control,ccontrol%ugo_control)
        call copy_lhs_control_out(fcontrol%lhs_control,ccontrol%lhs_control)

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

end module GALAHAD_BGO_double_ciface

subroutine bgo_initialize(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_BGO_double_ciface
    implicit none

    type(C_PTR), intent(out) :: cdata ! data is a black-box
    type(bgo_control_type), intent(out) :: ccontrol
    type(bgo_inform_type), intent(out) :: cinform

    type(f_bgo_full_data_type), pointer :: fdata
    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    logical :: f_indexing 

    ! Allocate fdata
    allocate(fdata); cdata = C_LOC(fdata)

    ! Call BGO_initialize
    call f_bgo_initialize(fdata, fcontrol, finform)

    ! C sparse matrix indexing by default
    f_indexing = .false.

    ! Copy control out 
    call copy_control_out(fcontrol, ccontrol, f_indexing)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine bgo_initialize

subroutine bgo_read_specfile(ccontrol, cspecfile) bind(C)
    use GALAHAD_BGO_double_ciface
    implicit none

    type(bgo_control_type), intent(inout) :: ccontrol
    type(C_PTR), intent(in), value :: cspecfile

    type(f_bgo_control_type) :: fcontrol
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
    
    ! Call BGO_read_specfile
    call f_bgo_read_specfile(fcontrol, device)

    ! Close specfile
    close(device)

    ! Copy control out
    call copy_control_out(fcontrol, ccontrol, f_indexing)
end subroutine bgo_read_specfile

subroutine bgo_import(ccontrol, cinform, cdata, n, xl, xu, ctype, ne, row, col, ptr) bind(C)
    use GALAHAD_BGO_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, ne
    real(wp), intent(in), dimension(n) :: xl, xu
    integer(C_INT), intent(in), dimension(ne), optional :: row, col
    integer(C_INT), intent(in), dimension(n+1), optional :: ptr

    type(C_PTR), intent(in), value :: ctype
    type(bgo_control_type), intent(inout) :: ccontrol
    type(bgo_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    character(kind=C_CHAR, len=opt_strlen(ctype)) :: ftype
    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    type(f_bgo_full_data_type), pointer :: fdata
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

    ! Call BGO_import
    if(f_indexing) then
        call f_bgo_import(fcontrol, finform, fdata, n, xl, xu, ftype, ne, row, col, ptr)
    else ! handle C sparse matrix indexing
        call f_bgo_import(fcontrol, finform, fdata, n, xl, xu, ftype, ne, row_find, col_find, ptr_find)
    end if

    ! Copy control out 
    call copy_control_out(fcontrol, ccontrol, f_indexing)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine bgo_import

subroutine bgo_solve_with_h(ccontrol, cinform, cdata, cuserdata, n, x, g, ne, &
                            ceval_f, ceval_g, ceval_h, ceval_hprod, ceval_prec) bind(C)
    use GALAHAD_BGO_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, ne
    real(wp), intent(inout), dimension(n) :: x, g 

    type(bgo_control_type), intent(in) :: ccontrol
    type(bgo_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata
    type(C_PTR), intent(in), value :: cuserdata
    type(C_FUNPTR), intent(in), value :: ceval_f, ceval_g, ceval_h, ceval_hprod, ceval_prec

    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    type(f_bgo_full_data_type), pointer :: fdata
    procedure(eval_f), pointer :: feval_f
    procedure(eval_g), pointer :: feval_g
    procedure(eval_h), pointer :: feval_h
    procedure(eval_hprod), pointer :: feval_hprod
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
    call C_F_PROCPOINTER(ceval_hprod, feval_hprod)
    if(C_ASSOCIATED(ceval_prec)) then 
        call C_F_PROCPOINTER(ceval_prec, feval_prec)
    else
        nullify(feval_prec)
    endif

    ! Call BGO_solve_with_h
    call f_bgo_solve_with_h(fcontrol, finform, fdata, fuserdata, x, g, &
                            wrap_eval_f, wrap_eval_g, wrap_eval_h, wrap_eval_hprod, wrap_eval_prec)

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

end subroutine bgo_solve_with_h

subroutine bgo_solve_without_h(ccontrol, cinform, cdata, cuserdata, n, x, g, &
                               ceval_f, ceval_g, ceval_hprod, ceval_shprod, ceval_prec) bind(C)
    use GALAHAD_BGO_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n
    real(wp), intent(inout), dimension(n) :: x, g 

    type(bgo_control_type), intent(in) :: ccontrol
    type(bgo_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata
    type(C_PTR), intent(in), value :: cuserdata
    type(C_FUNPTR), intent(in), value :: ceval_f, ceval_g, ceval_hprod, ceval_shprod, ceval_prec

    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    type(f_bgo_full_data_type), pointer :: fdata
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

    ! Call BGO_solve_without_h
    call f_bgo_solve_without_h(fcontrol, finform, fdata, fuserdata, x, g, &
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

end subroutine bgo_solve_without_h

subroutine bgo_solve_reverse_with_h(ccontrol, cinform, cdata, eval_status, n, x, f, g, ne, val, u, v) bind(C)
    use GALAHAD_BGO_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, ne
    integer(C_INT), intent(inout) :: eval_status
    real(wp), intent(in), value :: f
    real(wp), intent(inout), dimension(n) :: x, g, u, v 
    real(wp), intent(inout), dimension(ne) :: val

    type(bgo_control_type), intent(in) :: ccontrol
    type(bgo_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    type(f_bgo_full_data_type), pointer :: fdata
    logical :: f_indexing

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Call BGO_solve_reverse_with_h
    call f_bgo_solve_reverse_with_h(fcontrol, finform, fdata, eval_status, x, f, g, val, u, v)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
    
end subroutine bgo_solve_reverse_with_h

subroutine bgo_solve_reverse_without_h(ccontrol, cinform, cdata, eval_status, n, x, f, g, u, v, &
                                       index_nz_v, nnz_v, index_nz_u, nnz_u) bind(C)
    use GALAHAD_BGO_double_ciface
    implicit none

    integer(C_INT), intent(in), value :: n, nnz_u
    integer(C_INT), intent(inout) :: eval_status
    integer(C_INT), intent(out) :: nnz_v
    real(wp), intent(in), value :: f
    real(wp), intent(inout), dimension(n) :: x, g, u, v
    integer(C_INT), intent(in), dimension(n) :: index_nz_u
    integer(C_INT), intent(out), dimension(n) :: index_nz_v

    type(bgo_control_type), intent(in) :: ccontrol
    type(bgo_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    type(f_bgo_full_data_type), pointer :: fdata
    logical :: f_indexing

    ! Copy control and inform in
    call copy_control_in(ccontrol, fcontrol, f_indexing)
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Call BGO_solve_reverse_without_h
    if(f_indexing) then
        call f_bgo_solve_reverse_without_h(fcontrol, finform, fdata, eval_status, x, f, g, u, v, &
                                           index_nz_v, nnz_v, index_nz_u, nnz_u)
    else
        call f_bgo_solve_reverse_without_h(fcontrol, finform, fdata, eval_status, x, f, g, u, v, &
                                           index_nz_v, nnz_v, index_nz_u+1, nnz_u) 
    end if

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    ! Convert to C indexing if required
    if(.not.f_indexing) then
        index_nz_v = index_nz_v - 1
    endif

end subroutine bgo_solve_reverse_without_h

subroutine bgo_terminate(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_BGO_double_ciface
    implicit none

    type(C_PTR), intent(inout) :: cdata
    type(bgo_control_type), intent(in) :: ccontrol
    type(bgo_inform_type), intent(inout) :: cinform

    type(f_bgo_full_data_type), pointer :: fdata
    type(f_bgo_control_type) :: fcontrol
    type(f_bgo_inform_type) :: finform
    logical :: f_indexing

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol, f_indexing)

    ! Copy inform in
    call copy_inform_in(cinform, finform)

    ! Associate data pointer
    call C_F_POINTER(cdata, fdata)

    ! Call BGO_terminate
    call f_bgo_terminate(fdata,fcontrol,finform)

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    ! Deallocate data
    deallocate(fdata); cdata = C_NULL_PTR 
end subroutine bgo_terminate
