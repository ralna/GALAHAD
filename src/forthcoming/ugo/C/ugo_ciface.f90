! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-  GALAHAD_UGO C INTERFACE  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Jaroslav Fowkes

!  History -
!   currently in development

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

module GALAHAD_UGO_double_ciface
    use iso_c_binding
    use GALAHAD_UGO_double, only:                  &
        f_ugo_time_type     => UGO_time_type,      &
        f_ugo_inform_type   => UGO_inform_type,    &
        f_ugo_control_type  => UGO_control_type,   &
        f_ugo_data_type     => UGO_data_type,      &
        f_ugo_initialize    => UGO_initialize,     &
        f_ugo_read_specfile => UGO_read_specfile,  &
        f_ugo_solve         => UGO_solve,          &
        f_ugo_terminate     => UGO_terminate
    use GALAHAD_NLPT_double, only:                 &
        f_nlpt_userdata_type => NLPT_userdata_type

    implicit none

    integer, parameter :: wp = C_DOUBLE ! double precision
    integer, parameter :: sp = C_FLOAT  ! single precision

    type, bind(C) :: ugo_time_type
        real(sp) :: total
        real(wp) :: clock_total 
    end type ugo_time_type

    type, bind(C) :: ugo_inform_type
        integer(C_INT) :: status
        integer(C_INT) :: eval_status
        integer(C_INT) :: alloc_status
        character(C_CHAR), dimension(81) :: bad_alloc
        integer(C_INT) :: iter
        integer(C_INT) :: f_eval
        integer(C_INT) :: g_eval
        integer(C_INT) :: h_eval 
        type(ugo_time_type) :: time
    end type ugo_inform_type

    type, bind(C) :: ugo_control_type
        integer(C_INT) :: error
        integer(C_INT) :: out
        integer(C_INT) :: print_level
        integer(C_INT) :: start_print
        integer(C_INT) :: stop_print
        integer(C_INT) :: print_gap
        integer(C_INT) :: maxit
        integer(C_INT) :: initial_points
        integer(C_INT) :: storage_increment 
        integer(C_INT) :: buffer
        integer(C_INT) :: lipschitz_estimate_used
        integer(C_INT) :: next_interval_selection 
        integer(C_INT) :: refine_with_newton 
        integer(C_INT) :: alive_unit 
        character(C_CHAR), dimension(31) :: alive_file
        real(wp) :: stop_length 
        real(wp) :: small_g_for_newton 
        real(wp) :: small_g 
        real(wp) :: obj_sufficient 
        real(wp) :: global_lipschitz_constant 
        real(wp) :: reliability_parameter 
        real(wp) :: lipschitz_lower_bound 
        real(wp) :: cpu_time_limit 
        real(wp) :: clock_time_limit 
        logical(C_BOOL) :: space_critical
        logical(C_BOOL) :: deallocate_error_fatal
        character(C_CHAR), dimension(31) :: prefix 
    end type ugo_control_type

    interface
        integer(C_SIZE_T) pure function strlen(cstr) bind(C)
            use iso_c_binding
            implicit none
            type(C_PTR), intent(in), value :: cstr
        end function strlen 
    end interface

    abstract interface
        function eval_fgh(x, f, g, h, userdata) result(status)
            use iso_c_binding
            import :: wp
            
            real(wp), intent(in), value :: x
            real(wp), intent(out) :: f, g, h
            type(C_PTR), intent(in), value :: userdata
            integer(C_INT) :: status
        end function eval_fgh
    end interface

contains

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

    subroutine copy_inform_in(cinform, finform) 
        type(ugo_inform_type), intent(in) :: cinform
        type(f_ugo_inform_type), intent(out) :: finform
        integer :: i

        ! Integers
        finform%status = cinform%status
        finform%alloc_status = cinform%alloc_status
        finform%iter = cinform%iter
        finform%f_eval = cinform%f_eval
        finform%g_eval = cinform%g_eval
        finform%h_eval = cinform%h_eval
        
        ! Time derived type
        finform%time%total = cinform%time%total
        finform%time%clock_total = cinform%time%clock_total

        ! Strings
        do i = 1, 81
            if (cinform%bad_alloc(i) == C_NULL_CHAR) exit
            finform%bad_alloc(i:i) = cinform%bad_alloc(i)
        end do
    end subroutine copy_inform_in

    subroutine copy_inform_out(finform, cinform) 
        type(f_ugo_inform_type), intent(in) :: finform
        type(ugo_inform_type), intent(out) :: cinform
        integer :: i

        ! Integers
        cinform%status = finform%status
        cinform%alloc_status = finform%alloc_status
        cinform%iter = finform%iter
        cinform%f_eval = finform%f_eval
        cinform%g_eval = finform%g_eval
        cinform%h_eval = finform%h_eval
        
        ! Time derived type
        cinform%time%total = finform%time%total
        cinform%time%clock_total = finform%time%clock_total

        ! Strings
        do i = 1,len(finform%bad_alloc)
            cinform%bad_alloc(i) = finform%bad_alloc(i:i)
        end do
        cinform%bad_alloc(len(finform%bad_alloc) + 1) = C_NULL_CHAR
    end subroutine copy_inform_out

    subroutine copy_control_in(ccontrol, fcontrol) 
        type(ugo_control_type), intent(in) :: ccontrol
        type(f_ugo_control_type), intent(out) :: fcontrol
        integer :: i
        
        ! Integers
        fcontrol%error = ccontrol%error
        fcontrol%out = ccontrol%out
        fcontrol%print_level = ccontrol%print_level
        fcontrol%start_print = ccontrol%start_print
        fcontrol%stop_print = ccontrol%stop_print
        fcontrol%print_gap = ccontrol%print_gap
        fcontrol%maxit = ccontrol%maxit
        fcontrol%initial_points = ccontrol%initial_points
        fcontrol%storage_increment  = ccontrol%storage_increment 
        fcontrol%buffer = ccontrol%buffer
        fcontrol%lipschitz_estimate_used = ccontrol%lipschitz_estimate_used
        fcontrol%next_interval_selection = ccontrol%next_interval_selection
        fcontrol%refine_with_newton = ccontrol%refine_with_newton
        fcontrol%alive_unit = ccontrol%alive_unit

        ! Doubles
        fcontrol%stop_length = ccontrol%stop_length 
        fcontrol%small_g_for_newton = ccontrol%small_g_for_newton
        fcontrol%small_g = ccontrol%small_g
        fcontrol%obj_sufficient = ccontrol%obj_sufficient
        fcontrol%global_lipschitz_constant = ccontrol%global_lipschitz_constant
        fcontrol%reliability_parameter = ccontrol%reliability_parameter
        fcontrol%lipschitz_lower_bound = ccontrol%lipschitz_lower_bound
        fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
        fcontrol%clock_time_limit = ccontrol%clock_time_limit 

        ! Logicals
        fcontrol%space_critical = ccontrol%space_critical
        fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

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

    subroutine copy_control_out(fcontrol, ccontrol) 
        type(f_ugo_control_type), intent(in) :: fcontrol
        type(ugo_control_type), intent(out) :: ccontrol
        integer :: i
        
        ! Integers
        ccontrol%error = fcontrol%error
        ccontrol%out = fcontrol%out
        ccontrol%print_level = fcontrol%print_level
        ccontrol%start_print = fcontrol%start_print
        ccontrol%stop_print = fcontrol%stop_print
        ccontrol%print_gap = fcontrol%print_gap
        ccontrol%maxit = fcontrol%maxit
        ccontrol%initial_points = fcontrol%initial_points
        ccontrol%storage_increment  = fcontrol%storage_increment 
        ccontrol%buffer = fcontrol%buffer
        ccontrol%lipschitz_estimate_used = fcontrol%lipschitz_estimate_used
        ccontrol%next_interval_selection = fcontrol%next_interval_selection
        ccontrol%refine_with_newton = fcontrol%refine_with_newton
        ccontrol%alive_unit = fcontrol%alive_unit

        ! Doubles
        ccontrol%stop_length = fcontrol%stop_length 
        ccontrol%small_g_for_newton = fcontrol%small_g_for_newton
        ccontrol%small_g = fcontrol%small_g
        ccontrol%obj_sufficient = fcontrol%obj_sufficient
        ccontrol%global_lipschitz_constant = fcontrol%global_lipschitz_constant
        ccontrol%reliability_parameter = fcontrol%reliability_parameter
        ccontrol%lipschitz_lower_bound = fcontrol%lipschitz_lower_bound
        ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
        ccontrol%clock_time_limit = fcontrol%clock_time_limit

        ! Logicals
        ccontrol%space_critical = fcontrol%space_critical
        ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

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

end module GALAHAD_UGO_double_ciface

subroutine ugo_initialize(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_UGO_double_ciface
    implicit none

    type(C_PTR), intent(out) :: cdata ! data is a black-box
    type(ugo_control_type), intent(out) :: ccontrol
    type(ugo_inform_type), intent(out) :: cinform

    type(f_ugo_data_type), pointer :: fdata
    type(f_ugo_control_type) :: fcontrol
    type(f_ugo_inform_type) :: finform

    ! Allocate fdata 
    allocate(fdata); cdata = C_LOC(fdata)

    ! Call UGO_initialize
    call f_ugo_initialize(fdata, fcontrol, finform) 

    ! Initialize eval_status (for reverse communication interface)
    cinform%eval_status = 0

    ! Copy control out
    call copy_control_out(fcontrol, ccontrol)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine ugo_initialize

subroutine ugo_read_specfile(ccontrol, cspecfile) bind(C)
    use GALAHAD_UGO_double_ciface
    implicit none

    type(ugo_control_type), intent(inout) :: ccontrol
    type(C_PTR), intent(in), value :: cspecfile

    type(f_ugo_control_type) :: fcontrol
    character(kind=C_CHAR, len=strlen(cspecfile)) :: fspecfile

    ! Device unit number for specfile
    integer(C_INT), parameter :: device = 10

    ! Convert C string to Fortran string
    fspecfile = cstr_to_fchar(cspecfile)

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol)
    
    ! Open specfile for reading
    open(unit=device, file=fspecfile)
    
    ! Call UGO_read_specfile
    call f_ugo_read_specfile(fcontrol, device)

    ! Close specfile
    close(device)

    ! Copy control out
    call copy_control_out(fcontrol, ccontrol)
end subroutine ugo_read_specfile

subroutine ugo_solve(x_l, x_u, x, f, g, h, ccontrol, cinform, cdata, cuserdata, ceval_fgh) bind(C) 
    use GALAHAD_UGO_double_ciface
    implicit none

    real(wp), intent(in), value :: x_l, x_u
    real(wp), intent(inout) :: x, f, g, h

    type(ugo_control_type), intent(in) :: ccontrol
    type(ugo_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata
    type(C_PTR), intent(in), value :: cuserdata
    type(C_FUNPTR), intent(in), value :: ceval_fgh

    type(f_ugo_control_type) :: fcontrol
    type(f_ugo_inform_type) :: finform
    type(f_ugo_data_type), pointer :: fdata
    procedure(eval_fgh), pointer :: feval_fgh

    ! Ignore Fortran userdata type (not interoperable)
    type(f_nlpt_userdata_type), pointer :: fuserdata => null()

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol)

    ! Copy inform in
    call copy_inform_in(cinform, finform)

    ! Associate data pointers
    call C_F_POINTER(cdata, fdata)

    ! Set eval_status (for reverse communication interface)
    fdata%eval_status = cinform%eval_status

    ! Associate eval_fgh procedure pointers 
    if(C_ASSOCIATED(ceval_fgh)) then ! if ceval_fgh is not NULL 
        call C_F_PROCPOINTER(ceval_fgh, feval_fgh)
    else ! otherwise nullify feval_fgh
        nullify(feval_fgh)
    endif  

    if(associated(feval_fgh)) then ! if eval_fgh is passed as not NULL
        ! Call UGO_solve forward communication version
        call f_ugo_solve(x_l, x_u, x, f, g, h, fcontrol, finform, fdata, fuserdata, wrap_eval_fgh)
    else
        ! Call UGO_solve reverse communication version
        call f_ugo_solve(x_l, x_u, x, f, g, h, fcontrol, finform, fdata, fuserdata)
    end if

    ! Copy inform out
    call copy_inform_out(finform, cinform)

contains

    ! eval_FGH wrapper
    subroutine wrap_eval_fgh(status, x, userdata, f, g, h)     
        integer(C_INT), intent(out) :: status
        real(wp), intent(in) :: x
        type(f_nlpt_userdata_type), intent(inout) :: userdata
        real(wp), intent(out) :: f, g, h

        ! Call C interoperable eval_fgh
        status = feval_fgh(x, f, g, h, cuserdata)
    end subroutine wrap_eval_fgh

end subroutine ugo_solve

subroutine ugo_terminate(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_UGO_double_ciface
    implicit none

    type(C_PTR), intent(inout) :: cdata
    type(ugo_control_type), intent(in) :: ccontrol
    type(ugo_inform_type), intent(inout) :: cinform

    type(f_ugo_control_type) :: fcontrol
    type(f_ugo_inform_type) :: finform
    type(f_ugo_data_type), pointer :: fdata

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol)

    ! Copy inform in
    call copy_inform_in(cinform, finform)

    ! Associate data pointers
    call C_F_POINTER(cdata, fdata)

    ! Call UGO_terminate
    call f_ugo_terminate(fdata,fcontrol,finform)

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    ! Deallocate fdata
    deallocate(fdata); cdata = C_NULL_PTR 
end subroutine ugo_terminate   
