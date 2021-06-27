! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-  GALAHAD_LHS C INTERFACE  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Jaroslav Fowkes

!  History -
!   currently in development

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

module GALAHAD_LHS_double_ciface
    use iso_c_binding
    use GALAHAD_LHS_double, only:                  &
        f_lhs_inform_type   => LHS_inform_type,    &
        f_lhs_control_type  => LHS_control_type,   &
        f_lhs_data_type     => LHS_data_type,      &
        f_lhs_initialize    => LHS_initialize,     &
        f_lhs_read_specfile => LHS_read_specfile,  &
        f_lhs_ihs           => LHS_ihs,            &
        f_lhs_get_seed      => LHS_get_seed,       &
        f_lhs_terminate     => LHS_terminate

    implicit none

    integer, parameter :: wp = C_DOUBLE ! double precision
    integer, parameter :: sp = C_FLOAT  ! single precision

    type, bind(C) :: lhs_inform_type
        integer(C_INT) :: status
        integer(C_INT) :: alloc_status
        character(C_CHAR), dimension(81) :: bad_alloc
    end type lhs_inform_type

    type, bind(C) :: lhs_control_type
        integer(C_INT) :: error
        integer(C_INT) :: out
        integer(C_INT) :: print_level
        integer(C_INT) :: duplication
        logical(C_BOOL) :: space_critical
        logical(C_BOOL) :: deallocate_error_fatal
        character(C_CHAR), dimension(31) :: prefix 
    end type lhs_control_type

    interface
        integer(C_SIZE_T) pure function strlen(cstr) bind(C)
            use iso_c_binding
            implicit none
            type(C_PTR), intent(in), value :: cstr
        end function strlen 
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

    subroutine copy_inform_out(finform, cinform) 
        type(f_lhs_inform_type), intent(in) :: finform
        type(lhs_inform_type), intent(out) :: cinform
        integer :: i

        ! Integers
        cinform%status = finform%status
        cinform%alloc_status = finform%alloc_status

        ! Strings
        do i = 1,len(finform%bad_alloc)
            cinform%bad_alloc(i) = finform%bad_alloc(i:i)
        end do
        cinform%bad_alloc(len(finform%bad_alloc) + 1) = C_NULL_CHAR
    end subroutine copy_inform_out

    subroutine copy_control_in(ccontrol, fcontrol) 
        type(lhs_control_type), intent(in) :: ccontrol
        type(f_lhs_control_type), intent(out) :: fcontrol
        integer :: i
        
        ! Integers
        fcontrol%error = ccontrol%error
        fcontrol%out = ccontrol%out
        fcontrol%print_level = ccontrol%print_level
        fcontrol%duplication = ccontrol%duplication

        ! Logicals
        fcontrol%space_critical = ccontrol%space_critical
        fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

        ! Strings
        do i = 1, 31
            if (ccontrol%prefix(i) == C_NULL_CHAR) exit
            fcontrol%prefix(i:i) = ccontrol%prefix(i)
        end do
    end subroutine copy_control_in

    subroutine copy_control_out(fcontrol, ccontrol) 
        type(f_lhs_control_type), intent(in) :: fcontrol
        type(lhs_control_type), intent(out) :: ccontrol
        integer :: i
        
        ! Integers
        ccontrol%error = fcontrol%error
        ccontrol%out = fcontrol%out
        ccontrol%print_level = fcontrol%print_level
        ccontrol%duplication = fcontrol%duplication

        ! Logicals
        ccontrol%space_critical = fcontrol%space_critical
        ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

        ! Strings
        do i = 1,len(fcontrol%prefix)
            ccontrol%prefix(i) = fcontrol%prefix(i:i)
        end do
        ccontrol%prefix(len(fcontrol%prefix) + 1) = C_NULL_CHAR
    end subroutine copy_control_out

end module GALAHAD_LHS_double_ciface

subroutine lhs_initialize(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_LHS_double_ciface
    implicit none

    type(C_PTR), intent(out) :: cdata ! data is a black-box
    type(lhs_control_type), intent(out) :: ccontrol
    type(lhs_inform_type), intent(out) :: cinform

    type(f_lhs_data_type), pointer :: fdata
    type(f_lhs_control_type) :: fcontrol
    type(f_lhs_inform_type) :: finform

    ! Allocate fdata 
    allocate(fdata); cdata = C_LOC(fdata)

    ! Call LHS_initialize
    call f_lhs_initialize(fdata, fcontrol, finform) 

    ! Copy control out
    call copy_control_out(fcontrol, ccontrol)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine lhs_initialize

subroutine lhs_read_specfile(ccontrol, cspecfile) bind(C)
    use GALAHAD_LHS_double_ciface
    implicit none

    type(lhs_control_type), intent(inout) :: ccontrol
    type(C_PTR), intent(in), value :: cspecfile

    type(f_lhs_control_type) :: fcontrol
    character(kind=C_CHAR, len=strlen(cspecfile)) :: fspecfile

    ! Device unit number for specfile
    integer(C_INT), parameter :: device = 10

    ! Convert C string to Fortran string
    fspecfile = cstr_to_fchar(cspecfile)

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol)
    
    ! Open specfile for reading
    open(unit=device, file=fspecfile)
    
    ! Call LHS_read_specfile
    call f_lhs_read_specfile(fcontrol, device)

    ! Close specfile
    close(device)

    ! Copy control out
    call copy_control_out(fcontrol, ccontrol)
end subroutine lhs_read_specfile

subroutine lhs_ihs(n_dimen, n_points, seed, X, ccontrol, cinform, cdata) bind(C)
    use GALAHAD_LHS_double_ciface
    implicit none

    integer(C_INT), value, intent(in) :: n_dimen, n_points
    integer(C_INT), intent(inout) :: seed
    integer(C_INT), dimension(n_dimen,n_points) :: X

    type(lhs_control_type), intent(in) :: ccontrol
    type(lhs_inform_type), intent(inout) :: cinform
    type(C_PTR), intent(inout) :: cdata

    type(f_lhs_control_type) :: fcontrol
    type(f_lhs_inform_type) :: finform
    type(f_lhs_data_type), pointer :: fdata

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol)

    ! Associate data pointers
    call C_F_POINTER(cdata, fdata)

    ! Call lhs_irhs
    call f_lhs_ihs(n_dimen, n_points, seed, X, fcontrol, finform, fdata)

    ! Copy inform out
    call copy_inform_out(finform, cinform)
end subroutine lhs_ihs

subroutine lhs_get_seed(seed) bind(C)
    use GALAHAD_LHS_double_ciface
    implicit none

    integer(C_INT), intent(out) :: seed

    ! Call LHS_get_seed    
    call f_lhs_get_seed(seed)

end subroutine lhs_get_seed

subroutine lhs_terminate(cdata, ccontrol, cinform) bind(C) 
    use GALAHAD_LHS_double_ciface
    implicit none

    type(C_PTR), intent(inout) :: cdata
    type(lhs_control_type), intent(in) :: ccontrol
    type(lhs_inform_type), intent(out) :: cinform

    type(f_lhs_control_type) :: fcontrol
    type(f_lhs_inform_type) :: finform
    type(f_lhs_data_type), pointer :: fdata

    ! Copy control in
    call copy_control_in(ccontrol, fcontrol)

    ! Associate data pointers
    call C_F_POINTER(cdata, fdata)

    ! Call LHS_terminate
    call f_lhs_terminate(fdata,fcontrol,finform)

    ! Copy inform out
    call copy_inform_out(finform, cinform)

    ! Deallocate fdata
    deallocate(fdata); cdata = C_NULL_PTR 
end subroutine lhs_terminate
