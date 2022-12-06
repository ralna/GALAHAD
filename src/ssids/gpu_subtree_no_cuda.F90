! This is dummy file compiled when there is no CUDA support
module spral_ssids_gpu_subtree
  use, intrinsic :: iso_c_binding
  use spral_ssids_contrib, only : contrib_type
  use spral_ssids_datatypes
  use spral_ssids_inform, only : ssids_inform
  use spral_ssids_subtree, only : symbolic_subtree_base, numeric_subtree_base
  implicit none

  private
  public :: gpu_symbolic_subtree, construct_gpu_symbolic_subtree
  public :: gpu_numeric_subtree, gpu_free_contrib

  type, extends(symbolic_subtree_base) :: gpu_symbolic_subtree
     integer(long) :: dummy
   contains
     procedure :: factor
     procedure :: cleanup => symbolic_cleanup
  end type gpu_symbolic_subtree

  type, extends(numeric_subtree_base) :: gpu_numeric_subtree
     real(wp) :: dummy !< Just so we can perform dummy ops to prevent warnings
   contains
     procedure :: get_contrib
     procedure :: solve_fwd
     procedure :: solve_diag
     procedure :: solve_diag_bwd
     procedure :: solve_bwd
     procedure :: enquire_posdef
     procedure :: enquire_indef
     procedure :: alter
     procedure :: cleanup => numeric_cleanup
  end type gpu_numeric_subtree

contains

  function construct_gpu_symbolic_subtree(device, n, sa, en, sptr, sparent, &
       rptr, rlist, nptr, nlist, options) result(this)
    implicit none
    class(gpu_symbolic_subtree), pointer :: this
    integer, intent(in) :: device
    integer, intent(in) :: n
    integer, intent(in) :: sa
    integer, intent(in) :: en
    integer, dimension(*), target, intent(in) :: sptr
    integer, dimension(*), target, intent(in) :: sparent
    integer(long), dimension(*), target, intent(in) :: rptr
    integer, dimension(*), target, intent(in) :: rlist
    integer(long), dimension(*), target, intent(in) :: nptr
    integer(long), dimension(2,*), target, intent(in) :: nlist
    class(ssids_options), intent(in) :: options

    nullify(this)

    print *, "construct_gpu_symbolic_subtree() called without GPU support."
    print *, "This should never happen."
    stop -1

    ! Dummy operations to prevent warnings
    this%dummy = device+n+sa+en+sptr(1)+sparent(1)+rptr(1)+rlist(1)+nptr(1)+&
         nlist(1,1)+options%print_level
  end function construct_gpu_symbolic_subtree

  subroutine symbolic_cleanup(this)
    implicit none
    class(gpu_symbolic_subtree), intent(inout) :: this

    ! Dummy operation to prevent warnings
    this%dummy = 0
  end subroutine symbolic_cleanup

  function factor(this, posdef, aval, child_contrib, options, inform, scaling)
    implicit none
    class(numeric_subtree_base), pointer :: factor
    class(gpu_symbolic_subtree), target, intent(inout) :: this
    logical, intent(in) :: posdef
    real(wp), dimension(*), target, intent(in) :: aval
    type(contrib_type), dimension(:), target, intent(inout) :: child_contrib
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform
    real(wp), dimension(*), target, optional, intent(in) :: scaling

    type(gpu_numeric_subtree), pointer :: subtree

    nullify(subtree)
    ! Dummy operations to prevent warnings
    factor => subtree
    if (posdef) &
         subtree%dummy = real(this%dummy,wp)+aval(1)+child_contrib(1)%val(1)+&
         options%gpu_perf_coeff
    if (present(scaling)) subtree%dummy = subtree%dummy * scaling(1)
    inform%flag = SSIDS_ERROR_UNKNOWN
  end function factor

  subroutine numeric_cleanup(this)
    implicit none
    class(gpu_numeric_subtree), intent(inout) :: this

    ! Dummy operations to prevent warnings
    this%dummy = 0
  end subroutine numeric_cleanup

  function get_contrib(this)
    implicit none
    type(contrib_type) :: get_contrib
    class(gpu_numeric_subtree), intent(in) :: this

    ! Dummy operation to prevent warnings
    get_contrib%n = int(this%dummy)
  end function get_contrib

  subroutine gpu_free_contrib(contrib)
    implicit none
    type(contrib_type), intent(inout) :: contrib

    ! Dummy operation to prevent warnings
    contrib%n = 0
  end subroutine gpu_free_contrib

  subroutine solve_fwd(this, nrhs, x, ldx, inform)
    implicit none
    class(gpu_numeric_subtree), intent(inout) :: this
    integer, intent(in) :: nrhs
    real(wp), dimension(*), intent(inout) :: x
    integer, intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    ! Dummy operations to prevent warnings
    x(nrhs+1*ldx) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
  end subroutine solve_fwd

  subroutine solve_diag(this, nrhs, x, ldx, inform)
    implicit none
    class(gpu_numeric_subtree), intent(inout) :: this
    integer, intent(in) :: nrhs
    real(wp), dimension(*), intent(inout) :: x
    integer, intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    ! Dummy operations to prevent warnings
    x(nrhs+1*ldx) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
  end subroutine solve_diag

  subroutine solve_diag_bwd(this, nrhs, x, ldx, inform)
    implicit none
    class(gpu_numeric_subtree), intent(inout) :: this
    integer, intent(in) :: nrhs
    real(wp), dimension(*), intent(inout) :: x
    integer, intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    ! Dummy operations to prevent warnings
    x(nrhs+1*ldx) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
  end subroutine solve_diag_bwd

  subroutine solve_bwd(this, nrhs, x, ldx, inform)
    implicit none
    class(gpu_numeric_subtree), intent(inout) :: this
    integer, intent(in) :: nrhs
    real(wp), dimension(*), intent(inout) :: x
    integer, intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    ! Dummy operations to prevent warnings
    x(nrhs+1*ldx) = this%dummy
    inform%flag = SSIDS_ERROR_UNKNOWN
  end subroutine solve_bwd

  subroutine enquire_posdef(this, d)
    implicit none
    class(gpu_numeric_subtree), target, intent(in) :: this
    real(wp), dimension(*), target, intent(out) :: d

    ! Dummy operation to prevent warnings
    d(1) = this%dummy
  end subroutine enquire_posdef

  subroutine enquire_indef(this, piv_order, d)
    implicit none
    class(gpu_numeric_subtree), target, intent(in) :: this
    integer, dimension(*), target, optional, intent(out) :: piv_order
    real(wp), dimension(2,*), target, optional, intent(out) :: d

    ! Dummy operation to prevent warnings
    if (present(d)) d(1,1) = this%dummy
    if (present(piv_order)) piv_order(1) = 1
  end subroutine enquire_indef

  subroutine alter(this, d)
    implicit none
    class(gpu_numeric_subtree), target, intent(inout) :: this
    real(wp), dimension(2,*), intent(in) :: d

    ! Dummy operation to prevent warnings
    this%dummy = d(1,1)
  end subroutine alter

end module spral_ssids_gpu_subtree
