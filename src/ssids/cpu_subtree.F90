! THIS VERSION: GALAHAD 5.1 - 2024-11-26 AT 12:50 GMT.

#include "spral_procedures.h"

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_cpu_subtree_precision
  use spral_kinds_precision
  use spral_ssids_contrib_precision, only : contrib_type
  use spral_ssids_cpu_iface_precision ! fixme only
  use spral_ssids_types_precision
  use spral_ssids_inform_precision, only : ssids_inform
  use spral_ssids_subtree_precision, only : symbolic_subtree_base, &
                                            numeric_subtree_base
  implicit none

  private
  public :: cpu_symbolic_subtree, construct_cpu_symbolic_subtree
  public :: cpu_numeric_subtree, cpu_free_contrib

  type, extends(symbolic_subtree_base) :: cpu_symbolic_subtree
     integer(ip_) :: n
     type(C_PTR) :: csubtree
   contains
     procedure :: factor
     procedure :: cleanup => symbolic_cleanup
  end type cpu_symbolic_subtree

  type, extends(numeric_subtree_base) :: cpu_numeric_subtree
     logical(C_BOOL) :: posdef
     type(cpu_symbolic_subtree), pointer :: symbolic
     type(C_PTR) :: csubtree
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
  end type cpu_numeric_subtree

  interface
     type(C_PTR) function c_create_symbolic_subtree(n, sa, en, sptr, sparent, &
          rptr, rlist, nptr, nlist, ncontrib, contrib_idx, options) &
          bind(C, name="spral_ssids_cpu_create_symbolic_subtree")
       use spral_kinds
       import :: cpu_factor_options
       implicit none
       integer(C_IP_), value :: n
       integer(C_IP_), value :: sa
       integer(C_IP_), value :: en
       integer(C_IP_), dimension(*), intent(in) :: sptr
       integer(C_IP_), dimension(*), intent(in) :: sparent
       integer(CLONG_), dimension(*), intent(in) :: rptr
       integer(C_IP_), dimension(*), intent(in) :: rlist
       integer(CLONG_), dimension(*), intent(in) :: nptr
       integer(CLONG_), dimension(*), intent(in) :: nlist
       integer(C_IP_), value :: ncontrib
       integer(C_IP_), dimension(*), intent(in) :: contrib_idx
       type(cpu_factor_options), intent(in) :: options
     end function c_create_symbolic_subtree

     subroutine c_destroy_symbolic_subtree(subtree) &
          bind(C, name="spral_ssids_cpu_destroy_symbolic_subtree")
       use spral_kinds
       implicit none
       type(C_PTR), value :: subtree
     end subroutine c_destroy_symbolic_subtree
  end interface

#ifdef REAL_32
  interface
     type(C_PTR) function c_create_numeric_subtree(posdef, symbolic_subtree, &
          aval, scaling, child_contrib, options, stats) &
          bind(C, name="spral_ssids_cpu_create_num_subtree_sgl")
       use spral_kinds_precision
       import :: cpu_factor_options, cpu_factor_stats
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: symbolic_subtree
       real(C_RP_), dimension(*), intent(in) :: aval
       type(C_PTR), value :: scaling
       type(C_PTR), dimension(*), intent(inout) :: child_contrib
       type(cpu_factor_options), intent(in) :: options
       type(cpu_factor_stats), intent(out) :: stats
     end function c_create_numeric_subtree

     subroutine c_destroy_numeric_subtree(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_destroy_num_subtree_sgl")
       use spral_kinds
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_destroy_numeric_subtree

     integer(C_IP_) function c_subtree_solve_fwd(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_fwd_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_fwd

     integer(C_IP_) function c_subtree_solve_diag(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_diag_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_diag

     integer(C_IP_) function c_subtree_solve_diag_bwd(posdef, subtree, nrhs, &
          x, ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_diag_bwd_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_diag_bwd

     integer(C_IP_) function c_subtree_solve_bwd(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_bwd_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_bwd

     subroutine c_subtree_enquire(posdef, subtree, piv_order, d) &
          bind(C, name="spral_ssids_cpu_subtree_enquire_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       type(C_PTR), value :: piv_order
       type(C_PTR), value :: d
     end subroutine c_subtree_enquire

     subroutine c_subtree_alter(posdef, subtree, d) &
          bind(C, name="spral_ssids_cpu_subtree_alter_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       real(C_RP_), dimension(*), intent(in) :: d
     end subroutine c_subtree_alter

     subroutine c_get_contrib(posdef, subtree, n, val, ldval, rlist, ndelay, &
          delay_perm, delay_val, lddelay) &
          bind(C, name="spral_ssids_cpu_subtree_get_contrib_sgl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_) :: n
       type(C_PTR) :: val
       integer(C_IP_) :: ldval
       type(C_PTR) :: rlist
       integer(C_IP_) :: ndelay
       type(C_PTR) :: delay_perm
       type(C_PTR) :: delay_val
       integer(C_IP_) :: lddelay
     end subroutine c_get_contrib

     subroutine c_free_contrib(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_subtree_free_contrib_sgl")
       use spral_kinds
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_free_contrib
  end interface

#elif REAL_128
  interface
     type(C_PTR) function c_create_numeric_subtree(posdef, symbolic_subtree, &
          aval, scaling, child_contrib, options, stats) &
          bind(C, name="spral_ssids_cpu_create_num_subtree_qul")
       use spral_kinds_precision
       import :: cpu_factor_options, cpu_factor_stats
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: symbolic_subtree
       real(C_RP_), dimension(*), intent(in) :: aval
       type(C_PTR), value :: scaling
       type(C_PTR), dimension(*), intent(inout) :: child_contrib
       type(cpu_factor_options), intent(in) :: options
       type(cpu_factor_stats), intent(out) :: stats
     end function c_create_numeric_subtree

     subroutine c_destroy_numeric_subtree(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_destroy_num_subtree_qul")
       use spral_kinds
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_destroy_numeric_subtree

     integer(C_IP_) function c_subtree_solve_fwd(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_fwd_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_fwd

     integer(C_IP_) function c_subtree_solve_diag(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_diag_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_diag

     integer(C_IP_) function c_subtree_solve_diag_bwd(posdef, subtree, nrhs, &
          x, ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_diag_bwd_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_diag_bwd

     integer(C_IP_) function c_subtree_solve_bwd(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_bwd_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_bwd

     subroutine c_subtree_enquire(posdef, subtree, piv_order, d) &
          bind(C, name="spral_ssids_cpu_subtree_enquire_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       type(C_PTR), value :: piv_order
       type(C_PTR), value :: d
     end subroutine c_subtree_enquire

     subroutine c_subtree_alter(posdef, subtree, d) &
          bind(C, name="spral_ssids_cpu_subtree_alter_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       real(C_RP_), dimension(*), intent(in) :: d
     end subroutine c_subtree_alter

     subroutine c_get_contrib(posdef, subtree, n, val, ldval, rlist, ndelay, &
          delay_perm, delay_val, lddelay) &
          bind(C, name="spral_ssids_cpu_subtree_get_contrib_qul")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_) :: n
       type(C_PTR) :: val
       integer(C_IP_) :: ldval
       type(C_PTR) :: rlist
       integer(C_IP_) :: ndelay
       type(C_PTR) :: delay_perm
       type(C_PTR) :: delay_val
       integer(C_IP_) :: lddelay
     end subroutine c_get_contrib

     subroutine c_free_contrib(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_subtree_free_contrib_qul")
       use spral_kinds
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_free_contrib
  end interface

#else

  interface
     type(C_PTR) function c_create_numeric_subtree(posdef, symbolic_subtree, &
          aval, scaling, child_contrib, options, stats) &
          bind(C, name="spral_ssids_cpu_create_num_subtree_dbl")
       use spral_kinds_precision
       import :: cpu_factor_options, cpu_factor_stats
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: symbolic_subtree
       real(C_RP_), dimension(*), intent(in) :: aval
       type(C_PTR), value :: scaling
       type(C_PTR), dimension(*), intent(inout) :: child_contrib
       type(cpu_factor_options), intent(in) :: options
       type(cpu_factor_stats), intent(out) :: stats
     end function c_create_numeric_subtree

     subroutine c_destroy_numeric_subtree(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_destroy_num_subtree_dbl")
       use spral_kinds
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_destroy_numeric_subtree

     integer(C_IP_) function c_subtree_solve_fwd(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_fwd_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_fwd

     integer(C_IP_) function c_subtree_solve_diag(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_diag_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_diag

     integer(C_IP_) function c_subtree_solve_diag_bwd(posdef, subtree, nrhs, &
          x, ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_diag_bwd_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_diag_bwd

     integer(C_IP_) function c_subtree_solve_bwd(posdef, subtree, nrhs, x, &
          ldx) &
          bind(C, name="spral_ssids_cpu_subtree_solve_bwd_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_), value :: nrhs
       real(C_RP_), dimension(*), intent(inout) :: x
       integer(C_IP_), value :: ldx
     end function c_subtree_solve_bwd

     subroutine c_subtree_enquire(posdef, subtree, piv_order, d) &
          bind(C, name="spral_ssids_cpu_subtree_enquire_dbl")
       use spral_kinds
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       type(C_PTR), value :: piv_order
       type(C_PTR), value :: d
     end subroutine c_subtree_enquire

     subroutine c_subtree_alter(posdef, subtree, d) &
          bind(C, name="spral_ssids_cpu_subtree_alter_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       real(C_RP_), dimension(*), intent(in) :: d
     end subroutine c_subtree_alter

     subroutine c_get_contrib(posdef, subtree, n, val, ldval, rlist, ndelay, &
          delay_perm, delay_val, lddelay) &
          bind(C, name="spral_ssids_cpu_subtree_get_contrib_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_IP_) :: n
       type(C_PTR) :: val
       integer(C_IP_) :: ldval
       type(C_PTR) :: rlist
       integer(C_IP_) :: ndelay
       type(C_PTR) :: delay_perm
       type(C_PTR) :: delay_val
       integer(C_IP_) :: lddelay
     end subroutine c_get_contrib

     subroutine c_free_contrib(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_subtree_free_contrib_dbl")
       use spral_kinds_precision
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_free_contrib
  end interface
#endif

contains

  function construct_cpu_symbolic_subtree(n, sa, en, sptr, sparent, rptr, &
       rlist, nptr, nlist, contrib_idx, options) result(this)
    implicit none
    class(cpu_symbolic_subtree), pointer :: this
    integer(ip_), intent(in) :: n
    integer(ip_), intent(in) :: sa
    integer(ip_), intent(in) :: en
    integer(ip_), dimension(*), target, intent(in) :: sptr
    integer(ip_), dimension(*), intent(in) :: sparent
    integer(long_), dimension(*), target, intent(in) :: rptr
    integer(ip_), dimension(*), target, intent(in) :: rlist
    integer(long_), dimension(*), target, intent(in) :: nptr
    integer(long_), dimension(2,*), target, intent(in) :: nlist
    integer(ip_), dimension(:), intent(in) :: contrib_idx
    class(ssids_options), intent(in) :: options

    integer(ip_) :: st
    type(cpu_factor_options) :: coptions

    nullify(this)

    ! Allocate output
    allocate(this, stat=st)
    if (st .ne. 0) return

    ! Store basic details
    this%n = n

    ! Call C++ subtree analyse
    call cpu_copy_options_in(options, coptions)
    this%csubtree = &
        c_create_symbolic_subtree(n, sa, en, sptr, sparent, rptr, rlist, nptr, &
        nlist, int(size(contrib_idx),kind=ip_), contrib_idx, coptions)
  end function construct_cpu_symbolic_subtree

  subroutine symbolic_cleanup(this)
    implicit none
    class(cpu_symbolic_subtree), intent(inout) :: this

    call c_destroy_symbolic_subtree(this%csubtree)
  end subroutine symbolic_cleanup

  function factor(this, posdef, aval, child_contrib, options, inform, scaling)
    implicit none
    class(numeric_subtree_base), pointer :: factor
    class(cpu_symbolic_subtree), target, intent(inout) :: this
    logical, intent(in) :: posdef
    real(rp_), dimension(*), target, intent(in) :: aval
    type(contrib_type), dimension(:), target, intent(inout) :: child_contrib
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform
    real(rp_), dimension(*), target, optional, intent(in) :: scaling

    type(cpu_numeric_subtree), pointer :: cpu_factor
    type(cpu_factor_options) :: coptions
    type(cpu_factor_stats) :: cstats
    type(C_PTR) :: cscaling
    integer(ip_) :: i
    type(C_PTR), dimension(:), allocatable :: contrib_ptr
    integer(ip_) :: st

    ! Leave output as null until successful exit
    nullify(factor)

    ! Allocate cpu_factor for output
    allocate(cpu_factor, stat=st)
    if (st .ne. 0) goto 10
    cpu_factor%symbolic => this

    ! Convert child_contrib to contrib_ptr
    allocate(contrib_ptr(size(child_contrib)), stat=st)
    if (st .ne. 0) goto 10
    do i = 1, size(child_contrib)
       contrib_ptr(i) = C_LOC(child_contrib(i))
    end do

    ! Call C++ factor routine
    cpu_factor%posdef = posdef
    cscaling = C_NULL_PTR
    if (present(scaling)) cscaling = C_LOC(scaling)
    call cpu_copy_options_in(options, coptions)

    cpu_factor%csubtree = &
         c_create_numeric_subtree(cpu_factor%posdef, this%csubtree, &
         aval, cscaling, contrib_ptr, coptions, cstats)
    if (cstats%flag .lt. 0) then
       call c_destroy_numeric_subtree(cpu_factor%posdef, cpu_factor%csubtree)
       deallocate(cpu_factor, stat=st)
       inform%flag = cstats%flag
       return
    end if

    ! Extract to Fortran data structures
    call cpu_copy_stats_out(cstats, inform)

    ! Success, set result and return
    factor => cpu_factor
    return

    ! Allocation error handler
10  continue
    inform%flag = SSIDS_ERROR_ALLOCATION
    inform%stat = st
    deallocate(cpu_factor, stat=st)
    return
  end function factor

  subroutine numeric_cleanup(this)
    implicit none
    class(cpu_numeric_subtree), intent(inout) :: this

    call c_destroy_numeric_subtree(this%posdef, this%csubtree)
  end subroutine numeric_cleanup

  function get_contrib(this)
    implicit none
    type(contrib_type) :: get_contrib
    class(cpu_numeric_subtree), intent(in) :: this

    type(C_PTR) :: cval, crlist, delay_perm, delay_val

    call c_get_contrib(this%posdef, this%csubtree, get_contrib%n, cval,        &
         get_contrib%ldval, crlist, get_contrib%ndelay, delay_perm, delay_val, &
         get_contrib%lddelay)
    call c_f_pointer(cval, get_contrib%val, shape = (/ get_contrib%n**2 /))
    call c_f_pointer(crlist, get_contrib%rlist, shape = (/ get_contrib%n /))
    if (c_associated(delay_val)) then
       call c_f_pointer(delay_perm, get_contrib%delay_perm, &
            shape = (/ get_contrib%ndelay /))
       call c_f_pointer(delay_val, get_contrib%delay_val, &
            shape = (/ get_contrib%ndelay*get_contrib%lddelay /))
    else
       nullify(get_contrib%delay_perm)
       nullify(get_contrib%delay_val)
    end if
    get_contrib%owner = 0 ! cpu
    get_contrib%posdef = this%posdef
    get_contrib%owner_ptr = this%csubtree
  end function get_contrib

  subroutine solve_fwd(this, nrhs, x, ldx, inform)
    implicit none
    class(cpu_numeric_subtree), intent(inout) :: this
    integer(ip_), intent(in) :: nrhs
    real(rp_), dimension(*), intent(inout) :: x
    integer(ip_), intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    integer(C_IP_) :: flag

    flag = c_subtree_solve_fwd(this%posdef, this%csubtree, nrhs, x, ldx)
    if (flag .ne. SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_fwd

  subroutine solve_diag(this, nrhs, x, ldx, inform)
    implicit none
    class(cpu_numeric_subtree), intent(inout) :: this
    integer(ip_), intent(in) :: nrhs
    real(rp_), dimension(*), intent(inout) :: x
    integer(ip_), intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    integer(C_IP_) :: flag

    flag = c_subtree_solve_diag(this%posdef, this%csubtree, nrhs, x, ldx)
    if (flag .ne. SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_diag

  subroutine solve_diag_bwd(this, nrhs, x, ldx, inform)
    implicit none
    class(cpu_numeric_subtree), intent(inout) :: this
    integer(ip_), intent(in) :: nrhs
    real(rp_), dimension(*), intent(inout) :: x
    integer(ip_), intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    integer(C_IP_) :: flag

    flag = c_subtree_solve_diag_bwd(this%posdef, this%csubtree, nrhs, x, ldx)
    if (flag .ne. SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_diag_bwd

  subroutine solve_bwd(this, nrhs, x, ldx, inform)
    implicit none
    class(cpu_numeric_subtree), intent(inout) :: this
    integer(ip_), intent(in) :: nrhs
    real(rp_), dimension(*), intent(inout) :: x
    integer(ip_), intent(in) :: ldx
    type(ssids_inform), intent(inout) :: inform

    integer(C_IP_) :: flag

    flag = c_subtree_solve_bwd(this%posdef, this%csubtree, nrhs, x, ldx)
    if (flag .ne. SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_bwd

  subroutine enquire_posdef(this, d)
    implicit none
    class(cpu_numeric_subtree), intent(in) :: this
    real(rp_), dimension(*), target, intent(out) :: d

    call c_subtree_enquire(this%posdef, this%csubtree, C_NULL_PTR, C_LOC(d))
  end subroutine enquire_posdef

  subroutine enquire_indef(this, piv_order, d)
    implicit none
    class(cpu_numeric_subtree), intent(in) :: this
    integer(ip_), dimension(*), target, optional, intent(out) :: piv_order
    real(rp_), dimension(2,*), target, optional, intent(out) :: d

    type(C_PTR) :: dptr, poptr

    ! Setup pointers
    poptr = C_NULL_PTR
    if (present(piv_order)) poptr = C_LOC(piv_order)
    dptr = C_NULL_PTR
    if (present(d)) dptr = C_LOC(d)

    ! Call C++ routine
    call c_subtree_enquire(this%posdef, this%csubtree, poptr, dptr)
  end subroutine enquire_indef

  subroutine alter(this, d)
    implicit none
    class(cpu_numeric_subtree), target, intent(inout) :: this
    real(rp_), dimension(2,*), intent(in) :: d

    call c_subtree_alter(this%posdef, this%csubtree, d)
  end subroutine alter

  subroutine cpu_free_contrib(posdef, csubtree)
    implicit none
    logical(C_BOOL), intent(in) :: posdef
    type(C_PTR), intent(inout) :: csubtree

    call c_free_contrib(posdef, csubtree)
  end subroutine cpu_free_contrib

end module spral_ssids_cpu_subtree_precision
