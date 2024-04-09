! THIS VERSION: GALAHAD 5.0 - 2024-03-27 AT 09:10 GMT.

#include "hsl_subset.h"
#include "hsl_subset_ciface.h"

!-*-*-  G A L A H A D  -  D U M M Y   M A 5 7 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma57_real_ciface
  use hsl_kinds_real, only: ipc_, rpc_, lp_, longc_
  use hsl_ma57_real, only:                                       &
       f_ma57_factors => ma57_factors,                           &
       f_ma57_control => ma57_control,                           &
       f_ma57_ainfo => ma57_ainfo,                               &
       f_ma57_sinfo => ma57_sinfo,                               &
       f_ma57_finfo => ma57_finfo,                               &
       f_ma57_initialize => ma57_initialize,                     &
       f_ma57_analyse => ma57_analyse,                           &
       f_ma57_factorize => ma57_factorize,                       &
       f_ma57_solve => ma57_solve,                               &
       f_ma57_finalize => ma57_finalize,                         &
       f_ma57_enquire => ma57_enquire,                           &
       f_ma57_alter_d => ma57_alter_d,                           &
       f_ma57_part_solve => ma57_part_solve,                     &
       f_ma57_sparse_lsolve => ma57_sparse_lsolve,               &
       f_ma57_fredholm_alternative => ma57_fredholm_alternative, &
       f_ma57_lmultiply => ma57_lmultiply,                       &
       f_ma57_get_factors => ma57_get_factors,                   &
       f_ma57_get_n__ => ma57_get_n__

  type, bind(C) :: ma57_control
     integer(ipc_) :: f_arrays
     real(rpc_)       :: multiplier
     real(rpc_)       :: reduce
     real(rpc_)       :: u
     real(rpc_)       :: static_tolerance
     real(rpc_)       :: static_level
     real(rpc_)       :: tolerance
     real(rpc_)       :: convergence
     real(rpc_)       :: consist
     integer(ipc_) :: lp
     integer(ipc_) :: wp
     integer(ipc_) :: mp
     integer(ipc_) :: sp
     integer(ipc_) :: ldiag
     integer(ipc_) :: nemin
     integer(ipc_) :: factorblocking
     integer(ipc_) :: solveblocking
     integer(ipc_) :: la
     integer(ipc_) :: liw
     integer(ipc_) :: maxla
     integer(ipc_) :: maxliw
     integer(ipc_) :: pivoting
     integer(ipc_) :: thresh
     integer(ipc_) :: ordering
     integer(ipc_) :: scaling
     integer(ipc_) :: rank_deficient

     integer(ipc_) :: ispare(5)
     real(rpc_) :: rspare(10)
  end type ma57_control

  type, bind(C) :: ma57_ainfo
     real(rpc_)       :: opsa
     real(rpc_)       :: opse
     integer(ipc_) :: flag
     integer(ipc_) :: more
     integer(ipc_) :: nsteps
     integer(ipc_) :: nrltot
     integer(ipc_) :: nirtot
     integer(ipc_) :: nrlnec
     integer(ipc_) :: nirnec
     integer(ipc_) :: nrladu
     integer(ipc_) :: niradu
     integer(ipc_) :: ncmpa
     integer(ipc_) :: ordering
     integer(ipc_) :: oor
     integer(ipc_) :: dup
     integer(ipc_) :: maxfrt
     integer(ipc_) :: stat

     integer(ipc_) :: ispare(5)
     real(rpc_) :: rspare(10)
  end type ma57_ainfo

  type, bind(C) :: ma57_finfo
     real(rpc_)       :: opsa
     real(rpc_)       :: opse
     real(rpc_)       :: opsb
     real(rpc_)       :: maxchange
     real(rpc_)       :: smin
     real(rpc_)       :: smax
     integer(ipc_) :: flag
     integer(ipc_) :: more
     integer(ipc_) :: maxfrt
     integer(ipc_) :: nebdu
     integer(ipc_) :: nrlbdu
     integer(ipc_) :: nirbdu
     integer(ipc_) :: nrltot
     integer(ipc_) :: nirtot
     integer(ipc_) :: nrlnec
     integer(ipc_) :: nirnec
     integer(ipc_) :: ncmpbr
     integer(ipc_) :: ncmpbi
     integer(ipc_) :: ntwo
     integer(ipc_) :: neig
     integer(ipc_) :: delay
     integer(ipc_) :: signc
     integer(ipc_) :: static_
     integer(ipc_) :: modstep
     integer(ipc_) :: rank
     integer(ipc_) :: stat

     integer(ipc_) :: ispare(5)
     real(rpc_) :: rspare(10)
  end type ma57_finfo

  type, bind(C) :: ma57_sinfo
     real(rpc_)       :: cond
     real(rpc_)       :: cond2
     real(rpc_)       :: berr
     real(rpc_)       :: berr2
     real(rpc_)       :: error
     integer(ipc_) :: flag
     integer(ipc_) :: stat

     integer(ipc_) :: ispare(5)
     real(rpc_) :: rspare(10)
  end type ma57_sinfo

contains

  subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
    type(ma57_control), intent(in)              :: ccontrol
    type(f_ma57_control), optional, intent(out) :: fcontrol
    logical(lp_), optional, intent(out)         :: f_arrays

    if (present(f_arrays)) f_arrays = (ccontrol%f_arrays.ne.0)

    if (present(fcontrol)) then
       fcontrol%multiplier            = ccontrol%multiplier
       fcontrol%reduce                = ccontrol%reduce
       fcontrol%u                     = ccontrol%u
       fcontrol%static_tolerance      = ccontrol%static_tolerance
       fcontrol%static_level          = ccontrol%static_level
       fcontrol%tolerance             = ccontrol%tolerance
       fcontrol%convergence           = ccontrol%convergence
       fcontrol%consist               = ccontrol%consist
       fcontrol%lp                    = ccontrol%lp
       fcontrol%wp                    = ccontrol%wp
       fcontrol%mp                    = ccontrol%mp
       fcontrol%sp                    = ccontrol%sp
       fcontrol%ldiag                 = ccontrol%ldiag
       fcontrol%nemin                 = ccontrol%nemin
       fcontrol%factorblocking        = ccontrol%factorblocking
       fcontrol%solveblocking         = ccontrol%solveblocking
       fcontrol%la                    = ccontrol%la
       fcontrol%liw                   = ccontrol%liw
       fcontrol%maxla                 = ccontrol%maxla
       fcontrol%maxliw                = ccontrol%maxliw
       fcontrol%pivoting              = ccontrol%pivoting
       fcontrol%thresh                = ccontrol%thresh
       fcontrol%ordering              = ccontrol%ordering
       fcontrol%scaling               = ccontrol%scaling
       fcontrol%rank_deficient        = ccontrol%rank_deficient
    end if
  end subroutine copy_control_in

  subroutine copy_ainfo_out(finfo, cinfo)
    type(f_ma57_ainfo), intent(in)  :: finfo
    type(ma57_ainfo), intent(out) :: cinfo

    cinfo%opsa     = finfo%opsa
    cinfo%opse     = finfo%opse
    cinfo%flag     = finfo%flag
    cinfo%more     = finfo%more
    cinfo%nsteps   = finfo%nsteps
    cinfo%nrltot   = finfo%nrltot
    cinfo%nirtot   = finfo%nirtot
    cinfo%nrlnec   = finfo%nrlnec
    cinfo%nirnec   = finfo%nirnec
    cinfo%nrladu   = finfo%nrladu
    cinfo%niradu   = finfo%niradu
    cinfo%ncmpa    = finfo%ncmpa
    cinfo%ordering = finfo%ordering
    cinfo%oor      = finfo%oor
    cinfo%dup      = finfo%dup
    cinfo%maxfrt   = finfo%maxfrt
    cinfo%stat     = finfo%stat
  end subroutine copy_ainfo_out

  subroutine copy_finfo_out(finfo, cinfo)
    type(f_ma57_finfo), intent(in)  :: finfo
    type(ma57_finfo), intent(out) :: cinfo

    cinfo%opsa      = finfo%opsa
    cinfo%opse      = finfo%opse
    cinfo%opsb      = finfo%opsb
    cinfo%maxchange = finfo%maxchange
    cinfo%smin      = finfo%smin
    cinfo%smax      = finfo%smax
    cinfo%flag      = finfo%flag
    cinfo%more      = finfo%more
    cinfo%maxfrt    = finfo%maxfrt
    cinfo%nebdu     = finfo%nebdu
    cinfo%nrlbdu    = finfo%nrlbdu
    cinfo%nirbdu    = finfo%nirbdu
    cinfo%nrltot    = finfo%nrltot
    cinfo%nirtot    = finfo%nirtot
    cinfo%nrlnec    = finfo%nrlnec
    cinfo%nirnec    = finfo%nirnec
    cinfo%ncmpbr    = finfo%ncmpbr
    cinfo%ncmpbi    = finfo%ncmpbi
    cinfo%ntwo      = finfo%ntwo
    cinfo%neig      = finfo%neig
    cinfo%delay     = finfo%delay
    cinfo%signc     = finfo%signc
    cinfo%static_   = finfo%static
    cinfo%modstep   = finfo%modstep
    cinfo%rank      = finfo%rank
    cinfo%stat      = finfo%stat
  end subroutine copy_finfo_out

  subroutine copy_sinfo_out(finfo, cinfo)
    type(f_ma57_sinfo), intent(in)  :: finfo
    type(ma57_sinfo), intent(out) :: cinfo

    cinfo%cond  = finfo%cond
    cinfo%cond2 = finfo%cond2
    cinfo%berr  = finfo%berr
    cinfo%berr2 = finfo%berr2
    cinfo%error = finfo%error
    cinfo%flag  = finfo%flag
    cinfo%stat  = finfo%stat
  end subroutine copy_sinfo_out

end module hsl_ma57_real_ciface
