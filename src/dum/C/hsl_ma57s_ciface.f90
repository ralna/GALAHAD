! THIS VERSION: 2022-11-23 AT 12:30:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M A 5 7 _ C I F A C E   M O D U L E  -*-*-

module hsl_ma57_single_ciface
  use iso_c_binding
  use hsl_zd11_single, only: zd11_type
  use hsl_ma57_single, only:                                     &
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

  integer, parameter :: wp = C_FLOAT ! pkg type
  integer, parameter :: rp = C_FLOAT ! real type

  type, bind(C) :: ma57_control
     integer(C_INT) :: f_arrays
     real(rp)       :: multiplier
     real(rp)       :: reduce
     real(rp)       :: u
     real(rp)       :: static_tolerance
     real(rp)       :: static_level
     real(rp)       :: tolerance
     real(rp)       :: convergence
     real(rp)       :: consist
     integer(C_INT) :: lp
     integer(C_INT) :: wp
     integer(C_INT) :: mp
     integer(C_INT) :: sp
     integer(C_INT) :: ldiag
     integer(C_INT) :: nemin
     integer(C_INT) :: factorblocking
     integer(C_INT) :: solveblocking
     integer(C_INT) :: la
     integer(C_INT) :: liw
     integer(C_INT) :: maxla
     integer(C_INT) :: maxliw
     integer(C_INT) :: pivoting
     integer(C_INT) :: thresh
     integer(C_INT) :: ordering
     integer(C_INT) :: scaling
     integer(C_INT) :: rank_deficient

     integer(C_INT) :: ispare(5)
     real(rp) :: rspare(10)
  end type ma57_control

  type, bind(C) :: ma57_ainfo
     real(rp)       :: opsa
     real(rp)       :: opse
     integer(C_INT) :: flag
     integer(C_INT) :: more
     integer(C_INT) :: nsteps
     integer(C_INT) :: nrltot
     integer(C_INT) :: nirtot
     integer(C_INT) :: nrlnec
     integer(C_INT) :: nirnec
     integer(C_INT) :: nrladu
     integer(C_INT) :: niradu
     integer(C_INT) :: ncmpa
     integer(C_INT) :: ordering
     integer(C_INT) :: oor
     integer(C_INT) :: dup
     integer(C_INT) :: maxfrt
     integer(C_INT) :: stat

     integer(C_INT) :: ispare(5)
     real(rp) :: rspare(10)
  end type ma57_ainfo

  type, bind(C) :: ma57_finfo
     real(rp)       :: opsa
     real(rp)       :: opse
     real(rp)       :: opsb
     real(rp)       :: maxchange
     real(rp)       :: smin
     real(rp)       :: smax
     integer(C_INT) :: flag
     integer(C_INT) :: more
     integer(C_INT) :: maxfrt
     integer(C_INT) :: nebdu
     integer(C_INT) :: nrlbdu
     integer(C_INT) :: nirbdu
     integer(C_INT) :: nrltot
     integer(C_INT) :: nirtot
     integer(C_INT) :: nrlnec
     integer(C_INT) :: nirnec
     integer(C_INT) :: ncmpbr
     integer(C_INT) :: ncmpbi
     integer(C_INT) :: ntwo
     integer(C_INT) :: neig
     integer(C_INT) :: delay
     integer(C_INT) :: signc
     integer(C_INT) :: static_
     integer(C_INT) :: modstep
     integer(C_INT) :: rank
     integer(C_INT) :: stat

     integer(C_INT) :: ispare(5)
     real(rp) :: rspare(10)
  end type ma57_finfo

  type, bind(C) :: ma57_sinfo
     real(rp)       :: cond
     real(rp)       :: cond2
     real(rp)       :: berr
     real(rp)       :: berr2
     real(rp)       :: error
     integer(C_INT) :: flag
     integer(C_INT) :: stat

     integer(C_INT) :: ispare(5)
     real(rp) :: rspare(10)
  end type ma57_sinfo

contains

  subroutine copy_control_in(ccontrol, fcontrol, f_arrays)
    type(ma57_control), intent(in)              :: ccontrol
    type(f_ma57_control), optional, intent(out) :: fcontrol
    logical, optional, intent(out)              :: f_arrays

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

end module hsl_ma57_single_ciface
