!> \file
!> \copyright 2011-2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg and Jennifer Scott
!> \note      Originally based on HSL_MA97 v2.2.0
module spral_ssids
!$  use omp_lib
  use, intrinsic :: iso_c_binding
  use spral_hw_topology, only : guess_topology, numa_region
  use spral_match_order, only : match_order_metis
  use spral_matrix_util, only : SPRAL_MATRIX_REAL_SYM_INDEF, &
                                SPRAL_MATRIX_REAL_SYM_PSDEF, &
                                convert_coord_to_cscl, clean_cscl_oop, &
                                apply_conversion_map
  use spral_metis_wrapper, only : metis_order
  use spral_scaling, only : auction_scale_sym, equilib_scale_sym, &
                            hungarian_scale_sym, &
                            equilib_options, equilib_inform, &
                            hungarian_options, hungarian_inform
  use spral_ssids_anal, only : analyse_phase, check_order, expand_matrix, &
                               expand_pattern
  use spral_ssids_datatypes
  use spral_ssids_akeep, only : ssids_akeep
  use spral_ssids_fkeep, only : ssids_fkeep
  use spral_ssids_inform, only : ssids_inform
  use spral_rutherford_boeing, only : rb_write_options, rb_write
  implicit none

  private
  ! Data types
  public :: ssids_akeep, ssids_fkeep, ssids_options, ssids_inform
  ! User interface routines
  public :: ssids_analyse,         & ! Analyse phase, CSC-lower input
            ssids_analyse_coord,   & ! Analyse phase, Coordinate input
            ssids_factor,          & ! Factorize phase
            ssids_solve,           & ! Solve phase
            ssids_free,            & ! Free akeep and/or fkeep
            ssids_enquire_posdef,  & ! Pivot information in posdef case
            ssids_enquire_indef,   & ! Pivot information in indef case
            ssids_alter              ! Alter diagonal

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> \brief Caches user OpenMP ICV values for later restoration
  type :: omp_settings
     logical :: nested, dynamic
     integer :: max_active_levels
  end type omp_settings

  ! Make interfaces generic.
  interface ssids_analyse
     module procedure analyse_double, analyse_double_ptr32
  end interface ssids_analyse

  interface ssids_analyse_coord
     module procedure ssids_analyse_coord_double
  end interface ssids_analyse_coord

  interface ssids_factor
     module procedure ssids_factor_ptr32_double, ssids_factor_ptr64_double
  end interface ssids_factor

  interface ssids_solve
     module procedure ssids_solve_one_double
     module procedure ssids_solve_mult_double
  end interface ssids_solve

  interface ssids_free
     module procedure free_akeep_double
     module procedure free_fkeep_double
     module procedure free_both_double
  end interface ssids_free

  interface ssids_enquire_posdef
     module procedure ssids_enquire_posdef_double
  end interface ssids_enquire_posdef

  interface ssids_enquire_indef
     module procedure ssids_enquire_indef_double
  end interface ssids_enquire_indef

  interface ssids_alter
     module procedure ssids_alter_double
  end interface ssids_alter

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Analyse phase (32-bit wrapper).
!>
!> This routine provides a wrapper around analyse_double() that copies the
!> 32-bit ptr to a 64-bit array before calling the 64-bit version.
  subroutine analyse_double_ptr32(check, n, ptr, row, akeep, options, inform, &
       order, val, topology)
    implicit none
    logical, intent(in) :: check
    integer, intent(in) :: n
    integer, intent(in) :: ptr(:)
    integer, intent(in) :: row(:)
    type(ssids_akeep), intent(inout) :: akeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    integer, optional, intent(inout) :: order(:)
    real(wp), optional, intent(in) :: val(:)
    type(numa_region), dimension(:), optional, intent(in) :: topology

    integer(long), dimension(:), allocatable :: ptr64

    ! Copy 32-bit ptr to 64-bit version
    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = SSIDS_ERROR_ALLOCATION
       akeep%inform = inform
       call inform%print_flag(options, 'ssids_analyse')
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    ! Call 64-bit version of routine
    call analyse_double(check, n, ptr64, row, akeep, options, inform, &
         order=order, val=val, topology=topology)
  end subroutine analyse_double_ptr32

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Analyse phase.
!>
!> Matrix entered in CSC format (lower triangle).
!> The user optionally inputs the pivot order. If not, metis called. 
!> Structure is then expanded.
!> Supervariables are computed
!> and then the assembly tree is constructed and the data structures
!> required by the factorization are set up.
!> There is no checking of the user's data if check = .false.
!> Otherwise, matrix_util routines are used to clean data.
!>
!> See user documentation for full detail on parameters.
!>
!> @param check Clean matrix data if true (cleaned version stored in akeep).
!> @param n Order of A.
!> @param ptr Column pointers of A.
!> @param row Row indices of A.
!> @param akeep Symbolic factorization out.
!> @param options User-supplied options.
!> @param inform Stats/information returned to user.
!> @param order Return ordering to user / allow user to supply order.
!> @param val Values of A. Only required if matching-based ordering requested.
!> @param topology Specify machine topology to work with.
  subroutine analyse_double(check, n, ptr, row, akeep, options, inform, &
       order, val, topology)
    implicit none
    logical, intent(in) :: check
    integer, intent(in) :: n
    integer(long), intent(in) :: ptr(:)
    integer, intent(in) :: row(:)
    type(ssids_akeep), intent(inout) :: akeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    integer, optional, intent(inout) :: order(:)
    real(wp), optional, intent(in) :: val(:)
    type(numa_region), dimension(:), optional, intent(in) :: topology

    character(50)  :: context      ! Procedure name (used when printing).
    integer :: mu_flag      ! error flag for matrix_util routines
    integer(long) :: nz     ! entries in expanded matrix
    integer :: st           ! stat parameter
    integer :: flag         ! error flag for metis

    integer, dimension(:), allocatable :: order2
    integer(long), dimension(:), allocatable :: ptr2 ! col ptrs for expanded mat
    integer, dimension(:), allocatable :: row2 ! row indices for expanded matrix

    ! The following are only used for matching-based orderings
    real(wp), dimension(:), allocatable :: val_clean ! cleaned values if
      ! val is present and checking is required. 
    real(wp), dimension(:), allocatable :: val2 ! expanded matrix if
      ! val is present.

    integer :: mo_flag
    integer :: free_flag
    type(ssids_inform) :: inform_default

    ! Initialise
    context = 'ssids_analyse'
    inform = inform_default
    call ssids_free(akeep, free_flag)
    if (free_flag .ne. 0) then
       inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
       inform%cuda_error = free_flag
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    ! Print status on entry
    call options%print_summary_analyse(context)
    if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
       write (options%unit_diagnostics,'(a,i15)') &
            ' n                         =  ',n
    end if

    akeep%check = check
    akeep%n = n

    ! Checking of matrix data
    if (n .lt. 0) then
       inform%flag = SSIDS_ERROR_A_N_OOR
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    if (n .eq. 0) then
       akeep%nnodes = 0
       allocate(akeep%sptr(0), stat=st) ! used to check if analyse has been run
       if (st .ne. 0) go to 490
       akeep%inform = inform
       return
    end if

    ! check options%ordering has a valid value
    if ((options%ordering .lt. 0) .or. (options%ordering .gt. 2)) then
       inform%flag = SSIDS_ERROR_ORDER
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    ! check val present when expected
    if (options%ordering .eq. 2) then
       if (.not. present(val)) then
          inform%flag = SSIDS_ERROR_VAL
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end if
    end if

    st = 0
    if (check) then
       allocate (akeep%ptr(n+1),stat=st)
       if (st .ne. 0) go to 490

       if (present(val)) then
          call clean_cscl_oop(SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ptr, row, &
               akeep%ptr, akeep%row, mu_flag, val_in=val, val_out=val_clean, &
               lmap=akeep%lmap, map=akeep%map, &
               noor=inform%matrix_outrange, ndup=inform%matrix_dup)
       else
          call clean_cscl_oop(SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ptr, row, &
               akeep%ptr, akeep%row, mu_flag, lmap=akeep%lmap, map=akeep%map, &
               noor=inform%matrix_outrange, ndup=inform%matrix_dup)
       end if
       ! Check for errors
       if (mu_flag .lt. 0) then
          if (mu_flag .eq. -1) inform%flag  = SSIDS_ERROR_ALLOCATION
          if (mu_flag .eq. -5) inform%flag  = SSIDS_ERROR_A_PTR
          if (mu_flag .eq. -6) inform%flag  = SSIDS_ERROR_A_PTR
          if (mu_flag .eq. -10) inform%flag = SSIDS_ERROR_A_ALL_OOR
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end if

       ! Check whether warning needs to be raised
       ! Note: same numbering of positive flags as in matrix_util
       if (mu_flag .gt. 0) then
          inform%flag = mu_flag
          call inform%print_flag(options, context)
       end if
       nz = akeep%ptr(n+1) - 1
    else
       nz = ptr(n+1)-1
    end if

    !
    ! If the pivot order is not supplied, we need to compute an order.
    ! Otherwise, we check the supplied order.
    !

    allocate(akeep%invp(n),order2(n),ptr2(n+1),row2(2*nz),stat=st)
    if (st .ne. 0) go to 490
    if (options%ordering .eq. 2) then
       allocate(akeep%scaling(n), val2(2*nz), stat=st)
       if (st .ne. 0) go to 490
    end if

    select case(options%ordering)
    case(0)
       if (.not. present(order)) then
          ! we have an error since user should have supplied the order
          inform%flag = SSIDS_ERROR_ORDER
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end if
       call check_order(n,order,akeep%invp,options,inform)
       if (inform%flag .lt. 0) go to 490
       order2(1:n) = order(1:n)
       if (check) then
          call expand_pattern(n, nz, akeep%ptr, akeep%row, ptr2, row2)
       else
          call expand_pattern(n, nz, ptr, row, ptr2, row2)
       end if
    case(1)
       ! METIS ordering
       if (check) then
          call metis_order(n, akeep%ptr, akeep%row, order2, akeep%invp, &
               flag, inform%stat)
          call expand_pattern(n, nz, akeep%ptr, akeep%row, ptr2, row2)
       else
          call metis_order(n, ptr, row, order2, akeep%invp, &
               flag, inform%stat)
          call expand_pattern(n, nz, ptr, row, ptr2, row2)
       end if
       if (flag .lt. 0) go to 490
    case(2)
       ! matching-based ordering required
       ! Expand the matrix as more efficient to do it and then
       ! call match_order_metis() with full matrix supplied

       if (check) then
          call expand_matrix(n, nz, akeep%ptr, akeep%row, val_clean, ptr2, &
               row2, val2)
          deallocate (val_clean,stat=st)
       else
          call expand_matrix(n, nz, ptr, row, val, ptr2, row2, val2)
       end if

       call match_order_metis(n, ptr2, row2, val2, order2, akeep%scaling, &
            mo_flag, inform%stat)

       select case(mo_flag)
       case(0)
          ! Success; do nothing
       case(1)
          ! singularity warning required
          inform%flag = SSIDS_WARNING_ANAL_SINGULAR
       case(-1)
          inform%flag = SSIDS_ERROR_ALLOCATION
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       case default
          inform%flag = SSIDS_ERROR_UNKNOWN
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end select

       deallocate(val2,stat=st)
    end select

    ! Figure out topology
    if (present(topology)) then
       ! User supplied
       allocate(akeep%topology(size(topology)), stat=st)
       if (st .ne. 0) goto 490
       akeep%topology(:) = topology(:)
    else
       ! Guess it
       call guess_topology(akeep%topology, st)
       if (st .ne. 0) goto 490
    end if
    call squash_topology(akeep%topology, options, st)
    if (st .ne. 0) goto 490

    ! perform rest of analyse
    if (check) then
       call analyse_phase(n, akeep%ptr, akeep%row, ptr2, row2, order2,  &
            akeep%invp, akeep, options, inform)
    else
       call analyse_phase(n, ptr, row, ptr2, row2, order2, akeep%invp, &
            akeep, options, inform)
    end if

    if (present(order)) order(1:n) = abs(order2(1:n))
    if (options%print_level .gt. DEBUG_PRINT_LEVEL) &
         print *, "order = ", order2(1:n)

490 continue
    inform%stat = st
    if (inform%stat .ne. 0) then
       inform%flag = SSIDS_ERROR_ALLOCATION
    end if
    akeep%inform = inform
    call inform%print_flag(options, context)
  end subroutine analyse_double

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Given an initial topology, modify it to squash any resources options
!>        parameters tell us to ignore.
!> @param topology
  subroutine squash_topology(topology, options, st)
    implicit none
    type(numa_region), dimension(:), allocatable, intent(inout) :: topology
    type(ssids_options), intent(in) :: options
    integer, intent(out) :: st

    logical :: no_omp
    integer :: i, j, ngpu
    type(numa_region), dimension(:), allocatable :: new_topology

    st = 0

    no_omp = .true.
!$  no_omp = .false.

    ! Get rid of GPUs if we're not using them
    if (.not. options%use_gpu) then
       do i = 1, size(topology)
          if (size(topology(i)%gpus) .ne. 0) then
             deallocate(topology(i)%gpus)
             allocate(topology(i)%gpus(0), stat=st)
             if (st .ne. 0) return
          end if
       end do
    end if

    ! FIXME: One can envisage a sensible coexistence of both
    ! no_omp=.true. AND options%ignore_numa=.false. (e.g., choose the
    ! "best" NUMA node, with the least utilised CPUs and/or GPUs...).
    if (no_omp) then
       allocate(new_topology(1), stat=st)
       if (st .ne. 0) return
       new_topology(1)%nproc = 1
       ! Count resources to reallocate
       ngpu = 0
       do i = 1, size(topology)
          ngpu = ngpu + size(topology(i)%gpus)
       end do
       ! FIXME: if no_omp=.true. AND options%ignore_numa=.true.,
       ! then take the "first" GPU (whichever it might be), only.
       ! A combination not meant for production, only for testing!
       if (options%ignore_numa) ngpu = min(ngpu, 1)
       ! Store list of GPUs
       allocate(new_topology(1)%gpus(ngpu), stat=st)
       if (st .ne. 0) return
       if (ngpu .gt. 0) then
          if (options%ignore_numa) then
             new_topology(1)%gpus(1) = huge(new_topology(1)%gpus(1))
             do i = 1, size(topology)
                new_topology(1)%gpus(1) = &
                     min(new_topology(1)%gpus(1), minval(topology(i)%gpus))
             end do
          else
             ngpu = 0
             do i = 1, size(topology)
                do j = 1, size(topology(i)%gpus)
                   new_topology(1)%gpus(ngpu + j) = topology(i)%gpus(j)
                end do
                ngpu = ngpu + size(topology(i)%gpus)
             end do
          end if
       end if
       ! Move new_topology into place, deallocating old one
       deallocate(topology)
       call move_alloc(new_topology, topology)
    ! Squash everything to single NUMA region if we're ignoring numa
    else if ((size(topology) .gt. 1) .and. options%ignore_numa) then
       allocate(new_topology(1), stat=st)
       if (st .ne. 0) return
       ! Count resources to reallocate
       new_topology(1)%nproc = 0
       ngpu = 0
       do i = 1, size(topology)
          new_topology(1)%nproc = new_topology(1)%nproc + topology(i)%nproc
          ngpu = ngpu + size(topology(i)%gpus)
       end do
       ! Store list of GPUs
       allocate(new_topology(1)%gpus(ngpu), stat=st)
       if (st .ne. 0) return
       if (ngpu .gt. 0) then
          ngpu = 0
          do i = 1, size(topology)
             do j = 1, size(topology(i)%gpus)
                new_topology(1)%gpus(ngpu + j) = topology(i)%gpus(j)
             end do
             ngpu = ngpu + size(topology(i)%gpus)
          end do
       end if
       ! Move new_topology into place, deallocating old one
       deallocate(topology)
       call move_alloc(new_topology, topology)
    end if
  end subroutine squash_topology

!****************************************************************************
!
! Analyse phase.
! Matrix entered in coordinate format.
! matrix_util routine is used to convert the data to CSC format.
! The user optionally inputs the pivot order. If not, metis called. 
! Structure is then expanded.
! Supervariables are computed
! and then the assembly tree is constructed and the data structures
! required by the factorization are set up.
!
  subroutine ssids_analyse_coord_double(n, ne, row, col, akeep, options, &
       inform, order, val, topology)
    implicit none
    integer, intent(in) :: n ! order of A
    integer(long), intent(in) :: ne ! entries to be input by user
    integer, intent(in) :: row(:) ! row indices
    integer, intent(in) :: col(:) ! col indices
    type(ssids_akeep), intent(inout) :: akeep ! See derived-type declaration
    type(ssids_options), intent(in) :: options ! See derived-type declaration
    type(ssids_inform), intent(out) :: inform ! See derived-type declaration
    integer, intent(inout), optional  :: order(:)
      ! Must be present and set on entry if options%ordering = 0 
      ! i is used to index a variable, order(i) must
      ! hold its position in the pivot sequence.
      ! If i is not used to index a variable,
      ! order(i) must be set to zero.
      ! On exit, holds the pivot order to be used by factorization.
    real(wp), optional, intent(in) :: val(:) ! must be present
      ! if a matching-based elimination ordering is required 
      ! (options%ordering = 2).
      ! If present, val(k) must hold value of entry in row(k) and col(k).
    type(numa_region), dimension(:), optional, intent(in) :: topology
      ! user specified topology

    integer(long), dimension(:), allocatable :: ptr2 ! col ptrs for expanded mat
    integer, dimension(:), allocatable :: row2 ! row indices for expanded matrix
    integer, dimension(:), allocatable :: order2 ! pivot order

    integer :: mo_flag

    real(wp), dimension(:), allocatable :: val_clean ! cleaned values if
      ! val is present.
    real(wp), dimension(:), allocatable :: val2 ! expanded matrix (val present)

    character(50)  :: context      ! Procedure name (used when printing).
    integer :: mu_flag      ! error flag for matrix_util routines
    integer(long) :: nz     ! entries in expanded matrix
    integer :: flag         ! error flag for metis
    integer :: st           ! stat parameter
    integer :: free_flag

    type(ssids_inform) :: inform_default

    ! Initialise
    context = 'ssids_analyse_coord'
    inform = inform_default
    call ssids_free(akeep, free_flag)
    if (free_flag .ne. 0) then
       inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
       inform%cuda_error = free_flag
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    ! Output status on entry
    call options%print_summary_analyse(context)
    if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
       write (options%unit_diagnostics,'(a,i15)') &
            ' n                         =  ', n
       write (options%unit_diagnostics,'(a,i15)') &
            ' ne                        =  ', ne
    end if

    akeep%check = .true.
    akeep%n = n

    !
    ! Checking of matrix data
    !
    if ((n .lt. 0) .or. (ne .lt. 0)) then
       inform%flag = SSIDS_ERROR_A_N_OOR
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    if (n .eq. 0) then
       akeep%nnodes = 0
       allocate(akeep%sptr(0), stat=st) ! used to check if analyse has been run
       if (st .ne. 0) go to 490
       akeep%inform = inform
       return
    end if

    ! check options%ordering has a valid value
    if ((options%ordering .lt. 0) .or. (options%ordering .gt. 2)) then
       inform%flag = SSIDS_ERROR_ORDER
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    ! check val present when expected
    if (options%ordering .eq. 2) then
       if (.not. present(val)) then
          inform%flag = SSIDS_ERROR_VAL
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end if
    end if

    st = 0
    allocate(akeep%ptr(n+1),stat=st)
    if (st .ne. 0) go to 490

    if (present(val)) then
       call convert_coord_to_cscl(SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ne, row,  &
            col, akeep%ptr, akeep%row, mu_flag, val_in=val, val_out=val_clean, &
            lmap=akeep%lmap, map=akeep%map,                                    &
            noor=inform%matrix_outrange,  ndup=inform%matrix_dup)
    else
       call convert_coord_to_cscl(SPRAL_MATRIX_REAL_SYM_INDEF, n, n, ne, row,   &
            col, akeep%ptr, akeep%row, mu_flag, lmap=akeep%lmap, map=akeep%map, &
            noor=inform%matrix_outrange,  ndup=inform%matrix_dup)
    end if

    ! Check for errors
    if (mu_flag .lt. 0) then
       if (mu_flag .eq. -1)  inform%flag = SSIDS_ERROR_ALLOCATION
       if (mu_flag .eq. -10) inform%flag = SSIDS_ERROR_A_ALL_OOR
       akeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    ! Check whether warning needs to be raised
    ! Note: same numbering of positive flags as in matrix_util
    if (mu_flag .gt. 0) inform%flag = mu_flag

    nz = akeep%ptr(n+1) - 1

    ! If the pivot order is not supplied, we need to compute an order
    ! here, before we expand the matrix structure.
    ! Otherwise, we must check the supplied order.

    allocate(akeep%invp(n),order2(n),ptr2(n+1),row2(2*nz),stat=st)
    if (st .ne. 0) go to 490
    if (options%ordering .eq. 2) then
       allocate(val2(2*nz),akeep%scaling(n),stat=st)
       if (st .ne. 0) go to 490
    end if

    select case(options%ordering)
    case(0)
       if (.not. present(order)) then
          ! we have an error since user should have supplied the order
          inform%flag = SSIDS_ERROR_ORDER
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end if
       call check_order(n,order,akeep%invp,options,inform)
       if (inform%flag .lt. 0) go to 490
       order2(1:n) = order(1:n)
       call expand_pattern(n, nz, akeep%ptr, akeep%row, ptr2, row2)

    case(1)
       ! METIS ordering
       call metis_order(n, akeep%ptr, akeep%row, order2, akeep%invp, &
            flag, inform%stat)
       if (flag .lt. 0) go to 490
       call expand_pattern(n, nz, akeep%ptr, akeep%row, ptr2, row2)

    case(2)
       ! matching-based ordering required

       call expand_matrix(n, nz, akeep%ptr, akeep%row, val_clean, ptr2, row2, &
            val2)
       deallocate (val_clean,stat=st)

       call match_order_metis(n,ptr2,row2,val2,order2,akeep%scaling,mo_flag, &
            inform%stat)

       select case(mo_flag)
       case(0)
          ! Success; do nothing
       case(1)
          ! singularity warning required
          inform%flag = SSIDS_WARNING_ANAL_SINGULAR
       case(-1)
          inform%flag = SSIDS_ERROR_ALLOCATION
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       case default
          inform%flag = SSIDS_ERROR_UNKNOWN
          akeep%inform = inform
          call inform%print_flag(options, context)
          return
       end select

       deallocate(val2,stat=st)
    end select

    ! Figure out topology
    if (present(topology)) then
       ! User supplied
       allocate(akeep%topology(size(topology)), stat=st)
       if (st .ne. 0) goto 490
       akeep%topology(:) = topology(:)
    else
       ! Guess it
       call guess_topology(akeep%topology, st)
       if (st .ne. 0) goto 490
    end if

    ! we now have the expanded structure held using ptr2, row2
    ! and we are ready to get on with the analyse phase.
    call analyse_phase(n, akeep%ptr, akeep%row, ptr2, row2, order2,  &
         akeep%invp, akeep, options, inform)
    if (inform%flag .lt. 0) go to 490

    if (present(order)) order(1:n) = abs(order2(1:n))
    if (options%print_level .gt. DEBUG_PRINT_LEVEL) &
         print *, "order = ", order2(1:n)

490 continue
    inform%stat = st
    if (inform%stat .ne. 0) then
       inform%flag = SSIDS_ERROR_ALLOCATION
    end if
    akeep%inform = inform
    call inform%print_flag(options, context)
  end subroutine ssids_analyse_coord_double

!****************************************************************************
!
! Factorize phase - 32-bit wrapper around 64-bit version
! Note ptr is non-optional
!
  subroutine ssids_factor_ptr32_double(posdef, val, akeep, fkeep, options, &
       inform, scale, ptr, row)
    implicit none
    logical, intent(in) :: posdef 
    real(wp), dimension(*), target, intent(in) :: val
    type(ssids_akeep), intent(in) :: akeep
    type(ssids_fkeep), intent(inout) :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    real(wp), dimension(:), optional, intent(inout) :: scale
    integer, dimension(akeep%n+1), intent(in) :: ptr
    integer, dimension(*), optional, intent(in) :: row 

    integer(long), dimension(:), allocatable :: ptr64

    ! Copy from 32-bit to 64-bit ptr
    allocate(ptr64(akeep%n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = SSIDS_ERROR_ALLOCATION
       call inform%print_flag(options, 'ssids_factor')
       fkeep%inform = inform
       return
    end if
    ptr64(1:akeep%n+1) = ptr(1:akeep%n+1)

    ! Call 64-bit routine
    call ssids_factor_ptr64_double(posdef, val, akeep, fkeep, options, &
         inform, scale=scale, ptr=ptr64, row=row)
  end subroutine ssids_factor_ptr32_double

!****************************************************************************
!
! Factorize phase
!
  subroutine ssids_factor_ptr64_double(posdef, val, akeep, fkeep, options, &
       inform, scale, ptr, row)
    implicit none
    logical, intent(in) :: posdef 
    real(wp), dimension(*), target, intent(in) :: val ! A values (lwr triangle)
    type(ssids_akeep), intent(in) :: akeep
    type(ssids_fkeep), intent(inout) :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    real(wp), dimension(:), optional, intent(inout) :: scale ! used to hold
      ! row and column scaling factors. Must be set on entry if
      ! options%scaling <= 0
      ! Note: Has to be assumed shape, not assumed size or fixed size to work
      ! around funny compiler bug
    integer(long), dimension(akeep%n+1), optional, intent(in) :: ptr ! must be
      ! present if on call to analyse phase, check = .false.. Must be unchanged 
      ! since that call.
    integer, dimension(*), optional, intent(in) :: row ! must be present if
      ! on call to analyse phase, check = .false.. Must be unchanged 
      ! since that call.

    real(wp), dimension(:), allocatable, target :: val2
    character(len=50) :: context

    integer :: i
    integer :: n
    integer(long) :: nz
    integer :: st
    ! Solve parameters. Tree is broken up into multiple chunks. Parent-child
    ! relations between chunks are stored in fwd_ptr and fwd (see solve routine
    ! comments)
    integer :: matrix_type
    real(wp), dimension(:), allocatable :: scaling

    ! Types related to scaling routines
    type(hungarian_options) :: hsoptions
    type(hungarian_inform) :: hsinform
    type(equilib_options) :: esoptions
    type(equilib_inform) :: esinform

    type(omp_settings) :: user_omp_settings
    type(rb_write_options) :: rb_options
    integer :: flag
   
    ! Setup for any printing we may require
    context = 'ssids_factor'

    ! Print summary of input options (depending on print level etc.)
    call options%print_summary_factor(posdef, context)

    ! Check for error in call sequence
    if ((.not. allocated(akeep%sptr)) .or. (akeep%inform%flag .lt. 0)) then
       ! Analyse cannot have been run
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       fkeep%inform = inform
       return
    end if

    ! Initialize
    inform = akeep%inform
    st = 0
    n = akeep%n

    ! Ensure OpenMP setup is as required
    call push_omp_settings(user_omp_settings, inform%flag)
    if (inform%flag .lt. 0) then
       fkeep%inform = inform
       call inform%print_flag(options, context)
       return
    end if

    ! Immediate return if analyse detected singularity and options%action=false
    if ((.not. options%action) .and. (akeep%n .ne. akeep%inform%matrix_rank)) then
       inform%flag = SSIDS_ERROR_SINGULAR
       goto 100
    end if

    ! Immediate return for trivial matrix
    if (akeep%nnodes .eq. 0) then
       inform%flag = SSIDS_SUCCESS
       inform%matrix_rank = 0
       goto 100
    end if

    fkeep%pos_def = posdef
    if (posdef) then
       matrix_type = SPRAL_MATRIX_REAL_SYM_PSDEF
    else
       matrix_type = SPRAL_MATRIX_REAL_SYM_INDEF
    end if

    ! If matrix has been checked, produce a clean version of val in val2
    if (akeep%check) then
       nz = akeep%ptr(n+1) - 1
       allocate(val2(nz),stat=st)
       if (st .ne. 0) go to 10
       call apply_conversion_map(matrix_type, akeep%lmap, akeep%map, val, &
            nz, val2)
    else
       ! analyse run with no checking so must have ptr and row present
       if (.not. present(ptr)) inform%flag = SSIDS_ERROR_PTR_ROW
       if (.not. present(row)) inform%flag = SSIDS_ERROR_PTR_ROW
       if (inform%flag .lt. 0) then
          fkeep%inform = inform
          goto 100
       end if
    end if

    ! At this point, either  ptr, row, val   
    !                  or    akeep%ptr, akeep%row, val2
    ! hold the lower triangular part of A

    ! Dump matrix if required
    if (allocated(options%rb_dump)) then
       write(options%unit_warning,*) "Dumping matrix to '", options%rb_dump, "'"
       if (akeep%check) then
          call rb_write(options%rb_dump, SPRAL_MATRIX_REAL_SYM_INDEF, &
               n, n, akeep%ptr, akeep%row, rb_options, flag, val=val2)
       else
          call rb_write(options%rb_dump, SPRAL_MATRIX_REAL_SYM_INDEF, &
               n, n, ptr, row, rb_options, flag, val=val)
       end if
       if (flag .ne. 0) then
          inform%flag = SSIDS_ERROR_UNKNOWN
          goto 100
       end if
    end if

    !
    ! Perform scaling if required
    !
    if ((options%scaling .gt. 0) .or. present(scale)) then
       if (allocated(fkeep%scaling)) then
          if (size(fkeep%scaling) .lt. n) then
             deallocate(fkeep%scaling, stat=st)
             allocate(fkeep%scaling(n), stat=st)
          end if
       else
          allocate(fkeep%scaling(n), stat=st)
       end if
       if (st .ne. 0) go to 10
    else
       deallocate(fkeep%scaling, stat=st)
    end if

    if (allocated(akeep%scaling) .and. (options%scaling .ne. 3)) then
       inform%flag = SSIDS_WARNING_MATCH_ORD_NO_SCALE
       call inform%print_flag(options, context)
    end if

    select case (options%scaling)
    case(:0) ! User supplied or none
       if (present(scale)) then
          do i = 1, n
             fkeep%scaling(i) = scale(akeep%invp(i))
          end do
       end if
    case(1) ! Matching-based scaling by Hungarian Algorithm (MC64 algorithm)
       ! Allocate space for scaling
       allocate(scaling(n), stat=st)
       if (st .ne. 0) goto 10
       ! Run Hungarian algorithm
       hsoptions%scale_if_singular = options%action
       if (akeep%check) then
          call hungarian_scale_sym(n, akeep%ptr, akeep%row, val2, scaling, &
               hsoptions, hsinform)
       else
          call hungarian_scale_sym(n, ptr, row, val, scaling, &
               hsoptions, hsinform)
       end if
       select case(hsinform%flag)
       case(-1)
          ! Allocation error
          st = hsinform%stat
          goto 10
       case(-2)
          ! Structually singular matrix and control%action=.false.
          inform%flag = SSIDS_ERROR_SINGULAR
          goto 100
       end select
       ! Permute scaling to correct order
       do i = 1, n
          fkeep%scaling(i) = scaling(akeep%invp(i))
       end do
       ! Copy scaling(:) to user array scale(:) if present
       if (present(scale)) then
          scale(1:n) = scaling(1:n)
       end if
       ! Cleanup memory
       deallocate(scaling, stat=st)

    case(2) ! Matching-based scaling by Auction Algorithm
       ! Allocate space for scaling
       allocate(scaling(n), stat=st)
       if (st .ne. 0) goto 10
       ! Run auction algorithm
       if (akeep%check) then
          call auction_scale_sym(n, akeep%ptr, akeep%row, val2, scaling, &
               options%auction, inform%auction)
       else
          call auction_scale_sym(n, ptr, row, val, scaling, &
               options%auction, inform%auction)
       end if
       if (inform%auction%flag .ne. 0) then
          ! only possible error is allocation failed
          st = inform%auction%stat
          goto 10 
       end if
       ! Permute scaling to correct order
       do i = 1, n
          fkeep%scaling(i) = scaling(akeep%invp(i))
       end do
       ! Copy scaling(:) to user array scale(:) if present
       if (present(scale)) then
          scale(1:n) = scaling(1:n)
       end if
       ! Cleanup memory
       deallocate(scaling, stat=st)

    case(3) ! Scaling generated during analyse phase for matching-based order
       if (.not. allocated(akeep%scaling)) then
          ! No scaling saved from analyse phase
          inform%flag = SSIDS_ERROR_NO_SAVED_SCALING
          goto 100
       end if
       do i = 1, n
          fkeep%scaling(i) = akeep%scaling(akeep%invp(i))
       end do

    case(4:) ! Norm equilibriation algorithm (MC77 algorithm)
       ! Allocate space for scaling
       allocate(scaling(n), stat=st)
       if (st .ne. 0) goto 10
       ! Run equilibriation algorithm
       if (akeep%check) then
          call equilib_scale_sym(n, akeep%ptr, akeep%row, val2, scaling, &
               esoptions, esinform)
       else
          call equilib_scale_sym(n, ptr, row, val, scaling, &
               esoptions, esinform)
       end if
       if (esinform%flag .ne. 0) then
          ! Only possible error is memory allocation failure
          st = esinform%stat
          goto 10
       end if
       ! Permute scaling to correct order
       do i = 1, n
          fkeep%scaling(i) = scaling(akeep%invp(i))
       end do
       ! Copy scaling(:) to user array scale(:) if present
       if (present(scale)) then
          do i = 1, n
             scale(akeep%invp(i)) = fkeep%scaling(i)
          end do
       end if
       ! Cleanup memory
       deallocate(scaling, stat=st)
    end select

    !if(allocated(fkeep%scaling)) &
    !   print *, "minscale, maxscale = ", minval(fkeep%scaling), &
    !      maxval(fkeep%scaling)

    ! Setup data storage
    if (allocated(fkeep%subtree)) then
       do i = 1, size(fkeep%subtree)
          if (associated(fkeep%subtree(i)%ptr)) then
             call fkeep%subtree(i)%ptr%cleanup()
             deallocate(fkeep%subtree(i)%ptr)
          end if
       end do
       deallocate(fkeep%subtree)
    end if

    ! Call main factorization routine
    if (akeep%check) then
       call fkeep%inner_factor(akeep, val2, options, inform)
    else
       call fkeep%inner_factor(akeep, val, options, inform)
    end if
    if (inform%flag .lt. 0) then
       fkeep%inform = inform
       goto 100
    end if

    if (akeep%n .ne. inform%matrix_rank) then
       ! Rank deficient
       ! Note: If we reach this point then must be options%action=.true.
       if (options%action) then
          inform%flag = SSIDS_WARNING_FACT_SINGULAR
       else
          inform%flag = SSIDS_ERROR_SINGULAR
       end if
       call inform%print_flag(options, context)
    end if

    if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
       write (options%unit_diagnostics,'(/a)') &
            ' Completed factorisation with:'
       write (options%unit_diagnostics, &
            '(a,2(/a,i12),2(/a,es12.4),5(/a,i12))') &
            ' information parameters (inform%) :', &
            ' flag                   Error flag                               = ',&
            inform%flag, &
            ' maxfront               Maximum frontsize                        = ',&
            inform%maxfront, &
            ' num_factor             Number of entries in L                   = ',&
            real(inform%num_factor), &
            ' num_flops              Number of flops performed                = ',&
            real(inform%num_flops), &
            ' num_two                Number of 2x2 pivots used                = ',&
            inform%num_two, &
            ' num_delay              Number of delayed eliminations           = ',&
            inform%num_delay, &
            ' rank                   Computed rank                            = ',&
            inform%matrix_rank, &
            ' num_neg                Computed number of negative eigenvalues  = ',&
            inform%num_neg
    end if

    ! Normal return just drops through
100 continue
    ! Clean up and return
    fkeep%inform = inform
    call inform%print_flag(options, context)
    call pop_omp_settings(user_omp_settings)
    return
    !!!!!!!!!!!!!!!!!!!!

    !
    ! Error handling
    !
10  continue
    inform%flag = SSIDS_ERROR_ALLOCATION
    inform%stat = st
    goto 100
  end subroutine ssids_factor_ptr64_double

!*************************************************************************
!
! Solve phase single x. 
!
  subroutine ssids_solve_one_double(x1, akeep, fkeep, options, inform, job)
    implicit none
    real(wp), dimension(:), intent(inout) :: x1 ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i) is the corresponding component of the
      ! right-hand side.On exit, if i has been used to index a variable,
      ! x(i) holds solution for variable i
    type(ssids_akeep), intent(in) :: akeep
    type(ssids_fkeep), intent(inout) :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    integer, optional, intent(in) :: job

    integer :: ldx

    ldx = size(x1)
    if (present(job)) then
       call ssids_solve_mult_double(1, x1, ldx, akeep, fkeep, options, inform, job)
    else
       call ssids_solve_mult_double(1, x1, ldx, akeep, fkeep, options, inform)
    end if
  end subroutine ssids_solve_one_double

!*************************************************************************

  subroutine ssids_solve_mult_double(nrhs, x, ldx, akeep, fkeep, options, &
       inform, job)
    implicit none
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(ldx,nrhs), intent(inout), target :: x
    type(ssids_akeep), intent(in) :: akeep ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i,j) is the corresponding component of the
      ! right-hand side for the jth system (j = 1,2,..., nrhs).
      ! On exit, if i has been used to index a variable,
      ! x(i,j) holds solution for variable i to system j
    ! For details of keep, options, inform : see derived type description
    type(ssids_fkeep), intent(inout) :: fkeep !inout for moving data
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    integer, optional, intent(in) :: job ! used to indicate whether
      ! partial solution required
      ! job = 1 : forward eliminations only (PLX = B)
      ! job = 2 : diagonal solve (DX = B) (indefinite case only)
      ! job = 3 : backsubs only ((PL)^TX = B)
      ! job = 4 : diag and backsubs (D(PL)^TX = B) (indefinite case only)
      ! job absent: complete solve performed

    character(50)  :: context  ! Procedure name (used when printing).
    integer :: local_job ! local job parameter
    integer :: n

    inform%flag = SSIDS_SUCCESS

    ! Perform appropriate printing
    if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
       write (options%unit_diagnostics,'(//a)') &
            ' Entering ssids_solve with:'
       write (options%unit_diagnostics,'(a,4(/a,i12),(/a,i12))') &
            ' options parameters (options%) :', &
            ' print_level         Level of diagnostic printing        = ', &
            options%print_level, &
            ' unit_diagnostics    Unit for diagnostics                = ', &
            options%unit_diagnostics, &
            ' unit_error          Unit for errors                     = ', &
            options%unit_error, &
            ' unit_warning        Unit for warnings                   = ', &
            options%unit_warning, &
            ' nrhs                                                    = ', &
            nrhs
       if (nrhs .gt. 1) write (options%unit_diagnostics,'(/a,i12)') &
            ' ldx                                                     = ', &
            ldx
    end if

    context = 'ssids_solve'

    if (akeep%nnodes .eq. 0) return

    if (.not. allocated(fkeep%subtree)) then
       ! factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    inform%flag = max(SSIDS_SUCCESS, fkeep%inform%flag) ! Preserve warnings
    ! immediate return if already had an error
    if ((akeep%inform%flag .lt. 0) .or. (fkeep%inform%flag .lt. 0)) then
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    n = akeep%n
    if (ldx .lt. n) then
       inform%flag = SSIDS_ERROR_X_SIZE
       call inform%print_flag(options, context)
       if ((options%print_level .ge. 0) .and. (options%unit_error .gt. 0)) &
            write (options%unit_error,'(a,i8,a,i8)') &
            ' Increase ldx from ', ldx, ' to at least ', n
       return
    end if

    if (nrhs .lt. 1) then
       inform%flag = SSIDS_ERROR_X_SIZE
       call inform%print_flag(options, context)
       if ((options%print_level .ge. 0) .and. (options%unit_error .gt. 0)) &
            write (options%unit_error,'(a,i8,a,i8)') &
            ' nrhs must be at least 1. nrhs = ', nrhs
       return
    end if

    ! Copy previous phases' inform data from akeep and fkeep
    inform = fkeep%inform

    ! Set local_job
    local_job = 0
    if (present(job)) then
       if ((job .lt. SSIDS_SOLVE_JOB_FWD) .or. (job .gt. SSIDS_SOLVE_JOB_DIAG_BWD)) &
            inform%flag = SSIDS_ERROR_JOB_OOR
       if (fkeep%pos_def .and. (job .eq. SSIDS_SOLVE_JOB_DIAG)) &
            inform%flag = SSIDS_ERROR_JOB_OOR
       if (fkeep%pos_def .and. (job .eq. SSIDS_SOLVE_JOB_DIAG_BWD)) &
            inform%flag = SSIDS_ERROR_JOB_OOR
       if (inform%flag .eq. SSIDS_ERROR_JOB_OOR) then
          call inform%print_flag(options, context)
          return
       end if
       local_job = job
    end if

    call fkeep%inner_solve(local_job, nrhs, x, ldx, akeep, inform)
    call inform%print_flag(options, context)
  end subroutine ssids_solve_mult_double

!*************************************************************************
!
! Return diagonal entries to user
!
  subroutine ssids_enquire_posdef_double(akeep, fkeep, options, inform, d)
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    type(ssids_fkeep), target, intent(in) :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    real(wp), dimension(*), intent(out) :: d

    character(50)  :: context      ! Procedure name (used when printing).

    context = 'ssids_enquire_posdef' 
    inform%flag = SSIDS_SUCCESS

    if (.not. allocated(fkeep%subtree)) then
       ! factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    if ((akeep%inform%flag .lt. 0) .or. (fkeep%inform%flag .lt. 0)) then
       ! immediate return if had an error
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    if (.not. fkeep%pos_def) then
       inform%flag = SSIDS_ERROR_NOT_LLT
       call inform%print_flag(options, context)
       return
    end if

    call fkeep%enquire_posdef(akeep, d)
    call inform%print_flag(options, context)
  end subroutine ssids_enquire_posdef_double

!*************************************************************************
! In indefinite case, the pivot sequence used will not necessarily be
! the same as that passed to ssids_factor (because of delayed pivots). This
! subroutine allows the user to obtain the pivot sequence that was
! actually used.
! also the entries of D^{-1} are returned using array d.
!
  subroutine ssids_enquire_indef_double(akeep, fkeep, options, inform, &
       piv_order, d)
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    type(ssids_fkeep), target, intent(in) :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform
    integer, dimension(*), optional, intent(out) :: piv_order
      ! If i is used to index a variable, its position in the pivot sequence
      ! will be placed in piv_order(i), with its sign negative if it is
      ! part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.
    real(wp), dimension(2,*), optional, intent(out) :: d ! The diagonal
      ! entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
      ! entries will be placed in d(2,:). The entries are held in pivot order.

    character(50)  :: context      ! Procedure name (used when printing).

    context = 'ssids_enquire_indef' 
    inform%flag = SSIDS_SUCCESS

    if (.not. allocated(fkeep%subtree)) then
       ! factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    if ((akeep%inform%flag .lt. 0) .or. (fkeep%inform%flag .lt. 0)) then
       ! immediate return if had an error
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    if (fkeep%pos_def) then
       inform%flag = SSIDS_ERROR_NOT_LDLT
       call inform%print_flag(options, context)
       return
    end if

    call fkeep%enquire_indef(akeep, inform, piv_order, d)
    call inform%print_flag(options, context)
  end subroutine ssids_enquire_indef_double

!*************************************************************************
!
! In indefinite case, the entries of D^{-1} may be changed using this routine.
!
  subroutine ssids_alter_double(d, akeep, fkeep, options, inform)
    implicit none
    real(wp), dimension(2,*), intent(in) :: d  ! The required diagonal entries
      ! of D^{-1} must be placed in d(1,i) (i = 1,...n)
      ! and the off-diagonal entries must be placed in d(2,i) (i = 1,...n-1).
    type(ssids_akeep), intent(in) :: akeep
    type(ssids_fkeep), target, intent(inout) :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(out) :: inform

    character(50)  :: context      ! Procedure name (used when printing).

    context = 'ssids_alter'
    inform%flag = SSIDS_SUCCESS

    if (.not. allocated(fkeep%subtree)) then
       ! factorize phase has not been performed
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    ! immediate return if already had an error
    if ((akeep%inform%flag .lt. 0) .or. (fkeep%inform%flag .lt. 0)) then
       inform%flag = SSIDS_ERROR_CALL_SEQUENCE
       call inform%print_flag(options, context)
       return
    end if

    if (fkeep%pos_def) then
       inform%flag = SSIDS_ERROR_NOT_LDLT
       call inform%print_flag(options, context)
       return
    end if

    call fkeep%alter(d, akeep)
    call inform%print_flag(options, context)
  end subroutine ssids_alter_double

!*************************************************************************

  subroutine free_akeep_double(akeep, flag)
    implicit none
    type(ssids_akeep), intent(inout) :: akeep
    integer, intent(out) :: flag

    call akeep%free(flag)
  end subroutine free_akeep_double

!*******************************

  subroutine free_fkeep_double(fkeep, cuda_error)
    implicit none
    type(ssids_fkeep), intent(inout) :: fkeep
    integer, intent(out) :: cuda_error

    call fkeep%free(cuda_error)
  end subroutine free_fkeep_double

!*******************************

  subroutine free_both_double(akeep, fkeep, cuda_error)
    implicit none
    type(ssids_akeep), intent(inout) :: akeep
    type(ssids_fkeep), intent(inout) :: fkeep
    integer, intent(out) :: cuda_error

    ! NB: Must free fkeep first as it may reference akeep
    call free_fkeep_double(fkeep, cuda_error)
    if (cuda_error .ne. 0) return
    call free_akeep_double(akeep, cuda_error)
    if (cuda_error .ne. 0) return
  end subroutine free_both_double

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> \brief Ensure OpenMP ICVs are as required, and store user versions for
!>        later restoration.
!> \sa pop_omp_settings()
  subroutine push_omp_settings(user_settings, flag)
    implicit none
    type(omp_settings), intent(out) :: user_settings
    integer, intent(inout) :: flag

    ! Dummy, for now.
    user_settings%nested = .true.
    user_settings%max_active_levels = huge(user_settings%max_active_levels)

!$  ! issue an error if we don't have cancellation (could lead to segfaults)
!$  if (.not. omp_get_cancellation()) then
!$     flag = SSIDS_ERROR_OMP_CANCELLATION
!$     return
!$  end if

!$  ! issue a warning if proc_bind is not enabled
!$  if (omp_get_proc_bind() .eq. OMP_PROC_BIND_FALSE) &
!$       flag = SSIDS_WARNING_OMP_PROC_BIND

!$  ! must have nested enabled
!$  user_settings%nested = omp_get_nested()
!$  if (.not. user_settings%nested) call omp_set_nested(.true.)

!$  ! we need OMP_DYNAMIC to be unset, to guarantee the number of threads
!$  user_settings%dynamic = omp_get_dynamic()
!$  if (user_settings%dynamic) call omp_set_dynamic(.false.)

!$  ! we will need at least 2 active levels
!$  user_settings%max_active_levels = omp_get_max_active_levels()
!$  if (user_settings%max_active_levels .lt. 2) call omp_set_max_active_levels(2)
  end subroutine push_omp_settings

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> \brief Restore user OpenMP ICV values.
!> \sa push_omp_settings()
  subroutine pop_omp_settings(user_settings)
    implicit none
    type(omp_settings), intent(in) :: user_settings

!$  if (.not. user_settings%nested) call omp_set_nested(user_settings%nested)
!$  if (user_settings%dynamic) call omp_set_dynamic(user_settings%dynamic)    
!$  if (user_settings%max_active_levels .lt. 2) &
!$       call omp_set_max_active_levels(user_settings%max_active_levels)
  end subroutine pop_omp_settings

end module spral_ssids
