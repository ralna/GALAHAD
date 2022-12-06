!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
!> \brief Hardware Topology module
!
!> \brief Provides routines for detecting and/or specifying hardware
!>        topology for topology-aware routines
module spral_hw_topology

  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: numa_region ! datatype describing regions
  public :: guess_topology ! returns best guess of hardware topology
  public :: c_numa_region

  !> Fortran interoperable definition of spral::hw_topology::NumaRegion
  type, bind(C) :: c_numa_region
     integer(C_INT) :: nproc
     integer(C_INT) :: ngpu
     type(C_PTR) :: gpus
  end type c_numa_region

  !> Represents a NUMA region
  type :: numa_region
     integer :: nproc !< Number of processors in region
     integer, dimension(:), allocatable :: gpus !< List of attached GPUs
  end type numa_region

  interface
     !> Interface to spral_hw_topology_guess()
     subroutine spral_hw_topology_guess(nregion, regions) bind(C)
       use, intrinsic :: iso_c_binding
       implicit none
       integer(C_INT), intent(out) :: nregion
       type(C_PTR), intent(out) :: regions
     end subroutine spral_hw_topology_guess
     !> Interface to spral_hw_topology_free()
     subroutine spral_hw_topology_free(nregion, regions) bind(C)
       use, intrinsic :: iso_c_binding
       implicit none
       integer(C_INT), value :: nregion
       type(C_PTR), value :: regions
     end subroutine spral_hw_topology_free
  end interface

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Return best guess for machine topology
!> @param regions Upon return allocated to have size equal to the number of
!>        NUMA regions. The members describe each region.
!> @param st Status return from allocate. If non-zero upon return, an allocation
!>        failed.
  subroutine guess_topology(regions, st)
    implicit none
    type(numa_region), dimension(:), allocatable, intent(out) :: regions
    integer, intent(out) :: st

    integer :: i
    integer(C_INT) :: nregions
    type(C_PTR) :: c_regions
    type(c_numa_region), dimension(:), pointer, contiguous :: f_regions
    integer(C_INT), dimension(:), pointer, contiguous :: f_gpus

    ! Get regions from C
    call spral_hw_topology_guess(nregions, c_regions)
    if (c_associated(c_regions)) then
       call c_f_pointer(c_regions, f_regions, shape=(/ nregions /))

       ! Copy to allocatable array
       allocate(regions(nregions), stat=st)
       if (st .ne. 0) return
       do i = 1, nregions
          regions(i)%nproc = f_regions(i)%nproc
          allocate(regions(i)%gpus(f_regions(i)%ngpu), stat=st)
          if (st .ne. 0) return
          if (f_regions(i)%ngpu .gt. 0) then
             call c_f_pointer(f_regions(i)%gpus, f_gpus, &
                  shape=(/ f_regions(i)%ngpu /))
             regions(i)%gpus = f_gpus(:)
          end if
       end do
    end if

    ! Free C version
    call spral_hw_topology_free(nregions, c_regions)
  end subroutine guess_topology

end module spral_hw_topology
