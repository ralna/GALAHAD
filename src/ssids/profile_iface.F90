! THIS VERSION: GALAHAD 4.3 - 2024-01-15 AT 14:40 GMT.

#include "ssids_routines.h"
#include "spral_procedures.h"

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
!> \author    Florent Lopez
module spral_ssids_profile
   use spral_kinds
   implicit none

   private
   public :: profile_begin, profile_end, profile_task_type, &
             profile_create_task, profile_set_state, profile_add_event

   type :: profile_task_type
      private
      type(C_PTR) :: ctask
   contains
      procedure :: end_task
   end type profile_task_type

#ifdef INTEGER_64
   interface
      subroutine c_begin(nregions, regions) &
            bind(C, name="spral_ssids_profile_begin_64")
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int), value :: nregions
        type(c_ptr), value, intent(in) :: regions
      end subroutine c_begin
      subroutine profile_end() &
            bind(C, name="spral_ssids_profile_end_64")
      end subroutine profile_end
      type(C_PTR) function c_create_task(name, thread) &
            bind(C, name="spral_ssids_profile_create_task_64")
         use, intrinsic :: iso_c_binding
         character(C_CHAR), dimension(*), intent(in) :: name
         integer(C_INT), value :: thread
      end function c_create_task
      subroutine c_end_task(task) &
            bind(C, name="spral_ssids_profile_end_task_64")
         use, intrinsic :: iso_c_binding
         type(C_PTR), value :: task
      end subroutine c_end_task
      subroutine c_set_state(container, type, name) &
            bind(C, name="spral_ssids_profile_set_state_64")
         use, intrinsic :: iso_c_binding
         character(C_CHAR), dimension(*), intent(in) :: container
         character(C_CHAR), dimension(*), intent(in) :: type
         character(C_CHAR), dimension(*), intent(in) :: name
      end subroutine c_set_state
      subroutine c_add_event(type, val, thread) &
        bind(C, name="spral_ssids_profile_add_event_64")
        use, intrinsic :: iso_c_binding
        implicit none
        character(C_CHAR), dimension(*), intent(in) :: type
        character(C_CHAR), dimension(*), intent(in) :: val
        integer(C_INT), value :: thread
      end subroutine c_add_event
   end interface
#else
   interface
      subroutine c_begin(nregions, regions) &
            bind(C, name="spral_ssids_profile_begin")
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int), value :: nregions
        type(c_ptr), value, intent(in) :: regions
      end subroutine c_begin
      subroutine profile_end() &
            bind(C, name="spral_ssids_profile_end")
      end subroutine profile_end
      type(C_PTR) function c_create_task(name, thread) &
            bind(C, name="spral_ssids_profile_create_task")
         use, intrinsic :: iso_c_binding
         character(C_CHAR), dimension(*), intent(in) :: name
         integer(C_INT), value :: thread
      end function c_create_task
      subroutine c_end_task(task) &
            bind(C, name="spral_ssids_profile_end_task")
         use, intrinsic :: iso_c_binding
         type(C_PTR), value :: task
      end subroutine c_end_task
      subroutine c_set_state(container, type, name) &
            bind(C, name="spral_ssids_profile_set_state")
         use, intrinsic :: iso_c_binding
         character(C_CHAR), dimension(*), intent(in) :: container
         character(C_CHAR), dimension(*), intent(in) :: type
         character(C_CHAR), dimension(*), intent(in) :: name
      end subroutine c_set_state
      subroutine c_add_event(type, val, thread) &
        bind(C, name="spral_ssids_profile_add_event")
        use, intrinsic :: iso_c_binding
        implicit none
        character(C_CHAR), dimension(*), intent(in) :: type
        character(C_CHAR), dimension(*), intent(in) :: val
        integer(C_INT), value :: thread
      end subroutine c_add_event
   end interface
#endif

contains

  subroutine profile_begin(regions)
    use spral_hw_topology, only : numa_region, c_numa_region
    implicit none

    type(numa_region), dimension(:), intent(in) :: regions

    type(c_numa_region), dimension(:), pointer, contiguous :: f_regions
    integer(c_int) :: nregions
    integer(ip_) :: ngpus
    integer(ip_) :: i
    integer(ip_) :: st
    integer(c_int), dimension(:), pointer, contiguous :: gpus
    type(c_ptr) :: c_regions

    nullify(gpus)

    nregions = size(regions, 1)
    allocate(f_regions(nregions), stat=st)
    do i = 1, nregions
       f_regions(i)%nproc = regions(i)%nproc
       ngpus = size(regions(i)%gpus, 1)
       f_regions(i)%ngpu = ngpus
       if (ngpus .gt. 0) then
          allocate(gpus(ngpus), stat=st)
          gpus(:) = int(regions(i)%gpus,kind=c_int)
          f_regions(i)%gpus = c_loc(gpus(1))
          nullify(gpus)
       end if
    end do

    c_regions = c_loc(f_regions)

    call c_begin(nregions, c_regions)

    ! TODO free data structures

  end subroutine profile_begin

  type(profile_task_type) function profile_create_task(name, thread)
    character(len=*), intent(in) :: name
    integer(ip_), optional, intent(in) :: thread

    integer(C_INT) :: mythread
    character(C_CHAR), dimension(200) :: cname

    mythread = -1 ! autodetect
    if(present(thread)) mythread = int(thread,kind=C_INT)
    call f2c_string(name, cname)

    profile_create_task%ctask = c_create_task(cname, mythread)
  end function profile_create_task

  subroutine end_task(this)
    class(profile_task_type), intent(in) :: this

    call c_end_task(this%ctask)
  end subroutine end_task

  subroutine profile_set_state(container, type, name)
    character(len=*), intent(in) :: container
    character(len=*), intent(in) :: type
    character(len=*), intent(in) :: name

    character(C_CHAR), dimension(200) :: cname, ctype, ccontainer

    call f2c_string(container, ccontainer)
    call f2c_string(type, ctype)
    call f2c_string(name, cname)
    call c_set_state(ccontainer, ctype, cname)
  end subroutine profile_set_state

  subroutine profile_add_event(type, val, thread)
    implicit none

    character(len=*), intent(in) :: type
    character(len=*), intent(in) :: val
    integer(ip_), optional, intent(in) :: thread

    integer(C_INT) :: mythread
    character(C_CHAR), dimension(200) :: ctype, cval

    call f2c_string(type, ctype)
    call f2c_string(val, cval)
    mythread = -1 ! autodetect
    if(present(thread)) mythread = int(thread,kind=C_INT)

    call c_add_event(ctype, cval, mythread)

  end subroutine profile_add_event

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Convert Fortran character to C string, adding null terminator.
  !> @param fstring Fortran string to convert.
  !> @param cstring On output, overwritten with C string. Must be long enough
  !>        to include null termination.
  !> @param stat Status, 0 on sucess, otherwise number of additional characters
  !>        required.
  subroutine f2c_string(fstring, cstring, stat)
    character(len=*), intent(in) :: fstring
    character(C_CHAR), dimension(:), intent(out) :: cstring
    integer(ip_), optional, intent(out) :: stat

    integer(ip_) :: i

    if(size(cstring).lt.len(fstring)+1) then
       ! Not big enough, need +1 for null terminator
       if(present(stat)) stat = len(fstring)+1 - size(cstring)
       return
    endif

    do i = 1, len(fstring)
       cstring(i) = fstring(i:i)
    end do
    cstring(len(fstring)+1) = C_NULL_CHAR
  end subroutine f2c_string

end module spral_ssids_profile
