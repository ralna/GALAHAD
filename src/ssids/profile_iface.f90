!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_profile
   use, intrinsic :: iso_c_binding
   implicit none

   private
   public :: profile_begin, &
             profile_end, &
             profile_task_type, &
             profile_create_task, &
             profile_set_state

   type :: profile_task_type
      private
      type(C_PTR) :: ctask
   contains
      procedure :: end_task
   end type profile_task_type

   interface
      subroutine profile_begin() &
            bind(C, name="spral_ssids_profile_begin")
      end subroutine profile_begin
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
   end interface

contains
   type(profile_task_type) function profile_create_task(name, thread)
      character(len=*), intent(in) :: name
      integer, optional, intent(in) :: thread

      integer(C_INT) :: mythread
      character(C_CHAR), dimension(200) :: cname

      mythread = -1 ! autodetect
      if(present(thread)) mythread = thread
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
      integer, optional, intent(out) :: stat

      integer :: i

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
