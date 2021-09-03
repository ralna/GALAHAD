!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_cpu_iface
   use, intrinsic :: iso_c_binding
   use spral_ssids_datatypes, only : ssids_options
   use spral_ssids_inform, only : ssids_inform
   implicit none

   private
   public :: cpu_factor_options, cpu_factor_stats
   public :: cpu_copy_options_in, cpu_copy_stats_out

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> @brief Interoperable subset of ssids_options
   !> @details Interoperates with cpu_factor_options C++ type
   !> @sa spral_ssids_datatypes::ssids_options
   !> @sa spral::ssids::cpu::cpu_factor_options
   type, bind(C) :: cpu_factor_options
      integer(C_INT) :: print_level
      logical(C_BOOL) :: action
      real(C_DOUBLE) :: small
      real(C_DOUBLE) :: u
      real(C_DOUBLE) :: multiplier
      integer(C_LONG_LONG) :: small_subtree_threshold
      integer(C_INT) :: cpu_block_size
      integer(C_INT) :: pivot_method
      integer(C_INT) :: failed_pivot_method
   end type cpu_factor_options

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> @brief Interoperable subset of ssids_inform
   !> @details Interoperates with ThreadStats C++ type
   !> @sa spral_ssids_inform::ssids_inform
   !> @sa spral::ssids::cpu::ThreadStats
   type, bind(C) :: cpu_factor_stats
      integer(C_INT) :: flag
      integer(C_INT) :: num_delay
      integer(C_INT) :: num_neg
      integer(C_INT) :: num_two
      integer(C_INT) :: num_zero
      integer(C_INT) :: maxfront
      integer(C_INT) :: not_first_pass
      integer(C_INT) :: not_second_pass
   end type cpu_factor_stats

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Copy subset of ssids_options to interoperable type
subroutine cpu_copy_options_in(foptions, coptions)
   type(ssids_options), intent(in) :: foptions
   type(cpu_factor_options), intent(out) :: coptions

   coptions%print_level    = foptions%print_level
   coptions%action         = foptions%action
   coptions%small          = foptions%small
   coptions%u              = foptions%u
   coptions%multiplier     = foptions%multiplier
   coptions%small_subtree_threshold = foptions%small_subtree_threshold
   coptions%cpu_block_size = foptions%cpu_block_size
   coptions%pivot_method   = min(3, max(1, foptions%pivot_method))
   coptions%failed_pivot_method = min(2, max(1, foptions%failed_pivot_method))
end subroutine cpu_copy_options_in

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Copy subset of ssids_inform from interoperable type
subroutine cpu_copy_stats_out(cstats, finform)
   type(cpu_factor_stats), intent(in) :: cstats
   type(ssids_inform), intent(inout) :: finform

   ! Combine stats
   if(cstats%flag < 0) then
      finform%flag = min(finform%flag, cstats%flag) ! error
   else
      finform%flag = max(finform%flag, cstats%flag) ! success or warning
   endif
   finform%num_delay    = finform%num_delay + cstats%num_delay
   finform%num_neg      = finform%num_neg + cstats%num_neg
   finform%num_two      = finform%num_two + cstats%num_two
   finform%maxfront     = max(finform%maxfront, cstats%maxfront)
   finform%not_first_pass = finform%not_first_pass + cstats%not_first_pass
   finform%not_second_pass = finform%not_second_pass + cstats%not_second_pass
   finform%matrix_rank  = finform%matrix_rank - cstats%num_zero
end subroutine cpu_copy_stats_out

end module spral_ssids_cpu_iface
