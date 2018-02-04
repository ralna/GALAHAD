! THIS VERSION: 20/01/2011 AT 12:30:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 3 4    M O D U L E  -*-*-*-

   module hsl_mc34_single
     implicit none
     private
     public mc34_expand
     integer, parameter :: wp = kind(0.0)
     interface mc34_expand
        module procedure mc34_expand_single
     end interface
   contains
     subroutine mc34_expand_single(n,row,ptr,iw,a,sym_type)
     integer, intent(in) :: n
     integer, intent(inout) :: row(*)
     integer, intent(inout) ::ptr(n+1)
     integer :: iw(n) ! workspace
     real(wp), optional, intent(inout) :: a(*) 
     integer, optional, intent(in) :: sym_type 
     end subroutine mc34_expand_single
   end module hsl_mc34_single
