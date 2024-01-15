! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 10:15 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 3 4    M O D U L E  -*-*-*-

   module hsl_mc34_single
     use GALAHAD_KINDS
     implicit none
     private
     public mc34_expand
     interface mc34_expand
        module procedure mc34_expand_single
     end interface
   contains
     subroutine mc34_expand_single(n,row,ptr,iw,a,sym_type)
     integer(ip_),  intent(in) :: n
     integer(ip_),  intent(inout) :: row(*)
     integer(ip_),  intent(inout) ::ptr(n+1)
     integer(ip_) :: iw(n) ! workspace
     real(sp_), optional, intent(inout) :: a(*)
     integer(ip_),  optional, intent(in) :: sym_type
     end subroutine mc34_expand_single
   end module hsl_mc34_single
