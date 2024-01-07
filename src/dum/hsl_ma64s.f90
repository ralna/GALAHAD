! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 10:15 GMT.

    module hsl_ma64_single
      use GALAHAD_KINDS
      private
      public ma64_control, ma64_info, ma64_factor
      type ma64_control
        integer(ip_) :: p_thresh=32
        real (sp_) :: small=tiny(0.0_sp_)
        real (sp_) :: static=0.0_sp_
        logical :: twos=.false.
        real (sp_) :: u=0.1_sp_
        real (sp_) :: umin=1.0_sp_
      end type ma64_control

      type ma64_info
        real (sp_) :: detlog=0 
        integer(ip_) :: detsign=0 
        integer(ip_) :: flag=0 
        integer(ip_) :: num_neg=0 
        integer(ip_) :: num_nothresh=0 
        integer(ip_) :: num_perturbed=0 
        integer(ip_) :: num_zero=0 
        integer(ip_) :: num_2x2=0 
        real (sp_) :: usmall
        real (sp_) :: u=0
      end type ma64_info
    contains
      subroutine ma64_factor(n,p,nb,nbi,a,la,cntl,q,ll,perm,d,                 &
                             buf,info,s,n_threads)
      integer(ip_),  intent (in) :: n 
      integer(ip_),  intent (in) :: p 
      integer(ip_),  intent (in) :: nb 
      integer(ip_),  intent (in) :: nbi 
      integer(long_), intent (in) :: la 
      real (sp_), intent (inout) :: a(la) 
      type(ma64_control), intent (in) :: cntl 
      integer(ip_),  intent (out) :: q 
      integer(long_), intent (out) :: ll 
      integer(ip_),  intent (inout) :: perm(p) 
      real (sp_), intent (out) :: d(2*p) 
      real (sp_) :: buf(nb*n+n) 
      type(ma64_info), intent (inout) :: info
      integer(ip_),  intent (in), optional :: s 
      integer(ip_),  intent (inout), optional, target :: n_threads 
      end subroutine ma64_factor
    end module hsl_ma64_single
