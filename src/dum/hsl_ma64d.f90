! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 10:15 GMT.

    module hsl_ma64_double
      use GALAHAD_KINDS
      private
      public ma64_control, ma64_info, ma64_factor
      type ma64_control
        integer(ip_) :: p_thresh=32
        real (dp_) :: small=tiny(0.0_dp_)
        real (dp_) :: static=0.0_dp_
        logical(lp_) :: twos=.false.
        real (dp_) :: u=0.1_dp_
        real (dp_) :: umin=1.0_dp_
      end type ma64_control

      type ma64_info
        real (dp_) :: detlog=0
        integer(ip_) :: detsign=0
        integer(ip_) :: flag=0
        integer(ip_) :: num_neg=0
        integer(ip_) :: num_nothresh=0
        integer(ip_) :: num_perturbed=0
        integer(ip_) :: num_zero=0
        integer(ip_) :: num_2x2=0
        real (dp_) :: usmall
        real (dp_) :: u=0
      end type ma64_info
    contains
      subroutine ma64_factor(n,p,nb,nbi,a,la,cntl,q,ll,perm,d,                 &
                             buf,info,s,n_threads)
      integer(ip_),  intent (in) :: n
      integer(ip_),  intent (in) :: p
      integer(ip_),  intent (in) :: nb
      integer(ip_),  intent (in) :: nbi
      integer(long_), intent (in) :: la
      real (dp_), intent (inout) :: a(la)
      type(ma64_control), intent (in) :: cntl
      integer(ip_),  intent (out) :: q
      integer(long_), intent (out) :: ll
      integer(ip_),  intent (inout) :: perm(p)
      real (dp_), intent (out) :: d(2*p)
      real (dp_) :: buf(nb*n+n)
      type(ma64_info), intent (inout) :: info
      integer(ip_),  intent (in), optional :: s
      integer(ip_),  intent (inout), optional, target :: n_threads
      end subroutine ma64_factor
    end module hsl_ma64_double
