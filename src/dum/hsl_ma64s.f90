    module hsl_ma64_single
      private
      public ma64_control, ma64_info, ma64_factor
      integer, parameter :: wp = kind(1.0) ! Precision parameter.
      integer,parameter:: long = selected_int_kind(18) ! Long integer.
      type ma64_control
        integer :: p_thresh=32
        real (wp) :: small=tiny(0.0_wp)
        real (wp) :: static=0.0_wp
        logical :: twos=.false.
        real (wp) :: u=0.1_wp
        real (wp) :: umin=1.0_wp
      end type ma64_control

      type ma64_info
        real (wp) :: detlog=0 
        integer :: detsign=0 
        integer :: flag=0 
        integer :: num_neg=0 
        integer :: num_nothresh=0 
        integer :: num_perturbed=0 
        integer :: num_zero=0 
        integer :: num_2x2=0 
        real (wp) :: usmall
        real (wp) :: u=0
      end type ma64_info
    contains
      subroutine ma64_factor(n,p,nb,nbi,a,la,cntl,q,ll,perm,d,                 &
                             buf,info,s,n_threads)
      integer, intent (in) :: n 
      integer, intent (in) :: p 
      integer, intent (in) :: nb 
      integer, intent (in) :: nbi 
      integer(long), intent (in) :: la 
      real (wp), intent (inout) :: a(la) 
      type(ma64_control), intent (in) :: cntl 
      integer, intent (out) :: q 
      integer(long), intent (out) :: ll 
      integer, intent (inout) :: perm(p) 
      real (wp), intent (out) :: d(2*p) 
      real (wp) :: buf(nb*n+n) 
      type(ma64_info), intent (inout) :: info
      integer, intent (in), optional :: s 
      integer, intent (inout), optional, target :: n_threads 
      end subroutine ma64_factor
    end module hsl_ma64_single
