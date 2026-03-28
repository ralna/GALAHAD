function [r,status] = snls_r(x)
 p = 4 ;
 r(1) = x(1) * x(2) - p ;
 r(2) = x(2) * x(3) - 1 ;
 r(3) = x(3) * x(4) - 1 ;
 r(4) = x(4) * x(5) - 1 ;
%r
 status = 0 ;
end
