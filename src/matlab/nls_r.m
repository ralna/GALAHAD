function [r,status] = nls_r(x)
 p = 4 ;
 r(1) = x(1)^2 * x(3) + p ;
 r(2) = x(2)^2 + x(3) ;
 status = 0 ;
end
