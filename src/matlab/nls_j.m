function [J,status] = nls_j(x)
 J(1) = 2 * x(1) * x(3) ;
 J(2) = 0 ;
 J(3) = x(1)^2 ;
 J(4) = 0 ;
 J(5) = 2 * x(2) ;
 J(6) = 1 ;
 status = 0 ;
end
