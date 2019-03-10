function [P,status] = nls_p(x,v)
 P(1) = 2 * x(3) * v(1) + 2 * x(1) * v(3) ;
 P(2) = 0 ;
 P(3) = 2 * x(1) * v(1) ;
 P(4) = 0 ;
 P(5) = 2 * v(2) ;
 P(6) = 0 ;
 status = 0 ;
end
