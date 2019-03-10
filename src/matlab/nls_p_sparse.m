function [P,status] = nls_p_sparse(x,v)
 P(1) = 2 * x(3) * v(1) + 2 * x(1) * v(3) ;
 P(2) = 2 * x(1) * v(1) ;
 P(3) = 2 * v(2) ;
 status = 0 ;
end
