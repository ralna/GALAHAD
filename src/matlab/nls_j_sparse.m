function [J,status] = nls_j_sparse(x)
 J(1) = 2 * x(1) * x(3) ;
 J(2) = x(1)^2 ;
 J(3) = 2 * x(2) ;
 J(4) = 1 ;
 status = 0 ;
end
