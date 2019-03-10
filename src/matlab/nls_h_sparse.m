function [H,status] = nls_h_sparse(x,y)
 H(1) = 2 * x(3) * y(1) ;
 H(2) = 2 * y(2) ;
 H(3) = 2 * x(1) * y(1) ;
 status = 0 ;
end
