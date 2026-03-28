function [Jr,status] = snls_j_sparse(x)
 Jr(1) = x(2) ;
 Jr(2) = x(1) ;
 Jr(3) = x(3) ;
 Jr(4) = x(2) ;
 Jr(5) = x(4) ;
 Jr(6) = x(3) ;
 Jr(7) = x(5) ;
 Jr(8) = x(4) ;
% Jr
 status = 0 ;
end
