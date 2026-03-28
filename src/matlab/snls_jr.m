function [Jr,status] = snls_j(x)
 Jr(1) = x(2) ;
 Jr(2) = x(1) ;
 Jr(3) = 0 ;
 Jr(4) = 0 ;
 Jr(5) = 0 ;
 Jr(6) = 0 ;
 Jr(7) = x(3) ;
 Jr(8) = x(2) ;
 Jr(9) = 0 ;
 Jr(10) = 0 ;
 Jr(11) = 0 ;
 Jr(12) = 0 ;
 Jr(13) = x(4) ;
 Jr(14) = x(3) ;
 Jr(15) = 0 ;
 Jr(16) = 0 ;
 Jr(17) = 0 ;
 Jr(18) = 0 ;
 Jr(19) = x(5) ;
 Jr(20) = x(4) ;
% Jr
 status = 0 ;
end
