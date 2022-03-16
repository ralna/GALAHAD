function [u,status] = camel6_hprod(x,u,v)
 p = -2.1;
 u(1) = u(1) + (8.0 + 12.0 * p * x(1)^2 + 10.0 * x(1)^4) * v(1) +  v(2);
 u(2) = u(2) + v(1) + (- 8.0 + 48.0 * x(2) * x(2)) * v(2);
 status = 0;
end
