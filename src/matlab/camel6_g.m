function [g,status] = camel6_g(x)
 p = -2.1;
 g(1) = (8.0 + 4.0 * p * x(1)^2 + 2.0 * x(1)^4) * x(1) + x(2);
 g(2) = x(1) + (- 8.0 + 16.0 * x(2)^2) * x(2);
 status = 0;
end
