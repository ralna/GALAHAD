function [f,status] = tridia_f(x)
 f = (x(1) - 1)^4 + (x(2) - 2)^4 + (x(3) - 3)^4 + x(1)^2 * x(2)^2 + ...
     x(2)^2 * x(3)^2 ;
 status = 0;
end
