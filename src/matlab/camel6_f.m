function [f,status] = camel6_f(x)
 p = -2.1;
 f = (4.0 + p * x(1)^2 + x(1)^4 / 3.0) * x(1)^2 + x(1) * x(2) ...
      + (- 4.0 + 4.0 * x(2)^2) * x(2)^2;
 status = 0;
end
