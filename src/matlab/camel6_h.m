function [h_val,status] = camel6_h(x)
 p = -2.1;
 h_val(1) = 8.0 + 12.0 * p * x(1)^2 + 10.0 * x(1)^4;
 h_val(2) = 1.0;
 h_val(3) = - 8.0 + 48.0 * x(2) * x(2);
 status = 0;
end
