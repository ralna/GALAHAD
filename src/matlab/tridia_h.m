function [h_val,status] = tridia_h(x)
 h_val(1) = 12 * (x(1) - 1)^2 + 2 * x(2)^2;
 h_val(2) = 4 * x(1) * x(2);
 h_val(3) = 12 * (x(2) - 2)^2 + 2 * x(1)^2 + 2 * x(3)^2;
 h_val(4) = 4 * x(2) * x(3);
 h_val(5) = 12 * (x(3) - 3)^2 + 2 * x(2)^2;
 status = 0;
end

