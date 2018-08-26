function [g,status] = tridia_g(x)
 g(1) = 4 * (x(1) - 1)^3 + 2 * x(1) * x(2)^2;
 g(2) = 4 * (x(2) - 2)^3 + 2 * x(2) * x(1)^2 + 2 * x(2) * x(3)^2;
 g(3) = 4 * (x(3) - 3)^3 + 2 * x(3) * x(2)^2;
 status = 0;
end


