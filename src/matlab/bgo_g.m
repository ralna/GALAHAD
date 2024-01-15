function [g,status] = bgo_g(x)
 p = 4.0;
 freq = 10.0;
 mag = 1000.0;
 g(1) = 2.0 * (x(1) + x(3) + p) - mag * freq * sin(freq*x(1)) + 1.0;
 g(2) = 2.0 * (x(2) + x(3)) + 1.0;
 g(3) = 2.0 * (x(1) + x(3) + p) + 2.0 * (x(2) + x(3)) + 1.0;
 status = 0;
end
