function [f,status] = bgo_f(x)
 p = 4.0;
 freq = 10.0;
 mag = 1000.0;
 f = (x(1) + x(3) + p)^2 + (x(2) + x(3))^2 + mag * cos(freq*x(1)) ...
      + x(1) + x(2) + x(3);
 status = 0;
end
