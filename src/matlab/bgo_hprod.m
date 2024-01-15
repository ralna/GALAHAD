function [u,status] = bgo_hprod(x,u,v)
   freq = 10.0;
   mag = 1000.0;
   u(1) = u(1) + (2.0 - mag * freq * freq * cos(freq*x(1))) * v(1) ...
            + 2.0 * v(3);
   u(2) = u(2) + 2.0 * (v(2) + v(3)) ;
   u(3) = u(3) + 2.0 * (v(1) + v(2) + 2.0 * v(3));
   status = 0;
end
