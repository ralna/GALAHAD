function [f,status] = quadratic_f(x)
 f = 100*x(1)^2 + x(2)^2;
 status = 0;
end
