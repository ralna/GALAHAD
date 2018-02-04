function [f,status] = rosenbrock_f(x)
 f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
 status = 0;
end
