function [g,status] = rosenbrock_g(x)
 g(1) =  - 400*(x(2) - x(1)^2) * x(1) + 2*(x(1) - 1);
 g(2) =    200*(x(2) - x(1)^2) ;
 status = 0;
end


