function [g,status] = quadratic_g(x)
 g(1) =  200*x(1);
 g(2) =  2*x(2);
 status = 0;
end


