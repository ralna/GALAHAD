function [h_val,status] = rosenbrock_h(x)
 h_val(1) = - 400*x(2) + 1200*x(1)^2 + 2;
 h_val(2) = - 400 * x(1);
 h_val(3) =   200 ;
 status = 0;
end

