function [h_val,status] = bgo_h(x)
 freq = 10.0;
 mag = 1000.0;
 h_val(1) = 2.0 - mag * freq * freq * cos(freq*x(1));
 h_val(2) = 2.0;
 h_val(3) = 2.0;
 h_val(4) = 2.0;
 h_val(5) = 4.0;
 status = 0;
end
