function [f,g,h,status] = wiggly_f(x)
 a = 10.0;
 ax = a * x;
 sax = sin( ax );
 cax = cos( ax );
 f = x * x * cax;
 g = - ax * x * sax + 2.0 * x * cax;
 h = - a * a* x * x * cax - 4.0 * ax * sax + 2.0 * cax;
 status = 0;
end
