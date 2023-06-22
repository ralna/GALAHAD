clear;

ntests=0;         % Number of tests run
pkg_failures=0;   % Nonzero error return code from package
syntax_errors=0;  % Syntax error or other
err = 'Error while testing package ';

%try
%    test_galahad_cqp
%    if inform.status ~= 0
%        pkg_failures = pkg_failures + 1;
%    end
%catch excpt
%    disp(sprintf('%d %d\n', err, 'CQP'));
%    syntax_errors = syntax_errors + 1;
%end

try
    pkg = 'ARC';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_arc
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'BGO';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_bgo
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'BQP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_bqp
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'BQPB';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_bqpb
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'BLLS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_blls
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'SLLS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_slls
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'DGO';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_dgo
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'LPB';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_lpb
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'LPA';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_lpa
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'CQP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_cqp
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'DQP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_dqp
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'EQP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_eqp
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end


try
    pkg = 'GLRT';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_glrt
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'GLTR';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_gltr
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'L2RT';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_l2rt
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'LLST';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_llst
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'LSQP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_lsqp
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'LSRT';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_lsrt
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'LSTR';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_lstr
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'NLS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_nls
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'QPA';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_qpa
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'QPB';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_qpb
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'QPC';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_qpc
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'QP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_qpc
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'RQS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_rqs
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'SBLS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_sbls
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

%try
%    pkg = 'SILS';
%    disp(sprintf('******* Testing %s *******', pkg));
%    ntests = ntests + 1;
%    test_galahad_sils
%    if inform.flag ~= 0
%        pkg_failures = pkg_failures + 1;
%    end
%catch excpt
%    disp(sprintf('%d %d\n', err, pkg));
%    syntax_errors = syntax_errors + 1;
%end

try
    pkg = 'SLS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_sls
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'TRB';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_trb
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'TRS';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_trs
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'TRU';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_tru
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'UGO';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_ugo
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

try
    pkg = 'WCP';
    disp(sprintf('******* Testing %s *******', pkg));
    ntests = ntests + 1;
    test_galahad_wcp
    if inform.status ~= 0
        pkg_failures = pkg_failures + 1;
    end
catch excpt
    disp(sprintf('%d %d\n', err, pkg));
    syntax_errors = syntax_errors + 1;
end

disp(sprintf('******* End of tests *******'));
disp(sprintf('Total tests run:    %d', ntests));
disp(sprintf('Total failures:     %d', pkg_failures));
disp(sprintf('Total fatal errors: %d', syntax_errors));
disp(sprintf('****************************\n'));

clear;
