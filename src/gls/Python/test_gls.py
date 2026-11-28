from galahad import gls
print("\n** python test: gls")

# the gls Python interface provides only the options and information of
# GLS, for use with other packages; see uls to solve unsymmetric systems

# allocate internal data and set default options
options = gls.initialize()
print(" pivot threshold u:", options['u'])

# get information from the analysis, factorization and solve phases
ainfo, finfo, sinfo = gls.information()
print(" analyse flag:", ainfo['flag'])
print(" factorize flag:", finfo['flag'])
print(" solve flag:", sinfo['flag'])

# deallocate internal data
gls.finalize()
