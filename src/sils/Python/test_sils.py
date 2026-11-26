from galahad import sils
print("\n** python test: sils")

# the sils Python interface provides only the options and information of
# SILS, for use with other packages; see sls to solve symmetric systems

# allocate internal data and set default options
options = sils.initialize()
print(" pivoting:", options['pivoting'])

# get information from the analysis, factorization and solve phases
ainfo, finfo, sinfo = sils.information()
print(" analyse flag:", ainfo['flag'])
print(" factorize flag:", finfo['flag'])
print(" solve flag:", sinfo['flag'])

# deallocate internal data
sils.finalize()
