using Test, GALAHAD

@info("GALAHAD_INSTALLATION : $(GALAHAD.GALAHAD_INSTALLATION)")

packages = ("arc", "bgo", "blls", "bqp", "bqpb", "bsc", "ccqp", "convert",
            "cqp", "cro", "dgo", "dps", "dqp", "eqp", "fdc", "fit", "glrt",
            "gls", "gltr", "hash", "ir", "l2rt", "lhs", "llsr", "llst", "lms",
            "lpa", "lpb", "lsqp", "lsrt", "lstr", "nls", "presolve", "psls",
            "qpa", "qpb", "roots", "rpd", "rqs", "sbls", "scu", "sec", "sha",
            "sils", "slls", "sls", "trb", "trs", "tru", "ugo", "uls", "wcp")

include("test_structures.jl")

# for package in packages
#   @testset "$package" begin
#     include("test_$package.jl")
#   end
# end
