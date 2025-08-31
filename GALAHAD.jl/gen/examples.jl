using JuliaFormatter

function examples(package::String, example::String; forthcoming::Bool=false)
  if example == ""
    text = "# test_" * package * ".jl\n# Simple code to test the Julia interface to " * uppercase(package) * "\n\n"
    text = text * "using GALAHAD\nusing Test\nusing Quadmath\n\nfunction test_$package(::Type{T}, ::Type{INT}) where {T,INT}\n"
    text = text * "  data = Ref{Ptr{Cvoid}}()\n"
    text = text * "  control = Ref{$(package)_control_type{T,INT}}()\n"
    text = text * "  inform = Ref{$(package)_inform_type{T,INT}}()\n\n"
    text = text * "  status = Ref{INT}()\n"
    text = text * "  $(package)_initialize(T, INT, data, control, status)\n"
    text = text * "  $(package)_information(T, INT, data, inform, status)\n"
    text = text * "  $(package)_terminate(T, INT, data, control, inform)\n\n"
    text = text * "  return 0\n"
  else
    subdir = forthcoming ? "../../src/forthcoming" : "../../src"
    src = "$subdir/$package/C/" * package * example * ".c"
    text = read(src, String)
    text = replace(text, "    " => "")
    text = replace(text, "int main(void) {\n\n" => "# test_$package.jl\n# Simple code to test the Julia interface to " * uppercase(package) * "\n\nusing GALAHAD\nusing Test\nusing Printf\nusing Accessors\nusing Quadmath\n\nfunction test_$package(::Type{T}, ::Type{INT}) where {T,INT}\n")
    text = replace(text, "\" d_ipc_ \"" => "d")
    text = replace(text, "ipc_" => "INT")
    text = replace(text, "rpc_" => "T")
    text = replace(text, "//" => "#")
    text = replace(text, ";" => "")
    text = replace(text, "&" => "")
    text = replace(text, "- INFINITY" => "-Inf")
    text = replace(text, "INFINITY" => "Inf")
    text = replace(text, "printf" => "@printf")
    text = replace(text, "else if" => "elseif")
    text = replace(text, "}else{" => "else")
    text = replace(text, "}elseif" => "elseif")
    text = replace(text, "} else {" => "else")
    text = replace(text, "void *data" => "data = Ref{Ptr{Cvoid}}()")
    text = replace(text, "struct $(package)_control_type control" => "control = Ref{$(package)_control_type{T,INT}}()")
    text = replace(text, "struct $(package)_inform_type inform" => "inform = Ref{$(package)_inform_type{T,INT}}()")
    text = replace(text, "control." => "control[].")
    text = replace(text, "inform." => "inform[].")
    text = replace(text, "} #" => "]  #")
    text = replace(text, "}  #" => "]  #")
    text = replace(text, "}   #" => "]  #")
    for var in ("A_val", "A_dense", "b", "c", "c_l", "c_u", "x_l", "x_u", "y_l", "y_u", "z_l", "z_u", "g", "x_0",
                "w", "x", "y", "z", "val", "dense", "rhs", "rhst", "sol", "H_val", "H_dense", "C_val",
                "C_dense", "H_diag", "C_diag", "H_scid", "C_scid", "Ao_val", "r", "M_val", "M_dense",
                "M_diag", "y", "W", "Ao_dense")
      text = replace(text, "real_wp_ $var[] = {" => "$var = T[")
    end
    for var in ("f", "power", "weight", "shift", "radius", "half_radius", "x_l", "x_u", "sigma", "rho_g", "rho_b")
      text = replace(text, "real_wp_ $var =" => "$var =")
    end
    for var in ("n", "ne", "m", "A_ne", "A_dense_ne", "H_ne", "H_dense_ne", "C_dense_ne",
                "C_ne", "dense_ne", "o", "Ao_ne", "Ao_ptr_ne", "A_ptr_ne", "m_equal",
                "M_ne", "M_dense_ne", "j_ne", "h_ne", "p_ne", "Ao_dense_ne")
      text = replace(text, "int $var =" => "$var = INT(")
    end
    for var in ("A_row", "A_col", "A_ptr", "row", "col", "ptr", "c_stat", "x_stat", "H_row", "H_col", "H_ptr",
                "C_row", "C_col", "C_ptr", "Ao_col", "Ao_ptr", "Ao_row", "M_row",
                "M_col", "M_ptr", "J_row", "J_col", "J_ptr", "P_row", "P_ptr")
      text = replace(text, "int $var[] = {" => "$var = INT[")
    end
    for val in ("1", "3", "5", "6", "7", "n", "n+m")
      text = replace(text, "for( int d=1 d <= $val d++){" => "for d = 1:$val")
      text = replace(text, "for(int d=1 d <= $val d++){" => "for d = 1:$val")
    end
    for index in ("unit_m", "new_radius", "a_is", "m_is")
      text = replace(text, "for( int $index=0 $index <= 1 $index++){" => "for $index = 0:1")
      text = replace(text, "for(int $index=0 $index <= 1 $index++){" => "for $index = 0:1")
    end
    for val in ("c", "g", "u", "v", "x", "w", "z", "x_l", "x_u", "r", "vector", "h_vector", "error")
      text = replace(text, "real_wp_ $val[n]" => "$val = zeros(T, n)")
      text = replace(text, "real_wp_ $val[m]" => "$val = zeros(T, m)")
      text = replace(text, "real_wp_ $val[o]" => "$val = zeros(T, o)")
    end
    for val in ("x_stat", "c_stat", "index_nz_u", "index_nz_v", "depen")
      text = replace(text, "int $val[n]" => "$val = zeros(INT, n)")
      text = replace(text, "int $val[m]" => "$val = zeros(INT, m)")
    end

    text = replace(text, "if(" => "if ")
    text = replace(text, "){" => ")")
    text = replace(text, ")\n}\n}" => ")\n")
    text = replace(text, ")\n}" => ")\n")
    text = replace(text, "\n\n\n" => "\n")
    text = replace(text, "}\n" => "]\n")
    text = replace(text, "NULL" => "C_NULL")
    text = replace(text, "char st" => "st = ' '")
    text = replace(text, "int status" => "status = Ref{INT}()")
    text = replace(text, "int n_depen" => "n_depen = Ref{INT}()")
    text = replace(text, "real_wp_ radius" => "radius = Ref{T}()")
    text = replace(text, " ]" => "]")
    text = replace(text, "== 0)" => "== 0")
    text = replace(text, "case 1: # sparse co-ordinate storage" => "# sparse co-ordinate storage\nif d == 1")
    text = replace(text, "case 2: # sparse by rows" => "# sparse by rows\nif d == 2")
    text = replace(text, "case 3: # dense" => "# dense\nif d == 3")
    text = replace(text, "case 4: # diagonal" => "# diagonal\nif d == 4")
    text = replace(text, "case 5: # scaled identity" => "# scaled identity\nif d == 5")
    text = replace(text, "case 6: # identity" => "# identity\nif d == 6")
    text = replace(text, "case 7: # zero" => "# zero\nif d == 7")
    text = replace(text, "break\n" => "end\n")
    text = replace(text, "int maxabsarray(real_wp_ a[], int n, real_wp_ *maxabs)" => "maxabsarray(a) = maximum(abs.(a))")
    for i = 0:5
      text = replace(text, "( status == $i ) {" => "status == $i")
    end
    text = replace(text, "( status < 0 ) {" => "status < 0")
    text = replace(text, "while(true)" => "while true")
    text = replace(text, "for( int i = 0 i < n i++)" => "for i = 1:n")
    text = replace(text, "for( int i = 0 i < m i++)" => "for i = 1:m")
    text = replace(text, "for( int i = 1 i < n i++)" => "for i = 2:n")
    text = replace(text, "for( int i = 1 i < m i++)" => "for i = 2:m")
    text = replace(text, "for( int i = 2 i < n i++)" => "for i = 3:n")
    text = replace(text, "for( int i = 2 i < m i++)" => "for i = 3:m")
    text = replace(text, "constrastatus = Ref{Cint}()" => "constraint status")
    text = replace(text, "}#" => "]  #")
    text = replace(text, "#for" => "# for")
    text = replace(text, "#@" => "# @")
    text = replace(text, "# for i = 1:n @" => "# for i = 1:n\n#   @")
    text = replace(text, "( " => "(")
    text = replace(text, " )" => ")")
    text = replace(text, "switch(d)\n" => "")
    text = replace(text, "for(i=0 i<n i++)" => "for i = 1:n\n")
    text = replace(text, "}\n" => "end\n")
    for var in ("x", "u", "v", "hval", "g")
      text = replace(text, "const real_wp_ $var[]" => "var::Vector{T}")
      text = replace(text, "real_wp_ $var[]" => "$var::Vector{T}")
    end
  end

  text = text * "end\n
for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset \"$(uppercase(package)) -- \$T -- \$INT\" begin
      @test test_$package(T, INT) == 0
    end
  end
end\n"
  dst = "../test/test_" * package * ".jl"
  write(dst, text)

  # Generate a symbolic link for the Julia tests
  current_folder = pwd()
  if forthcoming
    cd("../../src/forthcoming/$package")
  else
    cd("../../src/$package")
  end
  !isdir("Julia") && mkdir("Julia")
  cd("Julia")
  rm("test_$package.jl", force=true)
  if forthcoming
    symlink("../../../../GALAHAD.jl/test/test_$package.jl", "test_$package.jl")
  else
    symlink("../../../GALAHAD.jl/test/test_$package.jl", "test_$package.jl")
  end
  cd(current_folder)
  return nothing
end

function main(name::String)
  (name == "arc")      && examples("arc"     , "tf")
  (name == "bgo")      && examples("bgo"     , "tf")
  (name == "blls")     && examples("blls"    , "tf")
  (name == "bllsb")    && examples("bllsb"   , "tf")
  (name == "bnls")     && examples("bnls"    , "tf", forthcoming=true)
  (name == "bqp")      && examples("bqp"     , "tf")
  (name == "bqpb")     && examples("bqpb"    , "tf")
  (name == "bsc")      && examples("bsc"     , ""  )
  (name == "ccqp")     && examples("ccqp"    , "tf")
  (name == "clls")     && examples("clls"    , "tf")
  (name == "convert")  && examples("convert" , ""  )
  (name == "cqp")      && examples("cqp"     , "tf")
  (name == "cro")      && examples("cro"     , "tf")
  (name == "dgo")      && examples("dgo"     , "tf")
  (name == "dps")      && examples("dps"     , "tf")
  (name == "dqp")      && examples("dqp"     , "tf")
  (name == "eqp")      && examples("eqp"     , "tf")
  (name == "expo")     && examples("expo"    , "tf", forthcoming=true)
  (name == "fdc")      && examples("fdc"     , "tf")
  (name == "fit")      && examples("fit"     , ""  )
  (name == "glrt")     && examples("glrt"    , "t" )
  (name == "gls")      && examples("gls"     , ""  )
  (name == "gltr")     && examples("gltr"    , "t" )
  (name == "hash")     && examples("hash"    , ""  )
  (name == "ir")       && examples("ir"      , ""  )
  (name == "l2rt")     && examples("l2rt"    , "t" )
  (name == "lhs")      && examples("lhs"     , "t" )
  (name == "llsr")     && examples("llsr"    , "tf")
  (name == "llst")     && examples("llst"    , "tf")
  (name == "lms")      && examples("lms"     , ""  )
  (name == "lpa")      && examples("lpa"     , "tf")
  (name == "lpb")      && examples("lpb"     , "tf")
  (name == "lsqp")     && examples("lsqp"    , "tf")
  (name == "lsrt")     && examples("lsrt"    , "t" )
  (name == "lstr")     && examples("lstr"    , "t" )
  (name == "nls")      && examples("nls"     , "tf")
  (name == "nodend")   && examples("nodend"  , "tf")
  (name == "presolve") && examples("presolve", "tf")
  (name == "psls")     && examples("psls"    , "tf")
  (name == "qpa")      && examples("qpa"     , "tf")
  (name == "qpb")      && examples("qpb"     , "tf")
  (name == "roots")    && examples("roots"   , ""  )
  (name == "rpd")      && examples("rpd"     , "tf")
  (name == "rqs")      && examples("rqs"     , "tf")
  (name == "sbls")     && examples("sbls"    , "tf")
  (name == "scu")      && examples("scu"     , ""  )
  (name == "sec")      && examples("sec"     , ""  )
  (name == "sha")      && examples("sha"     , ""  )
  (name == "sils")     && examples("sils"    , ""  )
  (name == "slls")     && examples("slls"    , "tf")
  (name == "sls")      && examples("sls"     , "tf")
  (name == "ssids")    && examples("ssids"   , "tf")
  (name == "ssls")     && examples("ssls"    , "tf")
  (name == "trb")      && examples("trb"     , "tf")
  (name == "trs")      && examples("trs"     , "tf")
  (name == "tru")      && examples("tru"     , "tf")
  (name == "ugo")      && examples("ugo"     , "t" )
  (name == "uls")      && examples("uls"     , "tf")
  (name == "version")  && examples("version" , "t" )
  (name == "wcp")      && examples("wcp"     , "tf")
  return nothing
end

function clean_example(package::String)
  path = "../test/test_" * package * ".jl"
  isfile(path) || error("The file test_$package.jl doesn't exist.")
  format_file(path, YASStyle(), indent=2)
end
