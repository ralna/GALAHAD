modifications = Dict("unsymmetric_matrix_dense " => "",
                     "unsymmetric_matrix_coordinate" => "",
                     "unsymmetric_matrix_row_wise" => "",
                     "main_unsymmetric_matrices" => "",
                     "\\endlink " => "",
                     "\\endmanonly\n" => "",
                     "\\endhtmlonly\n" => "",
                     "\\endlatexonly\n" => "")

modifications2 = Dict("\\f\$" => "\$",
                      "\\f[" => "\$",
                      "\\f]" => "\$",
                      "\\section" => "#",
                      "\\subsection" => "##",
                      "\\subsubsection" => "###",
                      "<b>" => "**",
                      "</b>" => "**",
                      "\\e " => "",
                      "\\c " => "",
                      " */\n\n" => "",
                      "symmetric_matrix_dense " => "",
                      "symmetric_matrix_coordinate" => "",
                      "symmetric_matrix_row_wise" => "",
                      "main_symmetric_matrices" => "",
                      "symmetric_matrix_diagonal" => "",
                      "symmetric_matrix_scaled_identity" => "",
                      "symmetric_matrix_identity" => "",
                      "symmetric_matrix_zero" => "",
                      "\\link " => "",
                      "\\manonly\n" => "",
                      "\\htmlonly\n" => "",
                      "\\latexonly\n" => "")

modifications3 = Dict("``" => "“",
                      "''" => "”",
                      "_mat" => "\\_mat",
                      "_with" => "\\_with",
                      "_solve" => "\\_solve",
                      "_reverse" => "\\_reverse",
                      "_reset" => "\\_reset",
                      "_control" => "\\_control",
                      "_initialize" => "\\_initialize",
                      "_terminate" => "\\_terminate",
                      "_import" => "\\_import",
                      "_information" => "\\_information",
                      "_read" => "\\_read",
                      "_specfile" => "\\_specfile")

modifications4 = Dict("\nSee Section~\\ref{examples} for examples of use.\nSee the <a href=\"examples.html\">examples tab</a> for illustrations of use.\nSee the examples section for illustrations of use.\n" => "",
                      "\\n\nminimize q(x) := 1/2 x^T H x + g^T x + f\n\\n" => "\\[\nminimize q(x) := 1/2 x^T H x + g^T x + f\n\\]")

modifications5 = Dict("control parameters and\nset up" => "control parameters and set up",
                      "control values\nby" => "control values by",
                      "control\nparameters" => "control parameters",
                      "information about\nthe solution" => "information about the solution",
                      "fixed\nvalues" => "fixed values",
                      "calls to\n evaluate" => "calls to evaluate",
                      "to the\n calling" => "to the calling",
                      "values and\n Hessian" => "values and Hessian")

function generator(name::String, path::String)
  text = read(path, String)
  text = replace(text, "  " => "")
  text = split(text, "#ifdef __cplusplus")[1]
  texts = split(text, "/*! \\mainpage GALAHAD C package $name\n\n")
  if length(texts) ≥ 2
    text = texts[2]
  end
  for (keys, vals) in modifications
    text = replace(text, keys => vals)
  end
  for (keys, vals) in modifications2
    text = replace(text, keys => vals)
  end
  for (keys, vals) in modifications3
    text = replace(text, keys => vals)
  end
  for (keys, vals) in modifications4
    text = replace(text, keys => vals)
  end
  for (keys, vals) in modifications5
    text = replace(text, keys => vals)
  end
  for str in ("intro ", "purpose ", "authors ", "date ", "terminology ", "method ", "references ", "call_order ")
      text = replace(text, name * "_" * str => "")
  end
  text = replace(text, " - $name" => "- $name")
  write("../docs/src/$name.md", text)
end

function documentation(name::String="all")
  galahad = joinpath(ENV["GALAHAD"], "include")

  (name == "all" || name == "galahad")  && generator("galahad", "$galahad/galahad.h")
  (name == "all" || name == "arc")      && generator("arc", "$galahad/galahad_arc.h")
  (name == "all" || name == "bgo")      && generator("bgo", "$galahad/galahad_bgo.h")
  (name == "all" || name == "blls")     && generator("blls", "$galahad/galahad_blls.h")
  (name == "all" || name == "bqp")      && generator("bqp", "$galahad/galahad_bqp.h")
  (name == "all" || name == "bqpb")     && generator("bqpb", "$galahad/galahad_bqpb.h")
  (name == "all" || name == "bsc")      && generator("bsc", "$galahad/galahad_bsc.h")
  (name == "all" || name == "ccqp")     && generator("ccqp", "$galahad/galahad_ccqp.h")
  (name == "all" || name == "clls")     && generator("clls", "$galahad/galahad_clls.h")
  (name == "all" || name == "convert")  && generator("convert", "$galahad/galahad_convert.h")
  (name == "all" || name == "cqp")      && generator("cqp", "$galahad/galahad_cqp.h")
  (name == "all" || name == "cro")      && generator("cro", "$galahad/galahad_cro.h")
  (name == "all" || name == "dgo")      && generator("dgo", "$galahad/galahad_dgo.h")
  (name == "all" || name == "dps")      && generator("dps", "$galahad/galahad_dps.h")
  (name == "all" || name == "dqp")      && generator("dqp", "$galahad/galahad_dqp.h")
  (name == "all" || name == "eqp")      && generator("eqp", "$galahad/galahad_eqp.h")
  (name == "all" || name == "fdc")      && generator("fdc", "$galahad/galahad_fdc.h")
  (name == "all" || name == "fit")      && generator("fit", "$galahad/galahad_fit.h")
  (name == "all" || name == "glrt")     && generator("glrt", "$galahad/galahad_glrt.h")
  (name == "all" || name == "gls")      && generator("gls", "$galahad/galahad_gls.h")
  (name == "all" || name == "gltr")     && generator("gltr", "$galahad/galahad_gltr.h")
  (name == "all" || name == "hash")     && generator("hash", "$galahad/galahad_hash.h")
  (name == "all" || name == "icfs")     && generator("icfs", "$galahad/galahad_icfs.h")
  (name == "all" || name == "ir")       && generator("ir", "$galahad/galahad_ir.h")
  (name == "all" || name == "l2rt")     && generator("l2rt", "$galahad/galahad_l2rt.h")
  (name == "all" || name == "lhs")      && generator("lhs", "$galahad/galahad_lhs.h")
  (name == "all" || name == "lms")      && generator("lms", "$galahad/galahad_lms.h")
  (name == "all" || name == "lpa")      && generator("lpa", "$galahad/galahad_lpa.h")
  (name == "all" || name == "lpb")      && generator("lpb", "$galahad/galahad_lpb.h")
  (name == "all" || name == "lsqp")     && generator("lsqp", "$galahad/galahad_lsqp.h")
  (name == "all" || name == "lsrt")     && generator("lsrt", "$galahad/galahad_lsrt.h")
  (name == "all" || name == "lstr")     && generator("lstr", "$galahad/galahad_lstr.h")
  (name == "all" || name == "nls")      && generator("nls", "$galahad/galahad_nls.h")
  (name == "all" || name == "presolve") && generator("presolve", "$galahad/galahad_presolve.h")
  (name == "all" || name == "psls")     && generator("psls", "$galahad/galahad_psls.h")
  (name == "all" || name == "qpa")      && generator("qpa", "$galahad/galahad_qpa.h")
  (name == "all" || name == "qpb")      && generator("qpb", "$galahad/galahad_qpb.h")
  (name == "all" || name == "roots")    && generator("roots", "$galahad/galahad_roots.h")
  (name == "all" || name == "rpd")      && generator("rpd", "$galahad/galahad_rpd.h")
  (name == "all" || name == "rqs")      && generator("rqs", "$galahad/galahad_rqs.h")
  (name == "all" || name == "sbls")     && generator("sbls", "$galahad/galahad_sbls.h")
  (name == "all" || name == "scu")      && generator("scu", "$galahad/galahad_scu.h")
  (name == "all" || name == "sec")      && generator("sec", "$galahad/galahad_sec.h")
  (name == "all" || name == "sha")      && generator("sha", "$galahad/galahad_sha.h")
  (name == "all" || name == "sils")     && generator("sils", "$galahad/galahad_sils.h")
  (name == "all" || name == "slls")     && generator("slls", "$galahad/galahad_slls.h")
  (name == "all" || name == "sls")      && generator("sls", "$galahad/galahad_sls.h")
  (name == "all" || name == "trb")      && generator("trb", "$galahad/galahad_trb.h")
  (name == "all" || name == "trs")      && generator("trs", "$galahad/galahad_trs.h")
  (name == "all" || name == "tru")      && generator("tru", "$galahad/galahad_tru.h")
  (name == "all" || name == "ugo")      && generator("ugo", "$galahad/galahad_ugo.h")
  (name == "all" || name == "uls")      && generator("uls", "$galahad/galahad_uls.h")
  (name == "all" || name == "wcp")      && generator("wcp", "$galahad/galahad_wcp.h")
end
