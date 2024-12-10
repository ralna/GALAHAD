galahad_headers += files('spral_config.h',
                         'spral_procedures.h',
                         'spral_ssids.h')

if build_ciface
  galahad_headers += files('galahad_arc.h',
                           'galahad_bgo.h',
                           'galahad_blls.h',
                           'galahad_bllsb.h',
                           'galahad_bqpb.h',
                           'galahad_bqp.h',
                           'galahad_bsc.h',
                           'galahad_ccqp.h',
                           'galahad_cfunctions.h',
                           'galahad_clls.h',
                           'galahad_convert.h',
                           'galahad_cqp.h',
                           'galahad_cro.h',
                           'galahad_dgo.h',
                           'galahad_double.h',
                           'galahad_dps.h',
                           'galahad_dqp.h',
                           'galahad_eqp.h',
                           'galahad_fdc.h',
                           'galahad_fit.h',
                           'galahad_glrt.h',
                           'galahad_gls.h',
                           'galahad_gltr.h',
                           'galahad.h',
                           'galahad_hash.h',
                           'galahad_ir.h',
                           'galahad_l2rt.h',
                           'galahad_lhs.h',
                           'galahad_llsr.h',
                           'galahad_llst.h',
                           'galahad_lms.h',
                           'galahad_lpa.h',
                           'galahad_lpb.h',
                           'galahad_lsqp.h',
                           'galahad_lsrt.h',
                           'galahad_lstr.h',
                           'galahad_modules.h',
                           'galahad_nls.h',
                           'galahad_precision.h',
                           'galahad_presolve.h',
                           'galahad_psls.h',
                           'galahad_qpa.h',
                           'galahad_qpb.h',
                           'galahad_quadruple.h',
                           'galahad_roots.h',
                           'galahad_rpd.h',
                           'galahad_rqs.h',
                           'galahad_sbls.h',
                           'galahad_scu.h',
                           'galahad_sec.h',
                           'galahad_sha.h',
                           'galahad_sils.h',
                           'galahad_single.h',
                           'galahad_slls.h',
                           'galahad_sls_defaults.h',
                           'galahad_sls.h',
                           'galahad_trb.h',
                           'galahad_trs.h',
                           'galahad_tru.h',
                           'galahad_ugo.h',
                           'galahad_uls.h',
                           'galahad_wcp.h')
endif
# 'galahad_icfs.h'

if build_pythoniface
  galahad_headers += files('galahad_python.h')
endif

if build_ssids
  galahad_headers += files('ssids_contrib.h')
endif

if not libhsl.found()
  galahad_headers += files('hsl_ma48.h',
                           'hsl_ma57.h',
                           'hsl_ma77.h',
                           'hsl_ma86.h',
                           'hsl_ma87.h',
                           'hsl_ma97.h',
                           'hsl_mc64.h',
                           'hsl_mc68.h',
                           'hsl_mi20.h',
                           'hsl_mi28.h')
endif
