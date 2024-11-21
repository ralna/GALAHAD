#ifdef INTEGER_64
#define ma48_control_r ma48_control_s_64
#define ma48_ainfo_r ma48_ainfo_s_64
#define ma48_finfo_r ma48_finfo_s_64
#define ma48_sinfo_r ma48_sinfo_s_64
#define ma48_initialize_r ma48_initialize_s_64
#define ma48_default_control_r ma48_default_control_s_64
#define ma48_analyse_r ma48_analyse_s_64
#define ma48_get_perm_r ma48_get_perm_s_64
#define ma48_factorize_r ma48_factorize_s_64
#define ma48_solve_r ma48_solve_s_64
#define ma48_finalize_r ma48_finalize_s_64
#define ma48_special_rows_and_cols_r ma48_special_rows_and_cols_s_64
#define ma48_determinant_r ma48_determinant_s_64
#else
#define ma48_control_r ma48_control_s
#define ma48_ainfo_r ma48_ainfo_s
#define ma48_finfo_r ma48_finfo_s
#define ma48_sinfo_r ma48_sinfo_s
#define ma48_initialize_r ma48_initialize_s
#define ma48_default_control_r ma48_default_control_s
#define ma48_analyse_r ma48_analyse_s
#define ma48_get_perm_r ma48_get_perm_s
#define ma48_factorize_r ma48_factorize_s
#define ma48_solve_r ma48_solve_s
#define ma48_finalize_r ma48_finalize_s
#define ma48_special_rows_and_cols_r ma48_special_rows_and_cols_s
#define ma48_determinant_r ma48_determinant_s
#endif

#ifdef INTEGER_64
#define ma57_default_control_r      ma57_default_control_s_64
#define ma57_init_factors_r         ma57_init_factors_s_64
#define ma57_control_r              ma57_control_s_64
#define ma57_ainfo_r                ma57_ainfo_s_64
#define ma57_finfo_r                ma57_finfo_s_64
#define ma57_sinfo_r                ma57_sinfo_s_64
#define ma57_analyse_r              ma57_analyse_s_64
#define ma57_factorize_r            ma57_factorize_s_64
#define ma57_solve_r                ma57_solve_s_64
#define ma57_finalize_r             ma57_finalize_s_64
#define ma57_enquire_perm_r    	  ma57_enquire_perm_s_64
#define ma57_enquire_pivots_r  	  ma57_enquire_pivots_s_64
#define ma57_enquire_d_r       	  ma57_enquire_d_s_64
#define ma57_enquire_perturbation_r ma57_enquire_perturbation_s_64
#define ma57_enquire_scaling_r      ma57_enquire_scaling_s_64
#define ma57_alter_d_r              ma57_alter_d_s_64
#define ma57_part_solve_r           ma57_part_solve_s_64
#define ma57_sparse_lsolve_r        ma57_sparse_lsolve_s_64
#define ma57_fredholm_alternative_r ma57_fredholm_alternative_s_64
#define ma57_lmultiply_r            ma57_lmultiply_s_64
#define ma57_get_factors_r          ma57_get_factors_s_64
#else
#define ma57_default_control_r      ma57_default_control_s
#define ma57_init_factors_r         ma57_init_factors_s
#define ma57_control_r              ma57_control_s
#define ma57_ainfo_r                ma57_ainfo_s
#define ma57_finfo_r                ma57_finfo_s
#define ma57_sinfo_r                ma57_sinfo_s
#define ma57_analyse_r              ma57_analyse_s
#define ma57_factorize_r            ma57_factorize_s
#define ma57_solve_r                ma57_solve_s
#define ma57_finalize_r             ma57_finalize_s
#define ma57_enquire_perm_r    	  ma57_enquire_perm_s
#define ma57_enquire_pivots_r  	  ma57_enquire_pivots_s
#define ma57_enquire_d_r       	  ma57_enquire_d_s
#define ma57_enquire_perturbation_r ma57_enquire_perturbation_s
#define ma57_enquire_scaling_r      ma57_enquire_scaling_s
#define ma57_alter_d_r              ma57_alter_d_s
#define ma57_part_solve_r           ma57_part_solve_s
#define ma57_sparse_lsolve_r        ma57_sparse_lsolve_s
#define ma57_fredholm_alternative_r ma57_fredholm_alternative_s
#define ma57_lmultiply_r            ma57_lmultiply_s
#define ma57_get_factors_r          ma57_get_factors_s
#endif

#ifdef INTEGER_64
#define ma77_control_r ma77_control_s_64
#define ma77_info_r ma77_info_s_64
#define ma77_default_control_r ma77_default_control_s_64
#define ma77_open_nelt_r ma77_open_nelt_s_64
#define ma77_open_r ma77_open_s_64
#define ma77_input_vars_r ma77_input_vars_s_64
#define ma77_input_reals_r ma77_input_reals_s_64
#define ma77_analyse_r ma77_analyse_s_64
#define ma77_factor_r ma77_factor_s_64
#define ma77_factor_solve_r ma77_factor_solve_s_64
#define ma77_solve_r ma77_solve_s_64
#define ma77_resid_r ma77_resid_s_64
#define ma77_scale_r ma77_scale_s_64
#define ma77_enquire_posdef_r ma77_enquire_posdef_s_64
#define ma77_enquire_indef_r ma77_enquire_indef_s_64
#define ma77_alter_r ma77_alter_s_64
#define ma77_restart_r ma77_restart_s_64
#define ma77_finalise_r ma77_finalise_s_64
#define ma77_solve_fredholm_r ma77_solve_fredholm_s_64
#define ma77_lmultiply_r ma77_lmultiply_s_64
#else
#define ma77_control_r ma77_control_s
#define ma77_info_r ma77_info_s
#define ma77_default_control_r ma77_default_control_s
#define ma77_open_nelt_r ma77_open_nelt_s
#define ma77_open_r ma77_open_s
#define ma77_input_vars_r ma77_input_vars_s
#define ma77_input_reals_r ma77_input_reals_s
#define ma77_analyse_r ma77_analyse_s
#define ma77_factor_r ma77_factor_s
#define ma77_factor_solve_r ma77_factor_solve_s
#define ma77_solve_r ma77_solve_s
#define ma77_resid_r ma77_resid_s
#define ma77_scale_r ma77_scale_s
#define ma77_enquire_posdef_r ma77_enquire_posdef_s
#define ma77_enquire_indef_r ma77_enquire_indef_s
#define ma77_alter_r ma77_alter_s
#define ma77_restart_r ma77_restart_s
#define ma77_finalise_r ma77_finalise_s
#define ma77_solve_fredholm_r ma77_solve_fredholm_s
#define ma77_lmultiply_r ma77_lmultiply_s
#endif

#ifdef INTEGER_64
#define ma86_control_r ma86_control_s_64
#define ma86_info_r ma86_info_s_64
#define ma86_default_control_r ma86_default_control_s_64
#define ma86_analyse_r ma86_analyse_s_64
#define ma86_factor_r ma86_factor_s_64
#define ma86_factor_solve_r ma86_factor_solve_s_64
#define ma86_solve_r ma86_solve_s_64
#define ma86_finalise_r ma86_finalise_s_64
#else
#define ma86_control_r ma86_control_s
#define ma86_info_r ma86_info_s
#define ma86_default_control_r ma86_default_control_s
#define ma86_analyse_r ma86_analyse_s
#define ma86_factor_r ma86_factor_s
#define ma86_factor_solve_r ma86_factor_solve_s
#define ma86_solve_r ma86_solve_s
#define ma86_finalise_r ma86_finalise_s
#endif

#ifdef INTEGER_64
#define ma87_control_r ma87_control_s_64
#define ma87_info_r ma87_info_s_64
#define ma87_default_control_r ma87_default_control_s_64
#define ma87_analyse_r ma87_analyse_s_64
#define ma87_factor_r ma87_factor_s_64
#define ma87_factor_solve_r ma87_factor_solve_s_64
#define ma87_solve_r ma87_solve_s_64
#define ma87_sparse_fwd_solve_r ma87_sparse_fwd_solve_s_64
#define ma87_finalise_r ma87_finalise_s_64
#else
#define ma87_control_r ma87_control_s
#define ma87_info_r ma87_info_s
#define ma87_default_control_r ma87_default_control_s
#define ma87_analyse_r ma87_analyse_s
#define ma87_factor_r ma87_factor_s
#define ma87_factor_solve_r ma87_factor_solve_s
#define ma87_solve_r ma87_solve_s
#define ma87_sparse_fwd_solve_r ma87_sparse_fwd_solve_s
#define ma87_finalise_r ma87_finalise_s
#endif

#ifdef INTEGER_64
#define ma97_control_r ma97_control_s_64
#define ma97_info_r ma97_info_s_64
#define ma97_default_control_r ma97_default_control_s_64
#define ma97_analyse_r ma97_analyse_s_64
#define ma97_analyse_coord_r ma97_analyse_coord_s_64
#define ma97_factor_r ma97_factor_s_64
#define ma97_factor_solve_r ma97_factor_solve_s_64
#define ma97_solve_r ma97_solve_s_64
#define ma97_free_akeep_r ma97_free_akeep_s_64
#define ma97_free_fkeep_r ma97_free_fkeep_s_64
#define ma97_finalise_r ma97_finalise_s_64
#define ma97_enquire_posdef_r ma97_enquire_posdef_s_64
#define ma97_enquire_indef_r ma97_enquire_indef_s_64
#define ma97_alter_r ma97_alter_s_64
#define ma97_solve_fredholm_r ma97_solve_fredholm_s_64
#define ma97_lmultiply_r ma97_lmultiply_s_64
#define ma97_sparse_fwd_solve_r ma97_sparse_fwd_solve_s_64
#else
#define ma97_control_r ma97_control_s
#define ma97_info_r ma97_info_s
#define ma97_default_control_r ma97_default_control_s
#define ma97_analyse_r ma97_analyse_s
#define ma97_analyse_coord_r ma97_analyse_coord_s
#define ma97_factor_r ma97_factor_s
#define ma97_factor_solve_r ma97_factor_solve_s
#define ma97_solve_r ma97_solve_s
#define ma97_free_akeep_r ma97_free_akeep_s
#define ma97_free_fkeep_r ma97_free_fkeep_s
#define ma97_finalise_r ma97_finalise_s
#define ma97_enquire_posdef_r ma97_enquire_posdef_s
#define ma97_enquire_indef_r ma97_enquire_indef_s
#define ma97_alter_r ma97_alter_s
#define ma97_solve_fredholm_r ma97_solve_fredholm_s
#define ma97_lmultiply_r ma97_lmultiply_s
#define ma97_sparse_fwd_solve_r ma97_sparse_fwd_solve_s
#endif

#ifdef INTEGER_64
#define mc64_control_r mc64_control_s_64
#define mc64_info_r mc64_info_s_64
#define mc64_default_control_r mc64_default_control_s_64
#define mc64_matching_r mc64_matching_s_64
#else
#define mc64_control_r mc64_control_s
#define mc64_info_r mc64_info_s
#define mc64_default_control_r mc64_default_control_s
#define mc64_matching_r mc64_matching_s
#endif

#ifdef INTEGER_64
#define mi20_default_control_r mi20_default_control_s_64
#define mi20_control_r mi20_control_s_64
#define mi20_default_solve_control_r mi20_default_solve_control_s_64
#define mi20_solve_control_r mi20_solve_control_s_64
#define mi20_info_r mi20_info_s_64
#define mi20_setup_r mi20_setup_s_64
#define mi20_setup_csr_r mi20_setup_csr_s_64
#define mi20_setup_csc_r mi20_setup_csc_s_64
#define mi20_setup_coord_r mi20_setup_coord_s_64
#define mi20_finalize_r mi20_finalize_s_64
#define mi20_precondition_r mi20_precondition_s_64
#define mi20_solve_r mi20_solve_s_64
#else
#define mi20_default_control_r mi20_default_control_s
#define mi20_control_r mi20_control_s
#define mi20_default_solve_control_r mi20_default_solve_control_s
#define mi20_solve_control_r mi20_solve_control_s
#define mi20_info_r mi20_info_s
#define mi20_setup_r mi20_setup_s
#define mi20_setup_csr_r mi20_setup_csr_s
#define mi20_setup_csc_r mi20_setup_csc_s
#define mi20_setup_coord_r mi20_setup_coord_s
#define mi20_finalize_r mi20_finalize_s
#define mi20_precondition_r mi20_precondition_s
#define mi20_solve_r mi20_solve_s
#endif

#ifdef INTEGER_64
#define mi28_control_r mi28_control_s_64
#define mi28_info_r mi28_info_s_64
#define mi28_default_control_r mi28_default_control_s_64
#define mi28_factorize_r mi28_factorize_s_64
#define mi28_precondition_r mi28_precondition_s_64
#define mi28_solve_r mi28_solve_s_64
#define mi28_finalise_r mi28_finalise_s_64
#else
#define mi28_control_r mi28_control_s
#define mi28_info_r mi28_info_s
#define mi28_default_control_r mi28_default_control_s
#define mi28_factorize_r mi28_factorize_s
#define mi28_precondition_r mi28_precondition_s
#define mi28_solve_r mi28_solve_s
#define mi28_finalise_r mi28_finalise_s
#endif
