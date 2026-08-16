import time, numpy as np, pickle, warnings
import session_analysis as sa
p='sub-CSH-ZAD-024/sub-CSH-ZAD-024_ses-8207abc6-6b23-4762-92b4-82e05bed5143_desc-processed_behavior+ecephys.nwb'
t0=time.time()
r=sa.analyze_session(p, n_pseudo=200, keep_arrays=True, verbose=True)
print('elapsed', time.time()-t0)
for k in ['n_units','n_trials','n_zero','bias_zero_contrast','bias_chi2_p',
          'auc_block','p_block','auc_block_move','p_block_move','auc_block_resid','p_block_resid',
          'wheel_absvel_left','wheel_absvel_right','wheel_absvel_p',
          'auc_choice_zero','p_choice_zero','auc_choice_zero_move','p_choice_zero_move',
          'auc_crossdecode_choice','p_crossdecode','auc_crossdecode_block',
          'resid_coef','resid_p','block_coef','block_p','frac_units_sig']:
    print(f'{k:26s}', np.round(r[k],4) if isinstance(r[k],float) else r[k])
print('null_block mean/95', r['null_block'].mean().round(3), np.percentile(r['null_block'],95).round(3))
print('time auc  ', np.round(r['time_auc_block'],3))
print('time null95', np.round(r['time_null95'],3))
print(r['region_auc'].sort_values('auc',ascending=False).to_string(index=False))
pickle.dump(r, open('proto_result.pkl','wb'))
