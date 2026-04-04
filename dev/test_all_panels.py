import pvlib
from breos.pv_modules import list_modules, get_module

def test_all_panels():
    modules = list_modules()
    print(f"Testing {len(modules)} modules...")
    
    success_count = 0
    fail_count = 0
    
    for mod_name in modules:
        try:
            p = get_module(mod_name)
            
            # Replicate the logic from breos/solar.py
            # converting %/C to absolute values
            if p.alpha_sc_abs is not None:
                alpha_sc = p.alpha_sc_abs
            else:
                alpha_sc = (p.T_Isc_pct * p.Isc) / 100
                
            if p.beta_voc_abs is not None:
                beta_voc = p.beta_voc_abs
            else:
                beta_voc = (p.T_Voc_pct * p.Voc) / 100
                
            gamma_pmp = p.T_Pmax_pct
            
            # Attempt to run CEC fitting logic
            result = pvlib.ivtools.sdm.fit_cec_sam(
                celltype=p.celltype,
                v_mp=p.Vmp,
                i_mp=p.Imp,
                v_oc=p.Voc,
                i_sc=p.Isc,
                alpha_sc=alpha_sc,
                beta_voc=beta_voc,
                gamma_pmp=gamma_pmp,
                cells_in_series=p.N_Cells
            )
            print(f"[\u2713] {mod_name} passed")
            success_count += 1
            
        except Exception as e:
            print(f"[\u2717] {mod_name} failed: {e}")
            fail_count += 1
            
    print(f"\nResults: {success_count} passed, {fail_count} failed")

if __name__ == "__main__":
    test_all_panels()
