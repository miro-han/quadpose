"""Inference input contract. No evaluator, simulator, oracle or label argument."""
import numpy as np

CAMERA_FIELDS=("cx_m","cy_m","cz_m","fx_over_width","fy_over_height","cx_over_width","cy_over_height",
               *(f"Rcw_{i}{j}" for i in range(3) for j in range(3)),
               "corners","edges","local_valid","median_all_px","all_residual_available")
HYPOTHESIS_FIELDS=("tx_m","ty_m","tz_m",*(f"Rwo_{i}{j}" for i in range(3) for j in range(3)),
                   "valid","fit_rms_px","log1p_condition")
# Display order T123, T124, T134, T234. Solver order is omitted camera 1,2,3,4.
OMITTED=np.array([3,2,1,0])


def inference_packet(evidence):
    membership=(OMITTED[:,None]!=np.arange(4))
    camera_rows=[]
    observed=[]
    for j,(view,fit) in enumerate(zip(evidence.matches,evidence.local,strict=True)):
        cam=view.camera
        w,h=cam.image_size
        K=cam.intrinsic
        residual=evidence.all_residuals[j]
        camera_rows.append([*cam.center,K[0,0]/w,K[1,1]/h,K[0,2]/w,K[1,2]/h,*cam.rotation.ravel(),
                            view.corners,view.edges,fit.valid,residual if np.isfinite(residual) else 0,
                            np.isfinite(residual)])
        observed.append(view.corners+view.edges>0)
    rows=[]
    valid=[]
    for omitted in OMITTED:
        fit=evidence.hypotheses[omitted]
        valid.append(fit.valid)
        if fit.valid:
            rows.append([*fit.pose.translation,*fit.pose.rotation.ravel(),True,fit.rms,np.log1p(fit.condition)])
        else:
            rows.append(np.zeros(len(HYPOTHESIS_FIELDS)))
    matrix=evidence.residuals[OMITTED]
    available=np.isfinite(matrix)&np.array(valid)[:,None]
    return dict(camera_tokens=np.asarray(camera_rows,np.float32),
                hypothesis_tokens=np.asarray(rows,np.float32),
                cross_view_residual_px=np.where(available,matrix,0).astype(np.float32),
                subset_membership_mask=membership, residual_valid_mask=available,
                camera_has_matches=np.array(observed), hypothesis_valid=np.array(valid),
                view_support=np.asarray(evidence.support)[OMITTED]&available)
