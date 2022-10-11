import torch 
import numpy as np
import SimpleITK as sitk


def metric_OAR(dose,roi):
    """
    dose is the dose image
    roi represents seven OAR masks array including Brainstem, SpinalCord, RightParotid, LeftParotid,
                Esophagus, Larynx, Mandible
    """

    dose_array = sitk.GetArrayFromImage(dose)
    spacing = dose.GetSpacing()
    voxel_size = np.prod(spacing)
    voxels_in_tenth_of_cc = np.maximum(1,np.round(100/voxel_size))
    if roi.any() != 0.0:
        roi_dose = dose_array[np.nonzero(roi)]
        D_mean = roi_dose.mean()  ## the mean dose value in roi
        D_01cc = np.percentile(roi_dose, (100-voxels_in_tenth_of_cc/len(roi_dose) * 100))  ## the maximum dose received by 0.1cc volume of the OAR 
    else:
        D_mean = 0; D_01cc = 0
    return D_mean, D_01cc


def metric_PTV(dose,roi):
    """
    dose is the dose image
    roi represents three PTV masks array including PTV56 mask, PTV63 mask and PTV70 mask
    
    You should load one case to calculate the metric.
    """

    if roi.any() != 0.0:
        ### dose array
        dose = sitk.GetArrayFromImage(dose)
        ### calculate CI metric
        V_t = len(np.nonzero(roi)[0])   # the volume of the target (i.e., PTV)
        V_pi = len(np.where(dose>=56.)[0])   # the volume of the prescription isodose
        Volume_pi = np.zeros((128,128,128)); Volume_pi[np.where(dose>=56.)] = 1  # the array of the prescription isodose
        V_inter = len(np.nonzero(Volume_pi * roi)[0])  # the intersection volume of the target volume and the prescription isodose volume   
        CI = (V_inter * V_inter) / (V_t * V_pi)  
        ### calculate HI metric
        D2 = np.percentile(dose[np.nonzero(roi)],98)  # the minimum dose received by 2% of the PTV volume
        D98 = np.percentile(dose[np.nonzero(roi)],2)  # the minimum dose received by 98% of the PTV volume
        D50 = np.percentile(dose[np.nonzero(roi)],50)  # the minimum dose received by 50% of the PTV volume
        HI = (D2 - D98) / D50
        ### calculate D99, D95, D1
        D99 = np.percentile(dose[np.nonzero(roi)],1)  # the minimum dose received by 99% of the PTV volume
        D95 = np.percentile(dose[np.nonzero(roi)],5)  # the minimum dose received by 95% of the PTV volume
        D1 = np.percentile(dose[np.nonzero(roi)],99)  # the minimum dose received by 1% of the PTV volume
    else: 
        CI=0; HI=0; D99 = 0; D95 = 0; D1 = 0
    return CI, HI, D99, D95, D1

    
#### Calculate clinical metircs for one case
def cal_metric(pd,gt,roi):
    """
    pd is the predicted dose image, and gt is the ground truth dose image
    roi represents ten roi masks including seven OAR masks and three PTV masks

    You should load one case for calculating these metrics.
    """

    HI_list = []; CI_list = []; D99_list = []; D95_list = []; D1_list = []; Dmean_list = []; D01cc_list = []
    for i in len(range(roi)):
        if i < 7:
            pd_Dmean, pd_D01cc = metric_OAR(pd,roi); gt_Dmean, gt_D01cc = metric_OAR(gt,roi)
            ## calculate the difference between the prediction and the ground truth for OAR dose coverage
            diff_Dmean = np.abs(pd_Dmean-gt_Dmean); diff_D01cc = np.abs(pd_D01cc-gt_D01cc)
            Dmean_list.append(diff_Dmean); D01cc_list.append(diff_D01cc)
        if i > 6:
            pd_CI, pd_HI, pd_D99, pd_D95, pd_D1 = metric_PTV(pd,roi)
            gt_CI, gt_HI, gt_D99, gt_D95, gt_D1 = metric_PTV(gt,roi)
            ## calculate the difference between the prediction and the ground truth for PTV dose coverage
            diff_CI = np.abs(pd_CI-gt_CI); diff_HI = np.abs(pd_HI-gt_HI); diff_D99 = np.abs(pd_D99-gt_D99)
            diff_D95 = np.abs(pd_D95-gt_D95); diff_D1 = np.abs(pd_D1-gt_D1)
            CI_list.append(diff_CI); HI_list.append(diff_HI); D99_list.append(diff_D99)
            D95_list.append(diff_D95); D1_list.append(diff_D1)
    
    return sum(HI_list)/len(HI_list),sum(CI_list)/len(CI_list),sum(D99_list)/len(D99_list),sum(D95_list)/len(D95_list),sum(D1_list)/len(D1_list),sum(Dmean_list)/len(Dmean_list),sum(D01cc_list)/len(D01cc_list)  
            


    
    
