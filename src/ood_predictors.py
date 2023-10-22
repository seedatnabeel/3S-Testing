from .ATC_code.ATC_helper import *
from .ATC_code.predict_acc_helper import *

def test_atc(model_name, trained_model_dict, X_test, X_val):
    """Computes ATC
    Garg et al. Leveraging unlabeled data to predict out-of-distribution performance.
    https://github.com/saurabhgarg1996/ATC_code"""

    val_probs = trained_model_dict[model_name].predict_proba(X_val.drop(columns=['y']))
    val_labels =  X_val['y'].values

    test_probs = trained_model_dict[model_name].predict_proba(X_test.drop(columns=['y']))

    ## score function, e.g., negative entropy or argmax confidence 
    val_scores = get_entropy(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)

    test_sc = get_entropy(test_probs)

    _, ATC_thres = find_ATC_threshold(val_scores, val_labels == val_preds)
    ATC_accuracy = get_ATC_acc(ATC_thres, test_sc)

    return ATC_accuracy

def test_atc_mc(model_name, trained_model_dict, X_test, X_val):
    """Computes ATC Max Confidence
    Garg et al. Leveraging unlabeled data to predict out-of-distribution performance.
    https://github.com/saurabhgarg1996/ATC_code"""

    val_probs = trained_model_dict[model_name].predict_proba(X_val.drop(columns=['y']))
    val_labels =  X_val['y'].values

    test_probs = trained_model_dict[model_name].predict_proba(X_test.drop(columns=['y']))

    ## score function, e.g., negative entropy or argmax confidence 
    val_scores = get_max_conf(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)

    test_sc = get_max_conf(test_probs)

    _, ATC_thres = find_ATC_threshold(val_scores, val_labels == val_preds)
    ATC_accuracy = get_ATC_acc(ATC_thres, test_sc)

    return ATC_accuracy


def get_im_estimate(probs_source, probs_target, corr_source): 
    """Gets importance estimates via Histogram"""

    source_binning = HistogramDensity()
    source_binning.fit(probs_source)

    target_binning = HistogramDensity()
    target_binning.fit(probs_target)

    numer = target_binning.density(probs_source) 
    den = source_binning.density(probs_source)


    den[den == 0] = 0.0001

    weights = numer/den

    weights = weights/ np.mean(weights)

    return np.mean(weights*corr_source)*100.0


def test_im_est(model_name, trained_model_dict, X_test, X_val):
    """Computes IM-EST
    Chen et al. Mandoline: Model evaluation under distribution shift.
    From: https://github.com/saurabhgarg1996/ATC_code"""


    val_probs = trained_model_dict[model_name].predict_proba(X_val.drop(columns=['y']))
    val_preds =  trained_model_dict[model_name].predict(X_val.drop(columns=['y']))
    val_labels =  X_val['y'].values
    labelsv1 = X_val['y'].values

    v1acc = np.mean(val_preds == val_labels)*100

    test_probs = trained_model_dict[model_name].predict_proba(X_test.drop(columns=['y']))

    calib_pred_idxv1 = np.argmax(val_probs, axis=-1)
    calib_pred_probsv1 = np.max(val_probs, axis=-1)

    calib_pred_probs_new = np.max(test_probs, axis=-1)

    calib_im_estimtate = get_im_estimate(calib_pred_probsv1, calib_pred_probs_new, (calib_pred_idxv1 == labelsv1)) 

    return calib_im_estimtate

def test_doc_feat(model_name, trained_model_dict, X_test, X_val):
    """Computes Doc-Feat
      Guillory et al. Pre495 dicting with confidence on unseen distributions. 
      From: https://github.com/saurabhgarg1996/ATC_code"""

    val_probs = trained_model_dict[model_name].predict_proba(X_val.drop(columns=['y']))
    val_preds =  trained_model_dict[model_name].predict(X_val.drop(columns=['y']))
    val_labels =  X_val['y'].values
    labelsv1 = X_val['y'].values

    v1acc = np.mean(val_preds == val_labels)*100

    test_probs = trained_model_dict[model_name].predict_proba(X_test.drop(columns=['y']))

    calib_pred_idxv1 = np.argmax(val_probs, axis=-1)
    calib_pred_probsv1 = np.max(val_probs, axis=-1)

    calib_pred_probs_new = np.max(test_probs, axis=-1)

    calib_doc_feat = v1acc + get_doc(calib_pred_probsv1, calib_pred_probs_new)*100.0

    return calib_doc_feat


class HistogramDensity: 
    def _histedges_equalN(self, x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
    
    def __init__(self, num_bins = 10, equal_mass=False):
        self.num_bins = num_bins 
        self.equal_mass = equal_mass
        
        
    def fit(self, vals): 
        
        if self.equal_mass:
            self.bins = self._histedges_equalN(vals, self.num_bins)
        else: 
            self.bins = np.linspace(0,1.0,self.num_bins+1)
    
        self.bins[0] = 0.0 
        self.bins[self.num_bins] = 1.0
        
        self.hist, bin_edges = np.histogram(vals, bins=self.bins, density=True)
    
    def density(self, x): 
        curr_bins = np.digitize(x, self.bins, right=True)
        
        curr_bins -= 1
        return self.hist[curr_bins]
    