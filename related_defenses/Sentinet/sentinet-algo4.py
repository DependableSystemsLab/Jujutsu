import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import fmin_cobyla
import pandas
import matplotlib.pyplot as plt

def func(x, a, b, c):
    return a * np.power(x, 2) + b * x + c

def curve_dists(confidences, fooled_percentages, number_of_bins=25):
    assert len(confidences) == len(fooled_percentages)

    confidences = np.asarray(confidences)
    fooled_percentages = np.asarray(fooled_percentages)

    # number of bins used to digitize the confidences
    number_of_bins = 25
    min_conf = confidences.min()
    max_conf = confidences.max()

    # make number_of_bins bins from min to max confidence
    bins = np.arange(min_conf, max_conf, (max_conf - min_conf)/number_of_bins)

    # save which bin each sample belongs to in a new column
    bin_identifier = np.digitize(confidences, bins)

    xs = []
    ys = []
    for i, _ in enumerate(bins):
        # filter only the samples that belong to this confidence bin
        # (offset by one because of np.digitise above)
        bin_mask = bin_identifier == i+1
        bin_conf, bin_fp = confidences[bin_mask], fooled_percentages[bin_mask]
        if bin_conf.shape[0] > 0:
            # get largest FoolPercentage
            i_of_max_fp = np.argmax(bin_fp)
            xs.append(bin_conf[i_of_max_fp])
            ys.append(bin_fp[i_of_max_fp])

    # fit curve to per-bin points
    curve_params, covariance = curve_fit(func, np.array(xs), np.array(ys))

    avg_d = 0
    curve_func = lambda x: func(x, *curve_params)

    # compute threshold 
    for x, y in zip(xs, ys):
        # NB. there is a mismatch between the code for Algorithm 4 in 
        # the SentiNet paper and the text. In the following if statement
        # we use the interpretation found in the text.
        if curve_func(x) < y:  # for each point lying above the curve
            # create function that returns 2D distance from (x,y) and (a, f(a))
            distance_func = lambda a: np.sqrt((x-a)**2+(y-curve_func(a))**2)
            # find minimum value for a
            result = minimize(distance_func, np.array(x), method='COBYLA')
            distance = distance_func(result['x'])
            # accumulate distances of points above curve
            avg_d += distance

    d = avg_d / confidences.shape[0]
    return curve_func, d

def is_adversarial(curve, threshold, x, y):
    #x, y = point
    y_prime = curve(x)
    is_adv = False
    if y > y_prime:
        # we are above the curve, now calculate distance to curve
        distance_func = lambda a: np.sqrt((x-a)**2+(y-curve(a))**2)
        result = minimize(distance_func, x, method='COBYLA')
        distance = distance_func(result['x'])
        is_adv = distance > threshold
    return is_adv

dataBenign = pandas.read_csv('/home/pritamdash/Documents/adv-ml/short-lived-adversarial-perturbations/code/defences/sentinet-results/vggnet_005_benign_results.csv')
dataAdv = pandas.read_csv('/home/pritamdash/Documents/adv-ml/short-lived-adversarial-perturbations/code/defences/sentinet-results/vggnet_005_adversarial_results.csv')

benign_fooled = dataBenign['FoolPercentage']
benign_conf = dataBenign['Confidence']
adv_fooled = dataAdv['FoolPercentage']
adv_conf = dataAdv['Confidence']

#popt, pcov = curve_fit(func, benign_conf, benign_fooled)
curve, threshold = curve_dists(benign_conf, benign_fooled, 25) #popt instead

over_threshold = 0
for i in range(len(benign_conf)):
    result = is_adversarial(curve, threshold, benign_conf[i], benign_fooled[i])
    if result:
        over_threshold += 1

print(over_threshold)
print("False Positive: %f (detected: %d, total: %d)" % (over_threshold/len(benign_conf)*100, over_threshold, len(benign_conf)))

for i in range(len(adv_conf)):
    result = is_adversarial(curve, threshold, adv_conf[i], adv_fooled[i])
    if result:
        over_threshold += 1

print("True Positive: %f (detected: %d, total: %d)" % (over_threshold/len(adv_conf)*100, over_threshold, len(adv_conf)))


