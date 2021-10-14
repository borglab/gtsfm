
import os
import shutil
import time
from collections import defaultdict

import argoverse.utils.subprocess_utils as subprocess_utils
import numpy as np

import gtsfm.evaluation.merge_reports as merge_reports

"""

success_prob = 0.9999
max_iters = 10000
    
Average runtime: 79.014
Runtimes:  [126.4421980381012, 64.90089106559753, 108.59735107421875, 65.34914112091064, 74.16682577133179, 62.01019597053528, 85.91280722618103, 65.06695079803467, 54.32871985435486, 60.00282311439514, 120.76739716529846, 90.54296374320984, 65.81232690811157, 122.58955693244934, 65.3543119430542, 126.50497603416443, 74.4913318157196, 118.18653702735901, 77.69931387901306, 66.2814450263977, 63.18450593948364, 60.366381883621216, 67.66954708099365, 57.415183305740356, 53.353302001953125, 60.630722999572754, 69.80035901069641, 58.37591314315796, 120.35256099700928, 64.26366519927979]
Metric: mean_translation_error_distance. Mean: 3.032, Median: 3.494
Metric: mean_rotation_error_angle_deg. Mean: 0.702, Median: 0.707
Metric: mean_translation_angle_error_deg. Mean: 16.664, Median: 7.951
Metric: mean_reprojection_errors_unfiltered_px. Mean: 2333.344, Median: 9.024
Metric: mean_translation_error_distance [3.38321, 3.50426, 3.50426, 3.49166, 3.49166, 3.54916, 3.54409, 3.49433, 3.42772, 3.51498, 0.0199789, 3.18999, 3.44645, 3.55905, 3.49411, 3.53061, 3.54942, 0.0192056, 3.49589, 3.28352, 3.38683, 3.47507, 3.55513, 3.49411, 3.53066, 3.49548, 0.0483131, 0.449791, 3.47798, 3.53959]
Metric: mean_rotation_error_angle_deg [1.26591, 0.707008, 0.707008, 0.575737, 0.575737, 0.720962, 0.410552, 0.908298, 0.69887, 0.646488, 0.183609, 0.754262, 0.955709, 0.200397, 0.664287, 0.850176, 0.733635, 0.181943, 0.642684, 1.16818, 0.851519, 1.21224, 0.736699, 0.694639, 0.652749, 0.722105, 0.455655, 0.603998, 0.844745, 0.734871]
Metric: mean_translation_angle_error_deg [8.87768, 12.3965, 12.3965, 8.3684, 8.3684, 9.87543, 8.88834, 8.48412, 150.481, 8.0844, 0.390605, 9.75255, 6.31028, 6.11631, 6.12333, 6.16327, 8.41495, 0.388251, 8.7923, 7.26486, 5.33496, 7.46349, 9.61634, 7.81851, 148.207, 6.66586, 0.766895, 6.53662, 6.6315, 4.93988]
Metric: mean_reprojection_errors_unfiltered_px [7.65669, 383.561, 383.561, 9.02397, 9.02397, 7.8869, 4.06365, 22.4572, 66088.0, 1710.79, 3.44202, 9.36185, 5.51027, 3.40528, 37.1125, 6.19801, 7.79354, 3.43543, 8.45728, 8.04267, 9.18978, 121.542, 7.40745, 277.41, 104.472, 34.7856, 4.18322, 707.187, 5.10629, 10.2561]


success_prob = 0.99999
max_iters = 100000

Average runtime: 149.428
Runtimes:  [79.63612914085388, 74.73657608032227, 105.35205698013306, 83.31858110427856, 89.28291821479797, 93.76439094543457, 128.84246182441711, 232.42686486244202, 1129.5907678604126, 103.86062002182007, 97.09608387947083, 139.116849899292, 146.23423290252686, 144.37373518943787, 163.6239800453186, 77.29669499397278, 72.20337009429932, 71.21292400360107, 129.57792115211487, 192.76016688346863, 160.73425483703613, 82.77042412757874, 75.82557225227356, 83.38984179496765, 113.63561797142029, 86.55227303504944, 166.0594539642334, 87.13931488990784, 95.630117893219, 176.7869749069214]
Metric: mean_translation_error_distance. Mean: 0.332, Median: 0.198
Metric: mean_rotation_error_angle_deg. Mean: 2.206, Median: 1.465
Metric: mean_translation_angle_error_deg. Mean: 3.532, Median: 3.399
Metric: mean_reprojection_errors_unfiltered_px. Mean: 111.218, Median: 3.645
Metric: mean_translation_error_distance [0.139791, 0.363989, 0.340857, 0.198445, 0.398928, 0.297091, 0.525222, 0.472482, 0.125315, 3.41148, 0.208509, 0.0871472, 0.106294, 0.104136, 0.424293, 0.202987, 0.189257, 0.197742, 0.0316613, 0.0431978, 0.0361829, 0.11482, 0.399623, 0.359865, 0.145785, 0.13131, 0.465577, 0.324492, 0.0991654, 0.0292249]
Metric: mean_rotation_error_angle_deg [1.90907, 2.89907, 4.29392, 0.369483, 2.31437, 2.53142, 4.22995, 7.56102, 0.902314, 1.25943, 0.375184, 0.278331, 2.07446, 0.268746, 8.84864, 3.73649, 0.439942, 1.20848, 0.269984, 0.255625, 0.360815, 1.67129, 3.6083, 0.618663, 2.90147, 0.376124, 8.30672, 1.67873, 0.345101, 0.300094]
Metric: mean_translation_angle_error_deg [2.51625, 5.38656, 5.54895, 3.32152, 6.51699, 4.52291, 7.12734, 5.65138, 2.42819, 4.23626, 3.46566, 1.64801, 1.66148, 1.90276, 3.54428, 3.95044, 3.33244, 3.58413, 0.688439, 0.794781, 0.82543, 2.1627, 7.19351, 6.3553, 2.39113, 2.36763, 5.52288, 5.09242, 1.58341, 0.634111]
Metric: mean_reprojection_errors_unfiltered_px [3.59048, 25.737, 3.52525, 3.61253, 3.58809, 311.726, 5.58958, 3.39077, 4.04627, 5.69397, 3.60537, 3.42579, 3.37377, 3.41816, 3.37243, 4.0954, 678.81, 20.6111, 3.37409, 3.33683, 3.4979, 7.48143, 2182.89, 12.7982, 3.67833, 4.45238, 3.5195, 6.23886, 10.6972, 3.37712]


30

success_prob = 0.999999
max_iters = 1000000

Average runtime: 113.775
Runtimes:  [118.88381314277649, 119.21082305908203, 72.29905605316162, 126.28331303596497, 121.96631622314453, 120.8042631149292, 119.97605514526367, 119.15713000297546, 122.08276581764221, 63.339455127716064, 125.66869902610779, 118.89696288108826, 119.40161395072937, 119.11009907722473, 120.2872953414917, 65.68125677108765, 123.74551701545715, 74.12073111534119, 135.89754700660706, 132.7932789325714, 131.00863003730774, 129.69632625579834, 126.15195536613464, 118.85760712623596, 119.93599104881287, 125.92298007011414, 131.618497133255, 83.91199016571045, 138.51307702064514, 68.04125618934631]
Metric: mean_translation_error_distance. Mean: 0.490, Median: 0.022
Metric: mean_rotation_error_angle_deg. Mean: 0.309, Median: 0.224
Metric: mean_translation_angle_error_deg. Mean: 1.031, Median: 0.413
Metric: mean_reprojection_errors_unfiltered_px. Mean: 243.972, Median: 3.453
Metric: mean_translation_error_distance [0.021812, 0.0218105, 0.0253237, 0.021812, 0.0218118, 0.021812, 0.021812, 0.0219133, 0.021812, 3.46315, 0.0218104, 0.0218105, 0.021812, 0.0218104, 0.0218104, 3.46893, 0.021812, 0.0221799, 0.021812, 0.0217436, 0.021812, 0.0217878, 0.021812, 0.021812, 0.0218104, 0.0218105, 0.021812, 0.0260803, 3.5997, 3.58471]
Metric: mean_rotation_error_angle_deg [0.224041, 0.224072, 0.297896, 0.22404, 0.224062, 0.22404, 0.224041, 0.223565, 0.224041, 0.827717, 0.224072, 0.224072, 0.22404, 0.224072, 0.224072, 1.08067, 0.224041, 0.225066, 0.224041, 0.224174, 0.224041, 0.223522, 0.224041, 0.224041, 0.224072, 0.224072, 0.224041, 0.22105, 0.580611, 0.896345]
Metric: mean_translation_angle_error_deg [0.412655, 0.412639, 0.481136, 0.412654, 0.412656, 0.412654, 0.412655, 0.415557, 0.412655, 2.9594, 0.41264, 0.412639, 0.412654, 0.41264, 0.41264, 6.09405, 0.412655, 0.411967, 0.412655, 0.410794, 0.412655, 0.412343, 0.412655, 0.412655, 0.41264, 0.412639, 0.412655, 0.540638, 5.6618, 5.2883]
Metric: mean_reprojection_errors_unfiltered_px [3.45282, 3.45289, 4.13095, 3.4531, 3.45291, 3.4531, 3.45282, 3.45317, 3.45282, 1099.71, 3.45261, 3.45289, 3.4531, 3.45261, 3.45261, 8.34418, 3.45282, 3.85575, 3.45282, 3.45755, 3.45282, 3.45268, 3.45282, 3.45282, 3.45261, 3.45289, 3.45282, 3.70297, 4.57497, 6115.41]
"""


def main():
    """

    0.9999 (prob)
    10K (iters)
    Average runtime: 84.072
    Runtimes:  [97.02513098716736, 71.76464200019836, 55.53377103805542, 60.09945487976074, 59.923949003219604, 128.07391810417175, 85.39936876296997, 93.3581919670105, 72.99456429481506, 116.54803609848022]
    Metric: mean_translation_error_distance. Mean: 3.470, Median: 3.486
    Metric: mean_rotation_error_angle_deg. Mean: 0.705, Median: 0.714
    Metric: mean_translation_angle_error_deg. Mean: 6.421, Median: 6.528
    Metric: mean_reprojection_errors_unfiltered_px. Mean: 92.368, Median: 5.685
    Metric: mean_translation_error_distance [3.55275, 3.48585, 3.48585, 3.40899, 3.4886, 3.49712, 3.32427, 3.47722, 3.48196, 3.49351]
    Metric: mean_rotation_error_angle_deg [0.736127, 0.735617, 0.735617, 0.719665, 0.64719, 0.701185, 0.689078, 0.637351, 0.707884, 0.73748]
    Metric: mean_translation_angle_error_deg [10.764, 7.16767, 7.16767, 2.77766, 4.85013, 3.70525, 10.7828, 6.4989, 3.94321, 6.55756]
    Metric: mean_reprojection_errors_unfiltered_px [5.50953, 5.0591, 5.0591, 145.243, 732.368, 5.86004, 8.05693, 6.20147, 5.39281, 4.93041]


    0.99999 (prob)
    100K (iters)
    Average runtime: 92.333
    Runtimes:  [134.7291238307953, 70.74555897712708, 63.27304029464722, 102.88810706138611, 115.43924188613892, 132.74360585212708, 68.19479727745056, 60.2667920589447, 83.6165988445282, 91.433758020401]
    Metric: mean_translation_error_distance. Mean: 0.696, Median: 0.352
    Metric: mean_rotation_error_angle_deg. Mean: 4.421, Median: 3.925
    Metric: mean_translation_angle_error_deg. Mean: 4.833, Median: 4.296
    Metric: mean_reprojection_errors_unfiltered_px. Mean: 72.155, Median: 3.532
    Metric: mean_translation_error_distance [0.275294, 3.40802, 1.02932, 0.032367, 0.558639, 0.114505, 0.317603, 0.781708, 0.386115, 0.0521986]
    Metric: mean_rotation_error_angle_deg [1.57278, 1.34802, 10.7358, 0.290873, 7.33056, 2.44572, 7.17998, 7.63943, 5.40511, 0.25994]
    Metric: mean_translation_angle_error_deg [4.70763, 3.18485, 10.3532, 0.694227, 7.04745, 1.62406, 3.88353, 10.2243, 5.56186, 1.04853]
    Metric: mean_reprojection_errors_unfiltered_px [3.57779, 5.0976, 9.1294, 3.39232, 3.48697, 3.38382, 3.98847, 682.689, 3.42193, 3.38401]

    0.999999 (prob)
    1M (iters)
    Average runtime: 114.521
    Runtimes:  [124.20440888404846, 120.91057205200195, 124.40085196495056, 121.95313692092896, 55.30132079124451, 121.97788119316101, 164.87192511558533, 121.98621797561646, 62.44512414932251, 127.15669679641724]
    Metric: mean_translation_error_distance. Mean: 1.079, Median: 0.022
    Metric: mean_rotation_error_angle_deg. Mean: 0.398, Median: 0.224
    Metric: mean_translation_angle_error_deg. Mean: 18.156, Median: 0.413
    Metric: mean_reprojection_errors_unfiltered_px. Mean: 584.586, Median: 3.454
    Metric: mean_translation_error_distance [0.0218104, 0.021812, 0.0216075, 0.0218118, 3.55313, 0.021812, 0.0223788, 0.021812, 3.50596, 3.57464]
    Metric: mean_rotation_error_angle_deg [0.224072, 0.224041, 0.221597, 0.224062, 1.06328, 0.224041, 0.225331, 0.224041, 0.77459, 0.578106]
    Metric: mean_translation_angle_error_deg [0.41264, 0.412655, 0.411575, 0.412656, 169.939, 0.412655, 0.417293, 0.412655, 4.34079, 4.38871]
    Metric: mean_reprojection_errors_unfiltered_px [3.45261, 3.45282, 3.45466, 3.45291, 5810.37, 3.45282, 4.26415, 3.45282, 5.73184, 4.77236]

    10M
    0.999 9999
    Average runtime: 95.406
    Runtimes:  [71.07198905944824, 129.1638491153717, 56.402936935424805, 129.27205109596252, 59.05067300796509, 58.44094276428223, 130.07476687431335, 129.08208799362183, 126.45771384239197, 65.04058718681335]
    Metric: mean_translation_error_distance. Mean: 1.442, Median: 0.048
    Metric: mean_rotation_error_angle_deg. Mean: 0.431, Median: 0.454
    Metric: mean_translation_angle_error_deg. Mean: 2.143, Median: 0.911
    Metric: mean_reprojection_errors_unfiltered_px. Mean: 518.000, Median: 4.764
    Metric: mean_translation_error_distance [3.53079, 3.57249, 3.53404, 3.60156, 0.0483568, 0.0483728, 0.0211007, 0.0211007, 0.0210992, 0.0225823]
    Metric: mean_rotation_error_angle_deg [0.484868, 0.613415, 0.825138, 0.646736, 0.45437, 0.454408, 0.18689, 0.186889, 0.186853, 0.274219]
    Metric: mean_translation_angle_error_deg [3.35893, 4.58575, 3.99438, 5.95106, 0.911035, 0.911639, 0.413979, 0.413978, 0.413982, 0.476136]
    Metric: mean_reprojection_errors_unfiltered_px [4.58113, 3.90633, 1115.38, 4.94722, 997.03, 3037.65, 3.47782, 3.47782, 3.47747, 6.06836]
    """
    runtimes = []

    success_prob = 0.9999
    max_iters = 10000

    # success_prob = 0.99999
    # max_iters = 100000

    # success_prob = 0.999999
    # max_iters = 1000000

    # success_prob = 0.9999999
    # max_iters = 10000000

    num_trials = 1 # 30

    aggregated_ba_metrics = defaultdict(list)

    for i in range(num_trials):

        cmd = "python gtsfm/runner/run_scene_optimizer_colmaploader.py"
        cmd += " --images_dir /Users/johnlambert/Downloads/skydio_crane_mast_32imgs_w_colmap_GT/images"
        cmd += " --colmap_files_dirpath /Users/johnlambert/Downloads/skydio_crane_mast_32imgs_w_colmap_GT/colmap_crane_mast_32imgs"
        cmd += " --num_workers 4"
        cmd += " --config_name deep_front_end.yaml"
        cmd += " --max_frame_lookahead 32"
        cmd += " --share_intrinsics"
        cmd += f" --success_prob {success_prob}"
        cmd += f" --max_iters {max_iters}"

        start = time.time()
        subprocess_utils.run_command(cmd)
        end = time.time()
        duration = end - start
        runtimes.append(duration)

        report_fpath = "/Users/johnlambert/Downloads/gtsfm/result_metrics/gtsfm_metrics_report.html"
        tables_dict = merge_reports.extract_tables_from_report(report_fpath)

        os.makedirs(f"{success_prob}__{max_iters}", exist_ok=True)
        shutil.copyfile("/Users/johnlambert/Downloads/gtsfm/plots/results/poses_bev.png", f"{success_prob}__{max_iters}/{i}_poses_bev.png")

        ba_metrics = tables_dict['Bundle Adjustment Metrics']
        metric_names = [
            'mean_translation_error_distance',
            'mean_rotation_error_angle_deg',
            'mean_translation_angle_error_deg',
            'mean_reprojection_errors_unfiltered_px'
        ]
        for metric_name in metric_names:
            aggregated_ba_metrics[metric_name] += [ float(ba_metrics[metric_name]) ]
        print(i, aggregated_ba_metrics)

    print(f"Average runtime: {np.mean(runtimes):.3f}")
    print("Runtimes: ", runtimes)

    for metric_name, metrics in aggregated_ba_metrics.items():
        print(f"Metric: {metric_name}. Mean: {np.mean(metrics):.3f}, Median: {np.median(metrics):.3f}")

    for metric_name, metrics in aggregated_ba_metrics.items():
        print(f"Metric: {metric_name}", metrics)


if __name__ == "__main__":
    main()
