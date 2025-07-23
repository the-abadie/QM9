# region Import Packages
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import multiprocessing   as mp
import scienceplots # noqa: F401
import datetime
import time
import os
import argparse
import sys

from sklearn.kernel_ridge    import KernelRidge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics         import mean_absolute_error, mean_squared_error
from sklearn.preprocessing   import StandardScaler
from skopt                   import BayesSearchCV
# endregion

#region Function Definitions
def stratified_selection(values, n_strata: int, n_total: int, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    values = np.asarray(values)
    n_samples = len(values)

    if n_total > n_samples:
        raise ValueError(f"n_total ({n_total}) cannot be greater than number of samples ({n_samples})")
    if n_strata > n_total:
        n_strata = n_total  # Limit number of strata if too many

    # Sort and split values into strata
    sorted_indices = np.argsort(values)
    strata = np.array_split(sorted_indices, n_strata)

    # Compute how many from each stratum
    per_stratum = n_total // n_strata
    remainder = n_total % n_strata

    selected_indices = []

    for i, stratum in enumerate(strata):
        n_select = per_stratum + (1 if i < remainder else 0)
        if n_select > len(stratum):
            raise ValueError(f"Stratum {i} too small to select {n_select} elements.")
        selected_indices.extend(rng.choice(stratum, size=n_select, replace=False))

    return np.array(selected_indices)

#endregion

def main():

    #region Read Command-Line Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--DESC", type=str,
                        help="Path to molecular descriptors (.npy).")

    parser.add_argument("-T", "--TARGET", type=str,
                        help="Path to targets (.dat).")

    parser.add_argument("-O", "--OUTLIER", type=int, default=1,
                        help="Remove data points that are over 10x the median value of all targets.")

    parser.add_argument("-o", "--OUT", type=str  , default="results",
                        help="Path to save results.")

    parser.add_argument("-N", type=int, default=16,
                        help="Number of cores for parellization.")

    parser.add_argument("-n", "--TRAINFRAC", type=float, default=0.15,
                        help="Fraction of total data to use for training.")

    parser.add_argument("-K", "--KERNEL", type=str, default="rbf",
                        help="Kernel to use for training. Options: 'rbf', 'laplacian'.")

    parser.add_argument("-m", "--METRIC", type=str, default="MAE",
                        help="Metric to optimize during training. Options: 'MAE', 'RMSE'.")

    parser.add_argument("--STRATA", type=int, default=10,
                        help="Degree of stratification for training. '0' or '1' to turn off.")

    parser.add_argument("-k", "--KFOLD", type=int, default=5,
                        help="Number of cross-validation folds to use during training. \
                             '999' for leave-one-out.")

    parser.add_argument("--NORMALIZE_DESC", type=int, default=1,
                        help="Normalize descriptors before training. '0' to turn off.")

    parser.add_argument("--NORMALIZE_TARGET", type=int, default=0,
                        help="Normalize targets before training. '0' to turn off.")

    parser.add_argument("--SIGMA_MIN", type=int, default=0,
                        help="Exponent for smallest sigma value (2^).")

    parser.add_argument("--SIGMA_MAX", type=int, default=16,
                        help="Exponent for largest sigma value (2^).")

    parser.add_argument("--LAMBDA_MIN", type=int, default=-12,
                        help="Exponent for smallest lambda value (10^).")

    parser.add_argument("--LAMBDA_MAX", type=int, default=0,
                        help="Exponent for largest lambda value (10^).")

    parser.add_argument("--ITER", type=int, default=50,
                        help="Numer of iterations to use in Bayesian CV Search.")

    parser.add_argument("--PLOT", type=int, default=1,
                        help="Plot results, saved in path specified by --OUT. '0' to turn off.")

    parser.add_argument("-v", "--VERBOSE", type=int, default=1,
                        help="Turn on verbose output.")

    parser.add_argument("-d", "--DEBUG"  , type=int, default=0,
                        help="Turn on debug mode. Limits total number of data to 1000 points.")

    args = parser.parse_args()

    DESC_PATH        :str   = args.DESC
    TARGET_PATH      :str   = args.TARGET
    OUTPUT_PATH      :str   = args.OUT
    OUTLIER          :int   = args.OUTLIER
    N_CORES          :int   = args.N
    TRAINING_FRACTION:float = args.TRAINFRAC
    KERNEL           :str   = args.KERNEL
    METRIC           :str   = args.METRIC
    N_STRATA         :int   = args.STRATA
    N_FOLDS          :int   = args.KFOLD
    NORMAL_DESC      :int   = args.NORMALIZE_DESC
    NORMAL_TARGET    :int   = args.NORMALIZE_TARGET
    SIGMA_MIN        :int   = args.SIGMA_MIN
    SIGMA_MAX        :int   = args.SIGMA_MAX
    LAMBDA_MIN       :int   = args.LAMBDA_MIN
    LAMBDA_MAX       :int   = args.LAMBDA_MAX
    N_ITER           :int   = args.ITER
    PLOT             :int   = args.PLOT
    VERBOSITY        :int   = args.VERBOSE
    DEBUG            :int   = args.DEBUG
    SEED             :int = int(datetime.datetime.now().timestamp())

    #endregion

    #region Import Data
    if VERBOSITY > 1:
        print("Region: Import Data")
        print("Region: Import Data", file=sys.stderr)

    root, extension = os.path.splitext(DESC_PATH)
    if extension != ".npy":
        print("Descriptor path is not .npy. Exiting.")
        exit()

    try:
        descriptors_load_start    = time.time()
        DESCRIPTORS = np.load(DESC_PATH)
        descriptors_load_duration = time.time() - descriptors_load_start

        if VERBOSITY > 1:
            print(f"{DESC_PATH} loaded in {int(descriptors_load_duration)} seconds.")

        if DEBUG > 0:
            print("Debug mode enabled. Limiting total descriptor size to 1000.")
            DESCRIPTORS = DESCRIPTORS[:1000]

    except Exception as e:
        print(f"Loading descriptor failed. Does {DESC_PATH} exist?")
        print(e)
        exit()

    root, extension = os.path.splitext(TARGET_PATH)
    if extension != ".npy":
        try:
            TARGETS = np.loadtxt(TARGET_PATH)

            if DEBUG > 0:
                print("Debug mode enabled. Limiting total target size to 1000.")
                TARGETS = TARGETS[:1000]

        except Exception as e:
            print(f"Loading target failed. Does {TARGET_PATH} exist?")
            print(e)
            exit()
    else:
        try:
            TARGETS = np.load(TARGET_PATH)

            if DEBUG > 0:
                print("Debug mode enabled. Limiting total target size to 1000.")
                TARGETS = TARGETS[:1000]

        except Exception as e:
            print(f"Loading target failed. Does {TARGET_PATH} exist?")
            print(e)
            exit()
    #endregion

    #region Pre-Processing
    if VERBOSITY > 1:
        print("Region: Pre-Processing")

    assert len(DESCRIPTORS) == len(TARGETS), \
        f"Length of descriptor ({len(DESCRIPTORS)}) does not match length of target ({len(TARGETS)})."
    assert KERNEL in ["rbf", "gaussian", "laplacian"], \
        f"Kernel must be 'rbf', 'gaussian', or 'laplacian'."
    assert METRIC in ["MAE", "RMSE"], \
        f"Metric must be 'MAE' or 'RMSE'."
    assert 0. < TRAINING_FRACTION < 1., f"Training fraction ({TRAINING_FRACTION}) must be in range [0, 1]."

    if OUTLIER > 0:
        if VERBOSITY > 1:
            print("Outlier removal activated.")

        TARGET_MEDIAN = np.median(np.abs(TARGETS))

        mask = np.abs(TARGETS) <= 10 * TARGET_MEDIAN

        N_tot :int = len(TARGETS)
        N_kept:int = np.sum(mask)

        DESCRIPTORS = DESCRIPTORS[mask]
        TARGETS     = TARGETS    [mask]

        if VERBOSITY > 1:
            print(f"{N_tot - N_kept} ({round(100*(N_tot - N_kept)/N_tot, 3)}%) " + \
                   "points removed for being greater than 10x the median target value.")

    N      :int = len(TARGETS)
    N_TRAIN:int = int(TRAINING_FRACTION * N)

    OUTPUT_PATH = f"{OUTPUT_PATH}/{N_TRAIN}_{SEED}"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    WORKING_DESCRIPTORS = DESCRIPTORS
    WORKING_TARGETS     = TARGETS

    scaler = StandardScaler()
    if NORMAL_DESC:
        WORKING_DESCRIPTORS = scaler.fit_transform(DESCRIPTORS)
    if NORMAL_TARGET:
        WORKING_TARGETS     = scaler.fit_transform(TARGETS.reshape(-1, 1)).ravel()


    if KERNEL in ["rbf", "gaussian"]:
        gamma_MIN = 1.0 / (2.0 * (2. ** SIGMA_MIN) ** 2.)
        gamma_MAX = 1.0 / (2.0 * (2. ** SIGMA_MAX) ** 2.)
    elif KERNEL == "laplacian":
        gamma_MIN = 1.0 / SIGMA_MIN
        gamma_MAX = 1.0 / SIGMA_MAX
    else:
        gamma_MAX = 1.0 / SIGMA_MAX
        gamma_MIN = 1.0 / SIGMA_MIN

    search_space = {
        "alpha": (10 ** LAMBDA_MIN, 10 ** LAMBDA_MAX, "log-uniform"),
        "gamma": (gamma_MAX, gamma_MIN, "log-uniform"),
    }


    idx = np.arange(N)

    if N_STRATA > 1:
        local_train_idx = stratified_selection(values = TARGETS, n_strata = N_STRATA, n_total = N_TRAIN)
        train_idx       = idx[local_train_idx]
    else:
        train_idx = np.random.choice(a = idx, size = N_TRAIN, replace = False)

    test_idx  = np.setdiff1d(idx, train_idx)

    assert len(train_idx)                 == N_TRAIN
    assert len(train_idx) + len(test_idx) == N, f"{len(train_idx) + len(test_idx)} != {N}"

    TRAINING_DESCRIPTORS = WORKING_DESCRIPTORS [train_idx]
    TRAINING_TARGETS     = WORKING_TARGETS     [train_idx]

    TESTING_DESCRIPTORS  = WORKING_DESCRIPTORS [test_idx]
    TESTING_TARGETS      = WORKING_TARGETS     [test_idx]

    if   KERNEL in ["rbf", "gaussian"]:
        kernel = KernelRidge(kernel="rbf")
    elif KERNEL == "laplacian":
        kernel = KernelRidge(kernel="laplacian")
    else:
        kernel = KernelRidge(kernel="rbf")

    if   METRIC == "MAE":
        score = "neg_mean_absolute_error"
    elif METRIC == "RMSE":
        score = "neg_root_mean_squared_error"
    else:
        score = "neg_mean_absolute_error"

    cross_validation = LeaveOneOut() if N_FOLDS == 999 else N_FOLDS
    try:
        KRR_MODEL = BayesSearchCV(
            estimator     = kernel,
            search_spaces = search_space,
            n_iter        = N_ITER,
            cv            = cross_validation,
            scoring       = score,
            random_state  = SEED,
            n_jobs        = N_CORES,
            refit         = True,
            verbose       = 0
        )
    except Exception as e:
        print("Something went wrong when initializing the cross-validation model. Please check all parameters to make sure they are correct.")
        print(e)
        exit()
    #endregion

    #region Train/Evaluate Model
    if VERBOSITY > 1:
        print("Region: Train/Evaluate Model")
        print("Region: Train/Evaluate Model", file=sys.stderr)

    model_start_time = time.time()

    KRR_MODEL.fit(TRAINING_DESCRIPTORS, TRAINING_TARGETS)

    model_duration = time.time() - model_start_time

    if NORMAL_TARGET:
        y_pred = KRR_MODEL.predict(TESTING_DESCRIPTORS)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    else:
        y_pred = KRR_MODEL.predict(TESTING_DESCRIPTORS)

    prediction_MAE :float =         mean_absolute_error(TARGETS[test_idx], y_pred)
    prediction_RMSE:float = np.sqrt(mean_squared_error (TARGETS[test_idx], y_pred))

    TESTING_TARGETS_MAE :float = np.mean(np.abs (TARGETS[test_idx])    , dtype=float)
    TESTING_TARGETS_RMSE:float = np.sqrt(np.mean(TARGETS[test_idx]**2.), dtype=float)

    relative_MAE  = 100. * prediction_MAE  / TESTING_TARGETS_MAE
    relative_RMSE = 100. * prediction_RMSE / TESTING_TARGETS_RMSE

    if KERNEL in ["rbf", "gaussian"]:
        best_sigma = 1. / np.sqrt(2. * KRR_MODEL.best_params_["gamma"])
    else:
        best_sigma = KRR_MODEL.best_params_["gamma"] ** -1.

    best_lambda = KRR_MODEL.best_params_["alpha"]

    #endregion

    #region Save Results
    if VERBOSITY > 1:
        print("Region: Save Results")
        print("Region: Save Results", file=sys.stderr)


    if VERBOSITY > 0:
        print("--------MODEL EVALUATION COMPLETE--------")
        print(f"Target: {TARGET_PATH}")
        print(f"Relative MAE: {round(relative_MAE, 2)}%")
        print(f"Model cross-validated and retrained in {round(model_duration, 2)} seconds ({round(model_duration/60., 2)} minutes).")
        print(f"{round(model_duration/N_TRAIN, 2)} seconds/N_TRAIN.")
        print(f"{round(model_duration/N_TRAIN/N_FOLDS, 2)} seconds/N_TRAIN/N_FOLDS.")
        print(f"{round(model_duration/N_TRAIN/N_FOLDS/N_CORES, 2)} seconds/N_TRAIN/N_FOLDS/N_CORES.")
        print("-----------------------------------------")

    try:
        np.save(file = f"{OUTPUT_PATH}/training_descriptors.npy",
                arr  = TRAINING_DESCRIPTORS)
        np.save(file = f"{OUTPUT_PATH}/training_targets.npy",
                arr  = TRAINING_TARGETS)
    except Exception as e:
        print("Saving training data failed. Check to see if you have enough space, or if your output path exists.")
        print(e)

    try:
        np.save(file = f"{OUTPUT_PATH}/testing_descriptors.npy",
                arr  = TESTING_DESCRIPTORS)
        np.save(file = f"{OUTPUT_PATH}/testing_targets.npy",
                arr  = TESTING_TARGETS)
    except Exception as e:
        print("Saving testing data failed. Check to see if you have enough space, or if your output path exists.")
        print(e)

    try:
        np.save(file = f"{OUTPUT_PATH}/model_weights.npy",
                arr  = KRR_MODEL.best_estimator_.dual_coef_)

        np.savetxt(fname = f"{OUTPUT_PATH}/lambda.txt",
                   X     = np.asarray(best_lambda))

        np.savetxt(fname = f"{OUTPUT_PATH}/sigma.txt",
                   X     = np.asarray(best_sigma))
    except Exception as e:
        print("Saving model parameters failed. Check to see if you have enough space, or if your output path exists.")
        print(e)

    try:
        with open(f"{OUTPUT_PATH}/summary.txt", "w") as f:
            f.write("--------MODEL SUMMARY--------\n")
            f.write(f"Training Fraction: {TRAINING_FRACTION} (N = {N_TRAIN})\n\n")

            f.write(f"MAE : {prediction_MAE :.3f} ({relative_MAE :.2f}%)\n")
            f.write(f"RMSE: {prediction_RMSE:.3f} ({relative_RMSE:.2f}%)\n\n")

            f.write(f"Optimal Sigma : {best_sigma :.3f}\n")
            f.write(f"Optimal Lambda: {best_lambda:.3f}\n\n")

            f.write(f"Normalized Desciptors? : {NORMAL_DESC}\n")
            f.write(f"Normalized Targets?    : {NORMAL_TARGET}\n\n")

            f.write(f"Metric    : {METRIC}\n")
            f.write(f"Kernel    : {KERNEL}\n")
            f.write(f"Strata    : {N_STRATA}\n")
            f.write(f"KFolds    : {N_FOLDS}\n")
            f.write(f"Iterations: {N_ITER}\n\n")

            f.write(f"Sigma Range : 2^{SIGMA_MIN} -> 2^{SIGMA_MAX}\n")
            f.write(f"Lambda Range: 10^{LAMBDA_MIN} -> 10^{LAMBDA_MAX}\n")
    except Exception as e:
        print("Saving model summary failed. Check to see if you have enough space, or if your output path exists.")
        print(e)

    #endregion

    #region Plot Results
    if PLOT == 0:
        exit()

    if VERBOSITY > 1:
        print("Region: Plot Results")
        print("Region: Plot Results", file=sys.stderr)


    plt.style.use(["nature", "science"])

    results_df = pd.DataFrame(KRR_MODEL.cv_results_)

    scores = results_df['mean_test_score']
    best_so_far = np.maximum.accumulate(scores)

    _, ax = plt.subplots()
    ax.plot(best_so_far, marker='o', label='Best so far')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Mean Test Score')
    ax.set_title('BayesSearchCV Convergence')
    ax.grid(True)

    plt.savefig(f"{OUTPUT_PATH}/convergence.png", bbox_inches="tight", dpi=150)

    fig, ax = plt.subplots()
    sc = ax.scatter(results_df['param_alpha'], results_df['param_gamma'],
                    c=results_df['mean_test_score'], cmap='jet', s=40)
    fig.colorbar(sc, ax=ax, label='Mean Test Score')

    ax.set_xlabel('alpha')
    ax.set_ylabel('gamma')
    ax.set_title('Search Space Performance')
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.savefig(f"{OUTPUT_PATH}/search.png", bbox_inches="tight", dpi=150)

    _, ax = plt.subplots()
    ax.scatter(TARGETS[test_idx], y_pred, s=1, color="goldenrod", zorder=10)
    ax.plot([0.8*min(TARGETS[test_idx]), 1.2*max(TARGETS[test_idx])],
            [0.8*min(TARGETS[test_idx]), 1.2*max(TARGETS[test_idx])],
            color="royalblue", linestyle="--")
    ax.set_xlabel("Groud Truth")
    ax.set_ylabel("Prediction")
    ax.set_title("Model Predictions on Test Set")

    plt.savefig(f"{OUTPUT_PATH}/yy.png", bbox_inches="tight", dpi=150)

    _, ax = plt.subplots()
    ax.hist(x=TARGETS[test_idx] - y_pred, bins=201, density=True, color="goldenrod")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Test Set Residual Distribution")

    plt.savefig(f"{OUTPUT_PATH}/residual.png", bbox_inches="tight", dpi=150)

    #endregion

if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()