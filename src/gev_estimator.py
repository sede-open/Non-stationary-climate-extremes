import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import genextreme
from scipy.linalg import sqrtm
import openturns as ot

from multiprocessing import Lock, Process, Queue, current_process
import time
import queue  # imported for using queue.Empty exception


class extreme_value_mcmc:
    """
    This is a class which performs MCMC (Markov Chain Monte Carlo) to fit a non-stationary generalised extreme value distribution.
    """

    def __init__(self, data, non_stationary=False, verbose=True):
        self.non_stationary = non_stationary
        self.verbose = verbose  # To not print initial diagnostic information

        if (len(data) == 6) or (len(data) == 3):
            self.data = self.generate_sample_data(data)
            self.initial_params = self.find_starting_parameters(self.data)
        else:
            # If there is a nan value, the simply drop that value
            self.data = data[~np.isnan(data)]
            self.initial_params = self.find_starting_parameters(self.data)

    ### Core Functions ###
    def log_likelihood(self, params, time_steps):
        """
        This function calculates the log_likelihood of a generalised extreme value distribution

        Input (list, list): params which is a list of estimates for the three gev parameters. time_steps is an array of points from 0,1 which is the length of the data.

        Output (float): the log_likelihood of a generalised extreme value distribution with the passed in parameters.
        """
        if self.non_stationary:
            loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
            loc = np.array(loc_0 + time_steps * loc_1)
            scale = np.array(scale_0 + time_steps * scale_1)
            shape = np.array(shape_0 + time_steps * shape_1)
        else:
            loc, scale, shape = params

        # Manual
        # n_shape = -1 * shape

        # t_0 = 1 + n_shape * (self.data - loc) / scale

        # t = t_0 ** (-1 / n_shape)

        # ll = np.sum(-np.log(scale) + (n_shape + 1) * np.log(t) - t)
        # return ll
        return np.sum(genextreme.logpdf(self.data, shape, loc=loc, scale=scale))

    def log_prior(self, params, time_steps):
        """
        This function provides reasonable constraints on estimated parameters. We dont allow the scale to fall below zero or the shape to be greater than 1 or less than -1.

        Input (list, list): params which is a list of estimates for the three gev parameters. time_steps is an array of points from 0,1 which is the length of the data.

        Output (float): Will return either 0 or -inf depending on the input parameters.
        """
        if self.non_stationary:
            loc_0, loc_1, scale_0, scale_1, shape_0, shape_1 = params
            loc = loc_0 + time_steps * loc_1
            scale = scale_0 + time_steps * scale_1
            shape = shape_0 + time_steps * shape_1
        else:
            loc, scale, shape = params

        if (min(scale) < 0) or (min(shape) < -0.2) or (max(shape) >= 1):
            return -np.inf
        return 0

    def propose(
        self, params, iteration, beta, total_accepted, samples, burn_in, time_steps
    ):
        # Make sure the total_accepted array has float values
        total_accepted = np.array(total_accepted.iloc[max(1, iteration-999):, :], dtype=float)

        if iteration <= burn_in:
            new_parameters = [param + np.random.normal(0, 1) * 0.1 for param in params]

        else:
            SH = sqrtm(np.cov(total_accepted, rowvar=False))
            SH = np.real(np.array(SH))
            z1 = np.random.normal(0, 1, size=len(params))
            z2 = np.random.normal(0, 1, size=len(params))
            y1 = (2.38 / np.sqrt(len(params))) * np.matmul(SH, z1)
            y2 = (0.1 / np.sqrt(len(params))) * z2
            new_parameters = params + (1 - beta) * y1 + beta * y2

        return new_parameters

    def acceptance_prob(self, old_params, new_params, time_steps):
        log_prior_old = self.log_prior(old_params, time_steps)
        log_prior_new = self.log_prior(new_params, time_steps)
        log_likelihood_old = self.log_likelihood(old_params, time_steps)
        log_likelihood_new = self.log_likelihood(new_params, time_steps)

        self.nll = pd.concat(
            [self.nll, pd.DataFrame({"negative_log_likelihood": [log_likelihood_old]})],
            ignore_index=True,
        )

        log_ratio = (log_likelihood_new + log_prior_new) - (
            log_likelihood_old + log_prior_old
        )

        return np.exp(log_ratio)

    def metropolis_hastings(
        self, n_samples, n2plt, burn_in=1000, thinning=10, beta=0.5, NGTSTR=0.1
    ):
        if self.non_stationary:
            samples = pd.DataFrame(
                columns=[
                    "location_0",
                    "location_1",
                    "scale_0",
                    "scale_1",
                    "shape_0",
                    "shape_1",
                ]
            )
            total_accepted = pd.DataFrame(
                columns=[
                    "location_0",
                    "location_1",
                    "scale_0",
                    "scale_1",
                    "shape_0",
                    "shape_1",
                ]
            )
        else:
            samples = pd.DataFrame(columns=["location", "scale", "shape"])
            total_accepted = pd.DataFrame(columns=["location", "scale", "shape"])

        ar = pd.DataFrame(columns=["acceptance_rate"])
        self.nll = pd.DataFrame(columns=["negative_log_likelihood"])
        current_params = self.initial_params
        time_steps = np.linspace(0, 1, len(self.data))

        for i in range(n_samples):
            if i <= burn_in:
                # new_params = current_params
                for j in range(0, 6):
                    new_params = current_params.copy()

                    new_params[j] = new_params[j] + np.random.normal(0, 1) * NGTSTR

                    acceptance_probability = self.acceptance_prob(
                        current_params, new_params, time_steps
                    )

                    if np.random.uniform(0, 1) < acceptance_probability:

                        current_params = new_params.copy()
                
                total_accepted = pd.concat(
                    [
                        total_accepted,
                        pd.DataFrame(
                            [current_params], columns=total_accepted.columns
                        ),
                    ],
                    ignore_index=True,
                )

            else:
                new_params = self.propose(
                    current_params,
                    i,
                    beta,
                    total_accepted,
                    samples,
                    burn_in,
                    time_steps,
                )
                acceptance_probability = self.acceptance_prob(
                    current_params, new_params, time_steps
                )

                if np.random.uniform(0, 1) < acceptance_probability:

                    current_params = new_params.copy()

                total_accepted = pd.concat(
                    [
                        total_accepted,
                        pd.DataFrame(
                            [current_params], columns=total_accepted.columns
                        ),
                    ],
                    ignore_index=True,
                )


            if i >= n2plt and i % thinning == 0:
                samples = pd.concat(
                    [samples, pd.DataFrame([current_params], columns=samples.columns)],
                    ignore_index=True,
                )

            acceptance_rate = len(total_accepted) / (i + 1)

            ar = pd.concat(
                [ar, pd.DataFrame([acceptance_rate], columns=["acceptance_rate"])],
                ignore_index=True,
            )

        return samples, total_accepted, ar
    
    def find_starting_parameters(self, Y_Dat):
    
        def fit_gev_params(block_data):

            try:
                sample = ot.Sample(np.array(block_data).reshape(-1, 1))
                params = ot.GeneralizedExtremeValueFactory().buildAsGeneralizedExtremeValue(sample)
                loc, scale, shape = params.getParameter()
                shape = -1*shape
                
                return loc, scale, shape
            except (Exception, ValueError) as e:
                print(f"OpenTurns Failed: {e}. Trying Scipy")
            
            try:
                # Dont flip sign of shape.
                shape, loc, scale = genextreme.fit(np.array(block_data))
                
                return loc, scale, shape
            except (Exception, ValueError) as e:
                raise ValueError(f"Scipy & Openturns Failed: {e}. Terminate")
        
        def get_regression_coefficients(tRgr):
            Y_XSMStart = np.zeros(6)
            for j in range(3):
                coef = np.polyfit(tRgr[:, 0], tRgr[:, j + 1], 1)
                Y_XSMStart[2 * j : 2 * (j + 1)] = coef
            return Y_XSMStart
        
        def plot_regression(tRgr, Y_XSMStart):
            plt.figure(figsize=(12, 6))
            var = ["Shape", "Location", "Scale"]
            for j in range(3):
                plt.subplot(2, 3, 4 + j)
                plt.plot(tRgr[:, 0], tRgr[:, j + 1], "b.-", label="Data")
                plt.plot(tRgr[:, 0], np.polyval(Y_XSMStart[2 * j : 2 * (j + 1)], tRgr[:, 0]), "k.-", label="Regression Line")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title(var[j])
                plt.grid(True)
            plt.tight_layout()
            plt.show()

        def validate_initial_params(initial_params, Y_Tim):
            nll = -self.log_likelihood(params=initial_params, time_steps=Y_Tim) + self.log_prior(params=initial_params, time_steps=Y_Tim)
            
            if np.isfinite(nll):
                if self.verbose:
                    print("Initial parameters result in a valid negative log likelihood:", nll)
                return True
            else:
                if self.verbose:
                    print("Initial parameters result in an invalid negative log likelihood:", nll)
                return False
        
        Y_nT = len(Y_Dat)
        Y_Tim = np.linspace(0, 1, Y_nT)

        if self.non_stationary:
            try:
                # Attempt non-stationary fit
                Y_nB = 10
                Y_Blc = np.linspace(0, Y_nT, Y_nB + 1).astype(int)
                tRgr = np.zeros((Y_nB, 4))

                for iB in range(Y_nB):
                    block_data = Y_Dat[Y_Blc[iB] : Y_Blc[iB + 1]]
                    loc, scale, shape = fit_gev_params(block_data)
                    tTim = np.mean(Y_Tim[Y_Blc[iB] : Y_Blc[iB + 1]])
                    tRgr[iB] = [tTim, shape, loc, scale]

                tRgr = tRgr[tRgr[:, 0].argsort()]
                Y_XSMStart = get_regression_coefficients(tRgr)

                if self.verbose:
                    plot_regression(tRgr, Y_XSMStart)

                initial_params = [
                    Y_XSMStart[3], Y_XSMStart[2],
                    Y_XSMStart[5], Y_XSMStart[4],
                    Y_XSMStart[1], Y_XSMStart[0]
                ]
                
                if validate_initial_params(initial_params, Y_Tim):
                    return initial_params

            except (Exception, ValueError) as e:
                print("non-stationary gev fit failed. Try stationary Fit.")

            # try stationary solution
            try:
                
                loc, scale, shape = fit_gev_params(Y_Dat)
                initial_params = [loc, 0, scale, 0, shape, 0]

                if validate_initial_params(initial_params, Y_Tim):
                    return initial_params

            except (Exception, ValueError):
                print("Failed to find valid initial parameters.")
            
            # Try stock solution
            try:
                if shape < -0.2:
                    initial_params = [loc, 0, scale, 0, -0.1, 0]
                
                if validate_initial_params(initial_params, Y_Tim):
                    return initial_params
                else:
                    raise ValueError("Invalid Stock Stationary Solution. Terminating")
            
            except (Exception, ValueError) as e:
                if self.verbose:
                    print(f"Stationary fitting failed: {e}")
                raise ValueError("Failed to find valid initial parameters.")


        else:
            # try stationary solution
            try:
                loc, scale, shape = fit_gev_params(Y_Dat)

                # For stationary, just 3 params
                initial_params = [loc, scale, shape]

                if validate_initial_params(initial_params, Y_Tim):
                    return initial_params

            except (Exception, ValueError) as e:
                if self.verbose:
                    print(f"Stationary fitting failed: {e}")
                raise ValueError("Failed to find valid initial parameters.")
            
            # Use stock solution
            try:
                initial_params = [loc, scale, -0.1]

                if not validate_initial_params(initial_params, Y_Tim):
                    raise ValueError("Invalid Stock Stationary Solution. Terminating")
            
            except (Exception, ValueError) as e:
                if self.verbose:
                    print(f"Stationary fitting failed: {e}")
                raise ValueError("Failed to find valid initial parameters.")


    def generate_sample_data(self, x_prm0: list):
        # True parameters P0 = [xi0,xi1,sgm0,sgm1,mu0,mu1]

        x_n = 10000
        x_time = np.linspace(0, 1, x_n)

        # Block length
        x_block = 1

        if len(x_prm0) == 3:
            x_xsm = np.array(
                [
                    np.ones(x_n) * x_prm0[0],
                    np.ones(x_n) * x_prm0[1],
                    np.ones(x_n) * x_prm0[2],
                ]
            )

        elif len(x_prm0) == 6:
            x_xsm = np.array(
                [
                    np.ones(x_n) * x_prm0[0] + x_time * x_prm0[1],
                    np.ones(x_n) * x_prm0[2] + x_time * x_prm0[3],
                    np.ones(x_n) * x_prm0[4] + x_time * x_prm0[5],
                ]
            )

        x_xsm = x_xsm.T

        x_dat = genextreme.rvs(c=x_xsm[:, 0], scale=x_xsm[:, 1], loc=x_xsm[:, 2])

        return x_dat

    def plot_trace_plots(self, samples):
        num_params = len(self.initial_params)

        fig, axes = plt.subplots(num_params, figsize=(12, 12))

        # shape parameter compensation
        sample_data = samples.copy()

        if self.non_stationary:
            sample_data[["shape_0", "shape_1"]] = (
                sample_data[["shape_0", "shape_1"]] * -1
            )
        else:
            sample_data["shape"] = sample_data["shape"] * -1

        for i, ax in enumerate(axes):
            ax.plot(sample_data.iloc[:, i])
            # ax.axvline(x=1000, color='black')
            parameter = sample_data.columns[i]
            ax.set_ylabel(f"{parameter}")
            ax.set_xlabel("Iteration")
        plt.tight_layout()
        plt.show()

    def plot_parameter_distributions(self, samples):
        # Plot distribution plots
        fig, axes = plt.subplots(len(self.initial_params), figsize=(8, 6))

        # shape parameter compensation
        sample_data = samples.copy()

        if self.non_stationary:
            sample_data[["shape_0", "shape_1"]] = (
                sample_data[["shape_0", "shape_1"]] * -1
            )
        else:
            sample_data["shape"] = sample_data["shape"] * -1

        for i, ax in enumerate(axes):
            ax.hist(sample_data.iloc[:, i], bins=30)
            # ax.axvline(x=1000, color='black')
            parameter = sample_data.columns[i]
            ax.set_ylabel(f"{parameter}")
        plt.tight_layout()
        plt.show()

    def plot_acceptance_rate(self, ar):
        fig = px.line(ar)
        fig.show()

    def plot_negative_log_likelihood(self):
        fig = px.line(-self.nll)
        fig.show()

    def plot_return_values(self, samples):
        RtrPrd = 100  # Return Period
        nRls = 250  # Number of samples to use.

        # Get the time_step value for 2025 & 2125
        t_start = (2025 - 2015) / (2100 - 2015)
        t_end = (2125 - 2015) / (2100 - 2015)

        tXi = samples.iloc[:, 4] + t_start * samples.iloc[:, 5]
        tSgm = samples.iloc[:, 2] + t_start * samples.iloc[:, 3]
        tMu = samples.iloc[:, 0] + t_start * samples.iloc[:, 1]

        # Return value calculations at start
        RV_Start = genextreme.ppf(1 - 1 / RtrPrd, c=tXi, loc=tMu, scale=tSgm)

        # Parameter estimates at end
        tXi_End = samples.iloc[:, 4] + t_end * samples.iloc[:, 5]
        tSgm_End = samples.iloc[:, 2] + t_end * samples.iloc[:, 3]
        tMu_End = samples.iloc[:, 0] + t_end * samples.iloc[:, 1]

        # Return value calculations at end
        RV_End = genextreme.ppf(1 - 1 / RtrPrd, c=tXi_End, loc=tMu_End, scale=tSgm_End)
        RV_Delta = RV_End - RV_Start
        # Summary statistics
        Prb = np.nanmean(RV_End > RV_Start)
        Cdf = np.column_stack((np.sort(RV_Start), np.sort(RV_End)))
        Qnt = np.percentile(RV_Start, [2.5, 50, 97.5]), np.percentile(
            RV_End, [2.5, 50, 97.5]
        )

        RV_Delta_Return = [
            RV_Delta,
            RV_Start,
            RV_End,
            Prb,
            np.nanpercentile(RV_Delta, 2.5),
            np.nanmedian(RV_Delta),
            np.nanpercentile(RV_Delta, 97.5),
        ]
        if self.verbose:
            # Create subplots and plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot CDF for RV_Start and RV_End
            axes[0].plot(
                np.sort(RV_Start),
                np.linspace(0, 1, len(RV_Start), endpoint=False),
                "k",
                label="RV_Start",
                linewidth=2,
            )
            axes[0].plot(
                np.sort(RV_End),
                np.linspace(0, 1, len(RV_End), endpoint=False),
                "r",
                label="RV_End",
                linewidth=2,
            )
            axes[0].set_title("Distribution of RV_Start (k) and RV_End (r)")
            axes[0].set_xlabel(f"{RtrPrd}-year maximum value")
            axes[0].set_ylabel("$F_{%gYearMaximum}$" % RtrPrd)
            axes[0].legend()

            # Plot log-scale CDF for RV_Start and RV_End
            axes[1].plot(
                np.sort(RV_Start),
                np.log10(1 - np.linspace(0, 1, len(RV_Start), endpoint=False)),
                "k",
                linewidth=2,
            )
            axes[1].plot(
                np.sort(RV_End),
                np.log10(1 - np.linspace(0, 1, len(RV_End), endpoint=False)),
                "r",
                linewidth=2,
            )
            axes[1].set_title("Log-scale Distribution of RV_Start (k) and RV_End (r)")
            axes[1].set_xlabel(f"{RtrPrd}-year maximum value")
            axes[1].set_ylabel("$\\log_{10}(1-F_{%gYearMaximum})$" % RtrPrd)

            # Adjust layout
            plt.tight_layout()
            plt.show()

            if (Prb > 0.975) or (Prb <= 0.025):
                print(f"Prb(RVEnd > RVStart) = {Prb:.2f} SIGNIFICANT")
            else:
                print(f"Prb(RVEnd > RVStart) = {Prb:.2f} NOT SIGNIFICANT")

            # We want to display the difference in start and end
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            RV_Delta = RV_End - RV_Start
            axes[0].plot(
                np.sort(RV_Delta),
                np.linspace(0, 1, len(RV_Delta), endpoint=False),
                "k",
                label="RV_Delta",
                linewidth=2,
            )

            axes[0].set_title("Distribution of RV_Delta")
            axes[0].set_xlabel(f"{RtrPrd}-year maximum value difference")
            axes[0].set_ylabel("$F_{%gYearMaximumdiff}$" % RtrPrd)

            axes[1].hist(RV_Delta)

            axes[1].set_title("Histogram of RV_Delta")
            axes[1].set_xlabel(f"{RtrPrd}-year maximum value difference")
            plt.tight_layout()
            plt.show()

        return RV_Delta_Return

    # Define a function to swap the sign of the shape parameter.
    # This is needed due to non-standard parameterisation in python.
    def shape_swap(self, samples):
        sample_data = samples.copy()
        if self.non_stationary:
            sample_data[["shape_0", "shape_1"]] = (
                sample_data[["shape_0", "shape_1"]] * -1
            )
        else:
            sample_data["shape"] = sample_data["shape"] * -1

        return sample_data

    def run(self, n_samples, n2plt, burn_in=1000, thinning=1, beta=0.05, NGTSTR=0.1):
        """
        Note that the genextremefunction uses the convention for the sign of the shape
        given in the documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
        We compensate for this when displaying graphs, but keep it for computation purposes.
        """

        samples, total_accepted, ar = self.metropolis_hastings(
            n_samples, n2plt, burn_in, thinning, beta
        )
        parameter_medians = samples.median()

        return samples, total_accepted, ar, parameter_medians
