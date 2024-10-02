import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import norm
from scipy.linalg import sqrtm
import openturns as ot

from multiprocessing import Lock, Process, Queue, current_process
import time
import queue  # imported for using queue.Empty exception


class nhgr_mcmc:
    def __init__(self, data, verbose=True):
        self.verbose = verbose  # To not print initial diagnostic information

        # If there is a nan value, the simply drop that value
        self.data = data[~np.isnan(data)]
        self.initial_params = self.find_starting_parameters(self.data)

    ### Core Functions ###
    def log_likelihood(self, params, time_steps):
        loc_0, loc_1, scale_0, scale_1 = params
        loc = np.array(loc_0 + time_steps * loc_1)
        scale = np.array(scale_0 + time_steps * scale_1)

        return np.sum(norm.logpdf(self.data, loc=loc, scale=scale))

    def propose(
        self, params, iteration, beta, total_accepted, samples, burn_in, time_steps
    ):
        # Make sure the total_accepted array has float values
        total_accepted = np.array(total_accepted, dtype=float)

        SH = sqrtm(np.cov(total_accepted, rowvar=False))
        SH = np.real(np.array(SH))
        z1 = np.random.normal(0, 1, size=len(params))
        z2 = np.random.normal(0, 1, size=len(params))
        y1 = (2.38 / np.sqrt(len(params))) * np.matmul(SH, z1)
        y2 = (0.1 / np.sqrt(len(params))) * z2
        new_parameters = params + (1 - beta) * y1 + beta * y2

        return new_parameters

    def acceptance_prob(self, old_params, new_params, time_steps):
        log_likelihood_old = self.log_likelihood(old_params, time_steps)
        log_likelihood_new = self.log_likelihood(new_params, time_steps)

        self.nll = pd.concat(
            [self.nll, pd.DataFrame({"negative_log_likelihood": [log_likelihood_old]})],
            ignore_index=True,
        )

        log_ratio = log_likelihood_new - log_likelihood_old

        return np.exp(log_ratio)

    def metropolis_hastings(
        self, n_samples, n2plt, burn_in=1000, thinning=10, beta=0.5, NGTSTR=0.1
    ):
        samples = pd.DataFrame(
            columns=[
                "location_0",
                "location_1",
                "scale_0",
                "scale_1",
            ]
        )
        total_accepted = pd.DataFrame(
            columns=[
                "location_0",
                "location_1",
                "scale_0",
                "scale_1",
            ]
        )

        ar = pd.DataFrame(columns=["acceptance_rate"])
        self.nll = pd.DataFrame(columns=["negative_log_likelihood"])
        current_params = self.initial_params
        time_steps = np.linspace(0, 1, len(self.data))

        for i in range(n_samples):
            if i <= burn_in:
                for j in range(0, 4):
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

        Y_nT = len(Y_Dat)
        Y_Tim = np.linspace(0, 1, Y_nT)
        Y_nB = 10
        Y_Blc = np.linspace(0, Y_nT, Y_nB + 1).astype(int)
        tRgr = np.zeros((Y_nB, 3))

        for iB in range(Y_nB):
            block_data = Y_Dat[Y_Blc[iB] : Y_Blc[iB + 1]]

            loc = np.mean(block_data)
            scale = np.std(block_data)

            tTim = np.mean(Y_Tim[Y_Blc[iB] : Y_Blc[iB + 1]])
            tRgr[iB] = [tTim, loc, scale]

        tRgr = tRgr[tRgr[:, 0].argsort()]
        Y_XSMStart = np.zeros(4)

        if self.verbose:
            plt.figure(figsize=(12, 6))
            var = ["Location", "Scale"]

            for j in range(2):
                plt.subplot(2, 3, 4 + j)
                plt.plot(tRgr[:, 0], tRgr[:, j + 1], "b.-", label="Data")
                coef = np.polyfit(tRgr[:, 0], tRgr[:, j + 1], 1)
                plt.plot(
                    tRgr[:, 0],
                    np.polyval(coef, tRgr[:, 0]),
                    "k.-",
                    label="Regression Line",
                )
                Y_XSMStart[2 * j : 2 * (j + 1)] = coef
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title(f"{var[j]}")
                plt.grid(True)

            plt.tight_layout()
            plt.show()
        else:
            for j in range(2):
                coef = np.polyfit(tRgr[:, 0], tRgr[:, j + 1], 1)
                Y_XSMStart[2 * j : 2 * (j + 1)] = coef

        initial_params = [
            Y_XSMStart[1],
            Y_XSMStart[0],
            Y_XSMStart[3],
            Y_XSMStart[2],
        ]

        nll = -self.log_likelihood(params=initial_params, time_steps=Y_Tim)

        return initial_params

    def plot_trace_plots(self, samples):
        num_params = len(self.initial_params)

        fig, axes = plt.subplots(num_params, figsize=(12, 12))

        # shape parameter compensation
        sample_data = samples.copy()

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

        for i, ax in enumerate(axes):
            ax.hist(sample_data.iloc[:, i], bins=50)
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
        nRls = len(samples)  # Number of samples to use.

        # Parameter estimates at start
        tSgm = samples.iloc[:, 2]
        tMu = samples.iloc[:, 0]

        # Parameter estimates at end
        tSgm_End = samples.iloc[:, 2] + samples.iloc[:, 3]
        tMu_End = samples.iloc[:, 0] + samples.iloc[:, 1]

        RV_Start = norm.ppf(np.random.rand(nRls)) * tSgm + tMu
        RV_End = norm.ppf((np.random.rand(nRls))) * tSgm_End + tMu_End

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
