import numpy as np
import pandas as pd
from numpy.random import choice

import matplotlib.pyplot as plt


class BootstrapTools:
    """
    Tools for calculating bootstrap estimates
    """

    @classmethod
    def collection_rate(
        cls,
        dataframe,
        billed_column_name,
        collected_column_name,
        time_column,
        time_value,
    ):
        """


        :param dataframe:
        :param billed_column_name:
        :param collected_column_name:
        :param time_column:
        :param time_value:
        :return:

        This is the collection rate calculator used for the simulations where there is a single
        row for each of the users in an experiment
        """
        total_billed = dataframe[billed_column_name].sum()
        # only use the values that were collected before the time
        data_time = (
            dataframe[dataframe[time_column] <= time_value]
            .reset_index(drop=True)
            .copy()
        )
        total_collected = data_time[collected_column_name].sum()
        collection_rate = 100 * total_collected / total_billed
        return collection_rate

    @classmethod
    def experiment_collection_rate(
        cls,
        dataframe,
        billed_column_name,
        collected_column_name,
        time_column,
        time_value,
    ):
        """

        :param dataframe:
        :param billed_column_name:
        :param collected_column_name:
        :param time_column:
        :param time_value:
        :return:

        This collection rate calculator relies on a dataframe of the following form
            arm_name        billed_during       collected_during    invoice_id  time_passed   user_id
            no-phone-call    4818.80            0.00                35884234    15            20615997
            no-phone-call    4158.08            0.00                35882282    15            20614126
            no-phone-call    4131.20            0.00                35949783    15            20672658
            no-phone-call    3830.19            0.00                31592672    15            12906825
            no-phone-call    5400.32            0.00                35882274    15            20614118

            i.e. there are multiple rows for each user even when there is no payment
        """
        import numpy as np

        # only use the values that were collected before the time
        data_time = (
            dataframe[dataframe[time_column] == time_value]
            .reset_index(drop=True)
            .copy()
        )

        if data_time.shape[0] > 30:
            total_billed = data_time[billed_column_name].sum()
            total_collected = data_time[collected_column_name].sum()
            collection_rate = 100 * total_collected / total_billed
            return collection_rate
        else:
            return np.nan

    @classmethod
    def get_unique_column_values(cls, dataframe, column_names):

        unique_values = str(
            dataframe[column_names]
            .groupby(column_names)
            .size()
            .reset_index()[column_names]
            .values.tolist()[0]
        )

        return unique_values

    @classmethod
    def bootstrap_time_estimate(
        cls,
        dataframe,
        group_columns,
        time_column,
        bootstrap_parameter_function,
        billed_amounts="billed_during",
        collected_amounts="collected_during",
        n_bootstrap_samples=100,
    ):
        """

        :param dataframe:  Pandas dataframe
        :param group_columns:  List of columns that define the categorical groups e.g. experiment_id etc.
        :param time_column:  String that defines the column used for the time passed
        :param bootstrap_parameter_function: Function that is used to calculate the bootstrap e.g. bootstrap_tools.collection_rate
        :param billed_amounts: Column name that contains the billed amounts e.g. 'billed_during'
        :param collected_amounts: Column name that contains the collected amounts e.g. 'collected_during'
        :return: Pandas dataframe with the following columns 1. time, 2. group_id 3. estimate 4. std_dev 5.sample_size, 6. bootstrap estimates
        """
        import numpy as np

        # first extract the unique time values from the appropriate column and sort them
        time_values = sorted(list(set(list(dataframe[time_column].values))))
        # next determine the unique combinations of the categorical columns we are using
        grouped_dataframes = [
            pd.DataFrame(y) for x, y in dataframe.groupby(group_columns, as_index=False)
        ]
        # we use this to append each of the group estimates
        bootstrap_values = []
        for group_df in grouped_dataframes:
            for time_value in time_values:
                # This stores each of the resampled mean values
                bootstrap_means = []
                group_names = BootstrapTools.get_unique_column_values(
                    group_df, group_columns
                )
                # Calculate the metric using the appropriate bootstrap_parameter_function. Each estimate is generated
                # using sampling with replacement. Central limit theorem says that the resulting estimate will be normally distributed
                # so we can simply use normal distribution parameters e.g. mean and std_dev
                for j in range(0, n_bootstrap_samples):
                    sample = group_df.sample(frac=1, replace=True).copy()
                    bootstrap_estimate = bootstrap_parameter_function(
                        sample,
                        billed_amounts,
                        collected_amounts,
                        time_column,
                        time_value,
                    )
                    bootstrap_means.append(bootstrap_estimate)

                bootstrap_values.append(
                    {
                        "time": time_value,
                        "group_id": group_names,
                        "estimate": np.mean(bootstrap_means),
                        "std_dev": np.std(bootstrap_means),
                        "sample_size": group_df.shape[0],
                        "estimates": bootstrap_means,
                    }
                )
        return pd.DataFrame(bootstrap_values)

    def plot_bootstrapped_uncertainties(
        bs_estimates,
        xlabel="Time",
        ylabel="Collection Rate",
        title="Collection Rate",
        colormap="ocean",
        ylim=(0, 100),
        figsize=(10, 5),
        alpha=0.1,
        interpolation_n=1,
        markers=["x", "o", "v"],
        linestyles=["-", "--", "-."],
        uncertainty_estimates=True,
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        from cycler import cycler

        from scipy.interpolate import make_interp_spline, BSpline

        # first reshape the data to make it suitable for plotting
        val = bs_estimates[["time", "group_id", "estimate", "std_dev"]].pivot(
            index="time", columns="group_id", values=["estimate", "std_dev"]
        )
        val_stack = val.stack()
        data_plot = val_stack.reset_index()

        fig, ax = plt.subplots(figsize=figsize)
        # we get the categories that we will use for plotting
        categories = list(data_plot["group_id"].unique())

        n_markers = len(markers)
        n_linestyles = len(linestyles)

        n_colors = np.int(np.round(len(categories) / (n_markers * n_linestyles) + 1))
        new_colors = [
            plt.get_cmap(colormap)(1.0 * i / n_colors) for i in range(n_colors)
        ]
        cc = (
            cycler(linestyle=linestyles)
            * cycler(color=new_colors)
            * cycler(marker=markers)
        )
        ax.set_prop_cycle(cc)

        i = 0
        # iterate over the categories and plot the mean values against time, together with fill values to
        # indicate the 95% confidence interval (i.e. 2 * sigma )
        for i in range(0, len(categories)):
            category = categories[i]
            data_cat = data_plot[data_plot["group_id"] == category].copy()
            time = np.array(data_cat["time"].values, dtype=np.float64)
            data_cat = data_plot[data_plot["group_id"] == category].copy()
            meanst = np.array(data_cat["estimate"].values, dtype=np.float64)
            std = np.array(data_cat["std_dev"].values, dtype=np.float64)
            if interpolation_n > 1:
                # interpolation_n is the number of interpolation points between actual data points- i.e. if the data are
                # returned for every ten days , interpolation_n will provide a resolution of 5 days
                t_new = np.linspace(time.min(), time.max(), len(time) * interpolation_n)
                spl = make_interp_spline(time, meanst, k=3)  # type: BSpline
                spl_std = make_interp_spline(time, std, k=3)  # type: BSpline
                meanst_smooth = spl(t_new)
                std_smooth = spl_std(t_new)

                ax.plot(t_new, meanst_smooth, label=category, linewidth=3)
                if uncertainty_estimates:
                    ax.fill_between(
                        t_new,
                        meanst_smooth - 2 * std_smooth,
                        meanst_smooth + 2 * std_smooth,
                        alpha=alpha,
                        interpolate=True,
                    )
            else:
                ax.plot(time, meanst, label=category, linewidth=3)
                if uncertainty_estimates:
                    ax.fill_between(
                        time,
                        meanst - 2 * std,
                        meanst + 2 * std,
                        alpha=alpha,
                        interpolate=True,
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.suptitle(title)
        ax.legend(loc="upper left")
        ax.grid(True)
        ax.set_ylim(ylim)
        return (fig, ax)

    def plot_bootstrapped_uncertainties_errorbars(
        bs_estimates, xlabel="Time", ylabel="Collection Rate", title="Collection Rate"
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        # first reshape the data to make it suitable for plotting
        val = bs_estimates[["time", "group_id", "estimate", "std_dev"]].pivot(
            index="time", columns="group_id", values=["estimate", "std_dev"]
        )
        val_stack = val.stack()
        data_plot = val_stack.reset_index()

        fig, ax = plt.subplots()
        clrs = sns.color_palette("tab20c")
        i = 0

        # we get the categories that we will use for plotting
        categories = list(data_plot["group_id"].unique())
        # iterate over the categories and plot the mean values against time, together with fill values to
        # indicate the 95% confidence interval (i.e. 2 * sigma )
        for i in range(0, len(categories)):
            category = categories[i]
            data_cat = data_plot[data_plot["group_id"] == category].copy()
            time = np.array(data_cat["time"].values, dtype=np.float64)
            data_cat = data_plot[data_plot["group_id"] == category].copy()
            meanst = np.array(data_cat["estimate"].values, dtype=np.float64)
            std = np.array(data_cat["std_dev"].values, dtype=np.float64)

            ax.plot(time, meanst, label=category, c=clrs[i])
            ax.errorbar(
                time,
                meanst,
                yerr=2 * std,
                fmt="o",
                color=clrs[i],
                ecolor=clrs[i],
                elinewidth=3,
                capsize=0,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.suptitle(title)
        ax.legend()
        return fig

    def analyze_bootstrap_time_estimate(
        dataframe,
        bootstrap_parameter_function,
        group_columns,
        time_column="time_passed",
        n_bootstrap_samples=100,
    ):
        bs_estimates = BootstrapTools.bootstrap_time_estimate(
            dataframe,
            group_columns,
            time_column,
            bootstrap_parameter_function,
            billed_amounts="billed_during",
            collected_amounts="collected_during",
            n_bootstrap_samples=n_bootstrap_samples,
        )
        return bs_estimates


def generate_bootstrapped_plot(
    exp_arms, variable, figsize, layout, xlim, figtitle, xlabel, percentile, n_bins
):
    """
    This is a plotting function that can be used to display the different experimental bootstrap results
    This specifically plots the average amount collected per patient.

    Parameters:
    :arg1 exp_arms(list) - list of the experimental arms - these will be populated in the function
    :arg2 variable(string) - which of the bootstrapped estimates to use in the objects e.g. "df.bootstrap_means_df" or "df.bootstrap_ave_collected_df"
    :arg3 figsize(tuple) - size of the resulting plot figure (e.g. (10,10))
    :arg4 layout(tuple) - layout of subfigures (e.g. (2,2) for 2x2 plot)
    :arg5 xlim(tuple) - min/max of xlimits
    :arg6 title used for the plot
    :arg6 xlabel
    :arg7 percentile (tuple) - (2.5,97.5)
    :arg8 n_bins

    Example ...
    bootstrap.generate_bootstrapped_plot(exp_arms,
                                    "df.bootstrap_ave_collected_df",
                                    figsize=(10,10),
                                    layout = (2,2),
                                    xlim = (0,10),
                                    figtitle = 'Average Collected Amounts\nWith Discounts',
                                    xlabel = "Average Collected",
                                    percentile = (2.5,97.5),
                                    n_bins = 20)

    """
    arm_objects = [arm for arm in exp_arms]
    # generate bootstrapped estimates of the total collected per arm
    ncolumns = layout[0]
    nrows = layout[1]
    fig, axes = plt.subplots(nrows, ncolumns, figsize=figsize)
    i = 0
    for df in arm_objects:
        var = eval(variable)
        j = np.int(np.floor(i / ncolumns))
        k = i % ncolumns
        if nrows > 1 and ncolumns > 1:
            current_axis = axes[j, k]
        elif nrows == 1 and ncolumns > 1:
            current_axis = axes[k]
        elif nrows > 1 and ncolumns == 1:
            current_axis = axes[j]

        var.hist(bins=n_bins, ax=current_axis)
        current_axis.set_title(df.arm_name)
        perc_low = np.percentile(var, percentile[0])
        perc_high = np.percentile(var, percentile[1])
        current_axis.axvline(
            x=perc_low, ymin=0, ymax=1, linestyle="--", color="r", lw=4
        )
        current_axis.axvline(
            x=perc_high, ymin=0, ymax=1, linestyle="--", color="r", lw=4
        )
        current_axis.set_xlim(xlim)
        current_axis.set_xlabel(xlabel)
        i = i + 1
    fig.suptitle(figtitle, fontsize=20)


def generate_experiment_arms(df, arm_categories, n_bootstrapped_samples):
    """
    A useful function that can be used to generate a series of experiment_arms object based on values in the dataframe
    Parameters:
    :arg1 df(pandas dataframe) - dataframe of the data of all users involved in the experiments
    :arg2 arm_categories(list) - list of column names in the dataframe that will be used to define the different experimental arms
    :arg3 n_bootstrapped_samples(int) - number of bootstrapping resamples used to estimate the distribution

    Returns:
    :arg1 exp_arms - experiment_arms objects(list)
    :arg2 arms -
    :arg3 arm_names(list) - names of the arms used
    """
    from library.bootstrapping import bootstrap as bootstrap

    arm_combinations = (
        df[arm_categories]
        .groupby(arm_categories)
        .size()
        .reset_index()[arm_categories]
        .values.tolist()
    )
    arms = [pd.DataFrame(y) for x, y in df.groupby(arm_categories, as_index=False)]
    exp_arms = []
    for arm in arms:
        exp_arms.append(bootstrap.ExperimentArms())
    arm_names = [str(arm) for arm in arm_combinations]
    for i in range(0, len(exp_arms)):
        exp_arms[i].start_arm(arms[i], arm_names[i], n_bootstrapped_samples)
    return (exp_arms, arms, arm_names)


class ExperimentArms:
    """
    Here we create a simple class to provide methods to operate across an experimental arm
    It can run a bootstrap estimate of mean collection rate, and pre-populate various dataframes needed for effect size estimates
    Each arm should be associated with an instance of this class

    """

    def __init__(self):
        print("Starting Experiment Arm")

    def start_arm(self, data, arm_name, n_bootstrapped_samples):
        """
        This method is used to initialise the object with the data associated with a specific experimental arm. It sets various attributes of the object with these data fields.

        Parameters:
        :arg1 data(dataframe): DF with columns "collected_during", and "billed_during"  and "total_due" - "billed_during" is reduced after a discount event, hence the need for "total_due"
        :arg2 arm_name(string): name of the arm that will be represented by this object
        :arg3 n_bootstrapped_samples(int): number of bootstrapped samples that will be used to generate the estimates

        Returns:
        :List of data values
        """
        self.data = data
        self.n_bootstrapped_samples = n_bootstrapped_samples
        self.data_values = data[["collected_during", "billed_during"]].values
        self.collected = data[["collected_during"]].values
        self.billed = data[["billed_during"]].values
        self.total_due = data[["total_due"]].values
        self.collection_rate = (
            data["collected_during"].sum() / data["billed_during"].sum()
        )
        self.arm_name = arm_name
        self.bootstrap_estimate_collection_rate()
        self.bootstrap_estimate_average_collected_amount()
        self.bootstrap_estimate_total_billed()
        return self.data_values

    def bootstrap_estimate_collection_rate(self):
        """
        This method uses bootstrapping to estimates the collection rates

        Returns:
        :Pandas dataframe of the bootstrapped collection rates that can be used for the estimate distribution
        """
        bootstrap_means = []
        for j in range(0, self.n_bootstrapped_samples):
            sample = self.data_values[
                np.random.choice(
                    self.data_values.shape[0], self.data_values.shape[0], replace=True
                )
            ]
            sum_collected, sum_billed = sample.sum(axis=0)
            bootstrap_means.append(sum_collected / sum_billed)

        self.bs_estimate_collection_rate = 100.0 * pd.DataFrame(
            bootstrap_means
        )  # convert to percentage

    def bootstrap_estimate_average_collected_amount(self):
        """
        This method uses bootstrapping to estimates the average size of the collected amount

        Returns:
        :Pandas dataframe of the bootstrapped collected amount per bill that can be used for the estimate distribution
        """
        self.bootstrap_ave_collected = []
        for j in range(0, self.n_bootstrapped_samples):
            sample = self.collected[
                np.random.choice(
                    self.collected.shape[0], self.collected.shape[0], replace=True
                )
            ]
            ave_collected = np.nansum(sample) / self.collected.shape[0]
            self.bootstrap_ave_collected.append(ave_collected)

        self.bootstrap_ave_collected_df = pd.DataFrame(self.bootstrap_ave_collected)

    def bootstrap_estimate_total_billed(self):
        """
        This method uses bootstrapping to estimates the average size of the bills in this experimental arm

        Returns:
        :Pandas dataframe of the bootstrapped billed amount per bill that can be used for the estimate distribution
        """
        self.bootstrap_ave_total_due = []
        for j in range(0, self.n_bootstrapped_samples):
            sample = self.total_due[
                np.random.choice(
                    self.total_due.shape[0], self.total_due.shape[0], replace=True
                )
            ]
            ave_total_due = np.nansum(sample) / self.total_due.shape[0]
            self.bootstrap_ave_total_due.append(ave_total_due)

        self.bootstrap_ave_total_due_df = pd.DataFrame(self.bootstrap_ave_total_due)


def generate_simulated_repayments(percentages, probabilities, amounts):
    """
    Function to simulate a repayment probability distribution
    Parameters:
    :arg1 percentages(list) - list of ratio of repayment percentage- e.g. fully repaid is 1 and no repayment =0
    :arg2 probabilities(list) - list of the associated probability of all the repayment percentages- this must sum to 1
    :arg3 amounts(list) - list of amounts that need to have the repayment distribution applied to them

    Returns:
    :list of repayment amounts based on the indicated distributions from the input arguments
    Examples:
    Simulate a repayment rate of 60% of invoices fully paid
    ---- repayments = generate_simulated_repayments([0,1],[0.4,0.6],billed_amounts)
    Simulate a repayment rate of
    - 1. 60% of invoices fully paid
    - 2. 30% of invoices unpaid
    - 3. 10% of invoices half-paid
    ----repayments = generate_simulated_repayments([0,0.5,1],[0.3,0.5,0.6],billed_amounts)
    """
    repayment = []
    for amount in amounts:
        draw = choice(percentages, 1, p=probabilities)
        repayment.append(amount * draw[0])
    return repayment


def bootstrap_mean_comparison_one_sided(arms, alpha, n_bootstrap_samples):
    """
    Function to calculate the mean difference between two arms

    :arg1 arms - list of two  experiment_arms objects that will be used in the comparison
    :arg2 alpha - alpha value level e.g. 0.05
    :arg3 n_bootstrap_samples(int) - number of bootstrapped samples used
    :returns
    """
    mean_diff = []
    a = arms[0].data_values
    b = arms[1].data_values
    for i in range(0, n_bootstrap_samples):
        sample_a = a[np.random.choice(a.shape[0], a.shape[0], replace=True)]
        sample_b = b[np.random.choice(b.shape[0], b.shape[0], replace=True)]
        sum_collected_a, sum_billed_a = sample_a.sum(axis=0)
        sum_collected_b, sum_billed_b = sample_b.sum(axis=0)
        collection_rate_a = sum_collected_a / sum_billed_a
        collection_rate_b = sum_collected_b / sum_billed_b
        mean_diff.append(collection_rate_a - collection_rate_b)

    df_diff = pd.DataFrame(mean_diff)
    mean_either_side_zero = pd.DataFrame(df_diff[0] < 0)
    b = mean_either_side_zero[0].value_counts()
    if len(b) == 2:
        pvalue = (
            b.min() / b.max()
        )  # pvalue here is the ratio of those on the other side of zero from the mean, over those on the same side
    else:
        pvalue = 0
    low_ci = np.percentile(df_diff, alpha)
    high_ci = np.percentile(df_diff, 100.0 - (alpha))
    if high_ci < 0 or low_ci > 0:
        stat_test = "Alternative"
    else:
        stat_test = "Null"
    return (
        df_diff,
        (low_ci, high_ci),
        stat_test,
        pvalue,
        (arms[0].arm_name, arms[1].arm_name),
    )


def bootstrap_mean_comparison_multi_arm_one_sided(arms, alpha, n_bootstrap_samples):
    """
    Function to handled a list of all arms, generate the differen pair-wise comparisons and then run the one-sided
    mean comparison across all pairs
    """
    import itertools

    combinations = list(itertools.combinations(arms, 2))
    diff_bootstrap = []
    ci_bootstrap = []
    stat_test_bootstrap = []
    arm_comp_bootstrap = []
    pvalue_bootstrap = []
    for combination in combinations:
        (
            df_diff,
            CI,
            stat_test,
            pvalue,
            arm_comp,
        ) = bootstrap_mean_comparison_one_sided(combination, alpha, n_bootstrap_samples)
        diff_bootstrap.append(df_diff)
        ci_bootstrap.append(CI)
        stat_test_bootstrap.append(stat_test)
        pvalue_bootstrap.append(pvalue)
        arm_comp_bootstrap.append(arm_comp)
    return (
        diff_bootstrap,
        ci_bootstrap,
        stat_test_bootstrap,
        pvalue_bootstrap,
        arm_comp_bootstrap,
    )
