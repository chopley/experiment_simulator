class PermutationTest:
    import copy
    import random
    import numpy as np
    import time

    @classmethod
    def test_collection_rate_function(cls, a, b, kwargs):
        """
        Helper function for get_collection_rate
        """
        import pandas as pd

        time_column = kwargs["time_column"]
        df_a = pd.DataFrame(a.groupby(time_column).sum()).reset_index()
        df_b = pd.DataFrame(b.groupby(time_column).sum()).reset_index()
        df_a["collection_rate_a"] = 100 * (
            df_a["collected_during"] / df_a["billed_during"]
        )
        df_b["collection_rate_b"] = 100 * (
            df_b["collected_during"] / df_b["billed_during"]
        )
        df_c = df_a.merge(df_b, left_on=["time_passed"], right_on=["time_passed"])
        df_c["diff"] = df_c["collection_rate_a"] - df_c["collection_rate_b"]
        test_stat = df_c["diff"].sum()
        # print(df_c)
        return test_stat

    @classmethod
    def randomize_by_invoice_id(cls, experiment_df):
        import pandas as pd

        e_df = experiment_df.copy()
        invoice_ids = pd.DataFrame(experiment_df["invoice_id"].unique())
        randomize_df = invoice_ids.sample(frac=1).reset_index(drop=True)

        invoice_ids_a = randomize_df[0 : int(len(randomize_df) / 2)].copy()
        invoice_ids_b = randomize_df[int(len(randomize_df) / 2) :].copy()
        invoice_ids_a.columns = ["invoice_id"]
        invoice_ids_b.columns = ["invoice_id"]
        a = e_df.merge(
            invoice_ids_a, how="inner", left_on=["invoice_id"], right_on=["invoice_id"]
        ).copy()
        b = e_df.merge(
            invoice_ids_b, how="inner", left_on=["invoice_id"], right_on=["invoice_id"]
        ).copy()
        return (a, b)

    @classmethod
    def randomize_by_user_id(cls, experiment_df):
        """

        :param pS:
        :return:
        """
        import pandas as pd

        user_ids = pd.DataFrame(experiment_df["user_id"].unique())
        randomize_df = user_ids.sample(frac=1).reset_index(drop=True)
        user_ids_a = randomize_df[0 : int(len(randomize_df) / 2)].copy()
        user_ids_b = randomize_df[int(len(randomize_df) / 2) :].copy()
        user_ids_a.columns = ["user_id"]
        user_ids_b.columns = ["user_id"]
        a = experiment_df.merge(
            user_ids_a, how="inner", left_on=["user_id"], right_on=["user_id"]
        ).copy()
        b = experiment_df.merge(
            user_ids_b, how="inner", left_on=["user_id"], right_on=["user_id"]
        ).copy()
        return (a, b)

    @classmethod
    def randomize_by_row(cls, experiment_df):
        experiment_df = experiment_df.sample(frac=1).reset_index(drop=True)
        a = experiment_df[0 : int(len(experiment_df) / 2)].copy()
        b = experiment_df[int(len(experiment_df) / 2) :].copy()
        return (a, b)

    @classmethod
    def get_collection_rate(cls, dataframe, n_randomizations, kwargs):
        """
        Function to handle returning permutation test for collection rate
        First calculate the collection rate for each time period, and sum the absolute values
        for the un-randomized case- the test statistics is the difference between these values for the
        two experimental conditions
        Then shuffle, and recalculate the above.
        """
        import time
        import sys
        import copy
        import numpy as np

        group_columns = kwargs["group_columns"]
        group_values = kwargs["group_values"]
        time_column = kwargs["time_column"]
        randomization_function = kwargs["randomization_function"]

        keep_columns = group_columns
        keep_columns.append(time_column)
        keep_columns.append("collected_during")
        keep_columns.append("billed_during")
        keep_columns.append("user_id")
        keep_columns.append("invoice_id")
        # only keep the columns we need to speed up the shuffling...
        dataframe = dataframe[keep_columns].copy()
        # get the a experimental condition as defined by the group_column:group_value
        a = dataframe[(dataframe[group_columns[0]] == group_values[0])].copy()
        # get the b experimental condition as defined by the group_column:group_value
        b = dataframe[(dataframe[group_columns[0]] == group_values[1])].copy()
        threshold = PermutationTest.test_collection_rate_function(a, b, kwargs)
        print("Threshold " + str(threshold))

        experiment_df = copy.copy(
            dataframe[
                (dataframe[group_columns[0]] == group_values[0])
                | (dataframe[group_columns[0]] == group_values[1])
            ]
        )
        permuted_values = []
        start_time = time.time()

        for i in range(0, n_randomizations):
            (a, b) = randomization_function(experiment_df)
            permuted_values.append(
                PermutationTest.test_collection_rate_function(a, b, kwargs)
            )
            time_now = time.time()
            time_cycle = (time_now - start_time) / (i + 1)
            estimated_time_remaining = (n_randomizations - (i + 1)) * time_cycle
            f = (
                "Time [ms] per shuffle: "
                + str(np.round(time_cycle * 1000))
                + " Estimated time remaining: "
                + str(np.round(estimated_time_remaining, 0))
            )
            sys.stdout.write("\r" + str(f))
            sys.stdout.flush()

        p_val = len(np.where(permuted_values >= threshold)[0]) / n_randomizations
        return (threshold, p_val, permuted_values)

    def permutation_test_statistic_time_series(
        dataframe, n_randomizations, test_function, **kwargs
    ):
        (threshold, p_val, permuted_values) = test_function(
            dataframe, n_randomizations, kwargs
        )

        return (threshold, p_val, permuted_values)
