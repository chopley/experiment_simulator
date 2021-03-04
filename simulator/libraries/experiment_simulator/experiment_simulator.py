import numpy as np
import pandas as pd
from numpy.random import choice


class ExperimentSimulation:
    """
    This class is used to handle simulated experimental data
    """

    def __init__(self):
        print("Starting")

    def generate_pareto_simulated_billed_amounts(
        self, n_billed_amounts, shape=6, scale=1000
    ):
        self.simulated_billed_amounts = (
            np.random.pareto(shape, n_billed_amounts) * scale
        )
        return self.simulated_billed_amounts

    def get_repayment_defn(self, amount, exp_definition):
        """Function to return a repayment amount depending on underlying probability of repayment

        Args:
            amount (float): The amount billed,
            dictionary with the following key/value pairs:
                categories : Different labels for the ranges used - len(categories) = n
                payment_ranges: definition of the different labelled billed amounts e.g. high_amount, low_amount len(payment_ranges)=m
                payment_rates: rates of payments for the different ranges len(payment_rates) = m

        Returns:
            float: The amount repaid after accounting for the repayment distribution
        """
        categories = exp_definition["categories"]
        payment_ranges = exp_definition["payment_ranges"]
        payment_rates = exp_definition["payment_rates"]
        # categories to allow different repayment distributions i.e. large bills are less likely to get repaid than smaller bills
        categories = [cat for cat in categories]
        # this dictionary is used to define the difference categories:
        # key: category name -defined above
        # values: list [billed_amount_range][repayment percentages][probability vector of each repayment percentage]
        list_of_payment_percentages = {}
        for i in range(0, len(categories)):
            list_of_payment_percentages.update(
                {
                    categories[i]: [
                        payment_ranges[i],
                        exp_definition["partial_payment_brackets"],
                        payment_rates[i],
                    ]
                }
            )
        # could be faster:
        for cat in categories:
            if (
                amount <= list_of_payment_percentages[cat][0][1]
                and amount >= list_of_payment_percentages[cat][0][0]
            ):
                cat_payment = cat

        percentages = list_of_payment_percentages[cat_payment][
            1
        ].copy()  # these represent the amount of the bill that is paid e.g. 0 is not paid, 1 is totally paid
        probabilities = list_of_payment_percentages[cat_payment][
            2
        ].copy()  # these represent the probabilities associated with each amount above.
        draw = choice(percentages, 1, p=probabilities)
        repayment = amount * draw[0]
        return repayment

    def run_experiment_simulation(self, n_bootstrapped_samples, exp_definition):
        """
        Here we simulate the repayment of the simulated bills
        Arg1: n_bootstrapped_samples(int) - number of times the bootstrap process is run
        Arg2: exp_definition (dict) :
        {'categories': names of the bill payment categories,
        'payment_ranges' : bill sizes associated with the above categories
        'partial_payment_brackets' : discreet %'s of the bill that is paid- if you assume only non-payment or full payment this would be [0,1]- if you want to allow for 50% payment [0,0.5,1] etc.'
        'payment_rates' :   The percentage of bills that fall into the different payment brackets- must have the same length as partial_payment_brackets
                            e.g. partial_payment_ranges_arm_1 = [0,0.25,0.5,0.75,1] and payment_rates_arm_1 [0.2, 0.01, 0.01, 0.01, 0.77]
                            means that the bill is not paid at all 20% of the time, 25% of the bill is paid 1% of the time, and the full bill is paid 77% of the time
        'arm_name' :name that we will use to refer to the arm
        {
        categories_arm_1 = ['low_amounts','high_amounts'],
        payment_ranges_arm_1 = [
                        [0,300],
                        [300,20000]],
        partial_payment_ranges_arm_1 = [0,0.25,0.5,0.75,1]
        payment_rates_arm_1 = [
                    [0.09,0.01,0.01,0.01,0.88], #low amount
                    [0.2,0.01,0.01,0.01,0.77] #high amount
                    ]
        }
        experiment_definition = {'categories':categories_arm_1,
                   'payment_rates':payment_rates_arm_1,
                   'payment_ranges':payment_ranges_arm_1,
                    'partial_payment_brackets':partial_payment_ranges_arm_1,
                   'arm_name':'arm_1'}
        """
        self.n_billed_amounts = len(self.simulated_billed_amounts)
        (self.billed_amounts, self.collected_amounts) = self.get_collected_amounts_defn(
            exp_definition, self.simulated_billed_amounts
        )
        self.arm_name = exp_definition["arm_name"]
        # generate the lognormal distribution of repayment days since assignment
        mu = 3
        sigma = 0.8
        self.time_to_collect = np.random.lognormal(mu, sigma, self.n_billed_amounts)
        # to generate the collection curves, we bin the time_to_collect into 15 day chunks
        windows = 15
        self.time_to_collect = np.round(self.time_to_collect / windows) * windows

        self.df_bill_amounts = pd.DataFrame(
            {
                "arm": self.arm_name,
                "billed_amounts": self.billed_amounts,
                "collected_amounts": self.collected_amounts,
                "time_to_collect": self.time_to_collect,
            }
        )
        return self.df_bill_amounts

    def get_collected_amounts_defn(
        self, definition, billed_amount_simulated_distribution
    ):
        """
        Method to simulate repayments based on the distribution specified:
        """
        # generate a decaying exponential of billed amounts that should look similar to our book
        billed_amounts = billed_amount_simulated_distribution
        # get collected amounts generated by the get_repayment function
        collected_amounts = []
        for billed_amount in billed_amounts:
            collected_amounts.append(self.get_repayment_defn(billed_amount, definition))
        return (billed_amounts, collected_amounts)

    @classmethod
    def generate_experiment(
        cls,
        payment_rates,
        payment_ranges,
        partial_payment_ranges,
        categories,
        n_arms=15,
        n_invoices_month=50000,
        n_months_running=3,
        n_bootstrapped_samples=250,
        time_windows=15,
    ):

        n_billed_amounts = np.int(
            np.round(n_invoices_month * n_months_running / n_arms)
        )
        experiment = []
        arms = []
        billed = []
        for key, value in payment_rates.items():
            payment_rates_temp = [
                [1 - value[0], 0.00, 0.00, 0.00, value[0]],  # low amount
                [1 - value[1], 0.00, 0.00, 0.00, value[1]],  # high amount
            ]
            experiment_temp = {
                "categories": categories,
                "payment_rates": payment_rates_temp,
                "payment_ranges": payment_ranges,
                "partial_payment_brackets": partial_payment_ranges,
                "arm_name": key,
            }

            experiment.append(experiment_temp)
            arm_temp = ExperimentSimulation()
            arm_temp.generate_pareto_simulated_billed_amounts(n_billed_amounts)
            billed_temp = arm_temp.run_experiment_simulation(
                n_bootstrapped_samples, experiment_temp
            )
            billed_temp["time_to_collect"] = (
                round(billed_temp["time_to_collect"] / time_windows) * time_windows
            )
            billed.append(billed_temp)
            arms.append(arm_temp)
        billed_df = pd.DataFrame()
        for bill in billed:
            billed_df = billed_df.append(bill)

        billed_df["collected_during"] = billed_df.collected_amounts
        billed_df["billed_during"] = billed_df.billed_amounts
        billed_df.loc[billed_df["time_to_collect"] > 100, "time_to_collect"] = 100

        return (arms, billed_df)
