from django.shortcuts import render
from .forms import ModelForm, AddressForm
from .models import Predictions
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import HttpResponse
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np


def simulate_experiment(n_arms, n_bootstraps, n_months, n_invoices_month, arm_defns):
    import matplotlib
    matplotlib.use('agg')
    from .libraries.experiment_simulator.experiment_simulator import ExperimentSimulation
    from .libraries.bootstrapping.bootstrap import BootstrapTools

    print(arm_defns)
    payment_rates = {
        'control': [0.05, 0.05],
        '0%_6_months': [0.06, 0.06],
        '15%_6months': [0.065, 0.065],
        '45%_6months': [0.08, 0.08]
    }
    payment_rates = arm_defns
    payment_ranges = [
        [0, 300],
        [300, 20000]]



    partial_payment_ranges = [0, 0.25, 0.5, 0.75, 1]
    categories = ['low_amounts','high_amounts']


    n_billed_amounts = np.int(np.round(n_invoices_month * n_months / n_arms))

    dunning_cycle_months = 3
    if(n_months - dunning_cycle_months <=0):
        n_dunning_cycles = 0.1
    else:
        n_dunning_cycles = n_months - dunning_cycle_months

    (arms, billed_df) = ExperimentSimulation.generate_experiment(
        payment_rates,
        payment_ranges,
        partial_payment_ranges,
        categories,
        n_arms=n_arms,
        n_invoices_month=n_invoices_month,
        n_months_running=n_dunning_cycles,
        n_bootstrapped_samples=n_bootstraps,
        time_windows=15)
    group_columns = ["arm"]
    time_column = "time_to_collect"
    bs_estimates = BootstrapTools.analyze_bootstrap_time_estimate(billed_df, BootstrapTools.collection_rate,
                                                                   group_columns, time_column)
    max_val = np.max(bs_estimates['estimate'] + bs_estimates['std_dev'])
    #max_val = 20
    (fig, ax) = BootstrapTools.plot_bootstrapped_uncertainties(bs_estimates, ylim=(0, np.ceil(max_val*1.1)))
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    print(bs_estimates)
    data = imgdata.getvalue()
    return data

def predict_model(request):
    n_arms = 4
    n_bootstraps = 250
    n_months = 3
    n_invoices_month = 10000
    arm_defns = {
        'control': [0.05, 0.05],
        '0%_6_months': [0.06, 0.06],
        '15%_6months': [0.065, 0.065],
        '45%_6months': [0.08, 0.08]
    }
    if request.method == 'POST':
        form = ModelForm(request.POST)

        # create a form instance and populate it with data from the request:
        form = ModelForm(request.POST)

        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            n_arms = form.cleaned_data['n_arms']
            arm_1_name = form.cleaned_data['arm_1_name']
            arm_2_name = form.cleaned_data['arm_2_name']
            arm_3_name = form.cleaned_data['arm_3_name']
            arm_4_name = form.cleaned_data['arm_4_name']
            arm_1_collection_rate = form.cleaned_data['arm_1_collection_rate'] /100
            arm_2_collection_rate = form.cleaned_data['arm_2_collection_rate'] /100
            arm_3_collection_rate = form.cleaned_data['arm_3_collection_rate'] /100
            arm_4_collection_rate = form.cleaned_data['arm_4_collection_rate'] /100
            n_bootstraps = form.cleaned_data['n_bootstraps']
            n_months = form.cleaned_data['n_months']
            n_invoices_month = form.cleaned_data['n_invoices_month']

            arm_defns = {arm_1_name: [arm_1_collection_rate,arm_1_collection_rate],
                    arm_2_name: [arm_2_collection_rate,arm_2_collection_rate],
                    arm_3_name: [arm_3_collection_rate,arm_3_collection_rate],
                    arm_4_name: [arm_4_collection_rate,arm_4_collection_rate]}

            context = {'form': form, 'graph': simulate_experiment(n_arms=n_arms,
                                                                  n_bootstraps=n_bootstraps,
                                                                  n_months=n_months,
                                                                  n_invoices_month=n_invoices_month,
                                                                  arm_defns=arm_defns)}
    # if a GET (or any other method) we'll create a blank form
    else:
        form = ModelForm(request.POST)
        context = {'form': form, 'graph': None}



    return render(request, 'home.html', context)

def predict_modela(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':

        # create a form instance and populate it with data from the request:
        form = ModelForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']

            # Run new features through ML model
            model_features = [
                [sepal_length, sepal_width, petal_length, petal_width]]
            loaded_model = pickle.load(
                open("ml_model/iris_model.pkl", 'rb'))
            prediction = loaded_model.predict(model_features)[0]

            prediction_dict = [{'name': 'setosa',
                                'img': 'https://alchetron.com/cdn/iris-setosa-0ab3145a-68f2-41ca-a529-c02fa2f5b02-resize-750.jpeg'},
                               {'name': 'versicolor',
                                'img': 'https://wiki.irises.org/pub/Spec/SpecVersicolor/iversicolor07.jpg'},
                               {'name': 'virginica',
                                'img': 'https://www.gardenia.net/storage/app/public/uploads/images/detail/xUM027N8JI22aQPImPoH3NtIMpXkm89KAIKuvTMB.jpeg'}]

            prediction_name = prediction_dict[prediction]['name']
            prediction_img = prediction_dict[prediction]['img']

            # Save prediction to database Predictions table
            Predictions.objects.create(sepal_length=sepal_length,
                                       sepal_width=sepal_width,
                                       petal_length=petal_length,
                                       petal_width=petal_width,
                                       prediction=prediction_name)

            return render(request, 'home.html', {'form': form, 'prediction': prediction,
                                                 'prediction_name': prediction_name,
                                                 'prediction_img': prediction_img})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = ModelForm()

    return render(request, 'home.html', {'form': form})
