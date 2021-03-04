from django import forms


class ModelForm(forms.Form):
    n_arms = forms.IntegerField(
        label='Number of Arms      ')
    n_bootstraps = forms.IntegerField(
        label='Bootstrap Resamples')
    n_months  = forms.IntegerField(
        label='Months to Simulate ')
    n_invoices_month  = forms.IntegerField(
        label='Invoices per Month  ')

