from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column



from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column

STATES = (
    ('', 'Choose...'),
    ('MG', 'Minas Gerais'),
    ('SP', 'Sao Paulo'),
    ('RJ', 'Rio de Janeiro')
)

class AddressForm(forms.Form):
    email = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Email'}))
    password = forms.CharField(widget=forms.PasswordInput())
    address_1 = forms.CharField(
        label='Address',
        widget=forms.TextInput(attrs={'placeholder': '1234 Main St'})
    )
    address_2 = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Apartment, studio, or floor'})
    )
    city = forms.CharField()
    state = forms.ChoiceField(choices=STATES)
    zip_code = forms.CharField(label='Zip')
    check_me_out = forms.BooleanField(required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Row(
                Column('email', css_class='form-group col-md-6 mb-0'),
                Column('password', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            'address_1',
            'address_2',
            Row(
                Column('city', css_class='form-group col-md-6 mb-0'),
                Column('state', css_class='form-group col-md-4 mb-0'),
                Column('zip_code', css_class='form-group col-md-2 mb-0'),
                css_class='form-row'
            ),
            'check_me_out',
            Submit('submit', 'Sign in')
        )


class ModelForm(forms.Form):

    n_arms = forms.IntegerField(
        label='Number of Arms      ')
    arm_1_name = forms.CharField(label='Arm 1 Name')
    arm_2_name = forms.CharField(label='Arm 2 Name')
    arm_3_name = forms.CharField(label='Arm 3 Name')
    arm_4_name = forms.CharField(label='Arm 4 Name')
    arm_1_collection_rate = forms.FloatField(label='Arm 1 Expected Collection Rate')
    arm_2_collection_rate = forms.FloatField(label='Arm 2 Expected Collection Rate')
    arm_3_collection_rate = forms.FloatField(label='Arm 3 Expected Collection Rate')
    arm_4_collection_rate = forms.FloatField(label='Arm 4 Expected Collection Rate')
    n_bootstraps = forms.IntegerField(
        label='Bootstrap Resamples')
    n_months  = forms.IntegerField(
        label='Number of months of to Run Experiment (at least one Dunning Cycle is Needed)')
    n_invoices_month  = forms.IntegerField(
        label='Average Number of Invoices per Month  ')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Row(
                Column('n_arms', css_class='form-group col-md-6 mb-0'),
                Column('n_bootstraps', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            Row(
                Column('arm_1_name', css_class='form-group col-md-6 mb-0'),
                Column('arm_1_collection_rate', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            Row(
                Column('arm_2_name', css_class='form-group col-md-6 mb-0'),
                Column('arm_2_collection_rate', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            Row(
                Column('arm_3_name', css_class='form-group col-md-6 mb-0'),
                Column('arm_3_collection_rate', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            Row(
                Column('arm_4_name', css_class='form-group col-md-6 mb-0'),
                Column('arm_4_collection_rate', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            'n_months',
            'n_invoices_month',
            Submit('submit', 'Submit')
        )



