from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd

# Create your views here.

def index(request):
    return render(request,"home.html")

def predict(request):
    #Load the model
    regressor = pickle.load(open("bostonModel.pkl","rb"))
    scaler = pickle.load(open("scaling.pkl","rb"))

    if request.method == 'POST':

        #Getting all the form values
        temp={}
        temp['CRIM'] = request.POST.get('CRIM')
        temp['ZN'] = request.POST.get('ZN')
        temp['INDUS'] = request.POST.get('INDUS')
        temp['CHAS'] = request.POST.get('CHAS')
        temp['NOX'] = request.POST.get('NOX')
        temp['RM'] = request.POST.get('RM')
        temp['Age'] = request.POST.get('Age')
        temp['DIS'] = request.POST.get('DIS')
        temp['RAD'] = request.POST.get('RAD')
        temp['TAX'] = request.POST.get('TAX')
        temp['PTRATIO'] = request.POST.get('PTRATIO')
        temp['B'] = request.POST.get('B')
        temp['LSTAT'] = request.POST.get('LSTAT')

        #Transforming the form_data into dataframe
        testdata = pd.DataFrame({'x':temp}).astype(float)

        #reshaping the array and using StandarScaler to transform the data
        final_input=scaler.transform(np.array(testdata).reshape(1,-1))

        #predicting the result
        result = regressor.predict(final_input)[0]
        
        #passing the result as context to result page
        result = round(result)
        context={'prediction':result}
    return render(request,"result.html",context)