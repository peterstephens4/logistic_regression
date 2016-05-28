#  Peter Stephens
#  5/26/2016

#  Logistic Regression


import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import subprocess
import math

#  Define the logistic function
def logistic_function(coeff, ficoScore, loanAmount):
    p = 1.0/(1.0 + math.exp(coeff['Interest.Intercept'] + coeff['FICO.Score']*ficoScore + coeff['Amount.Requested']*loanAmount))
    return(p)
  
#  Clean the directory of old png files
proc = subprocess.check_call("rm -rf *.png",  shell=True)

#  Read in Lending Club Data form git hub repository
loansData = pd.read_csv('loansData_clean.csv')

#  Clean the Data
loansData.dropna(inplace=True)
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x.rstrip('%')))
loansData['Loan.Length']   = loansData['Loan.Length'].map(lambda x: int(x.rstrip('months')))

#  Create a seperate FICO score collumn of data from the FICO range
loansData['FICO.Score']    = loansData['FICO.Range'].map(lambda x: int(x.split('-')[0]))

#  Return all of the FICO Score and Loan Ammount Information
ficoScore = loansData['FICO.Score']
loanAmount = loansData['Amount.Requested']

#  We only care about interest rates less than 12%
loansData['Interest.IR_TF'] = loansData['Interest.Rate'] < 12.0
IR_TF= loansData['Interest.IR_TF'].map(lambda x: int(x))

#  Create intercept column and return it's values
loansData['Interest.Intercept'] = 1.0
intercept = loansData['Interest.Intercept']

# Create list of ind var col names
ind_vars = ['FICO.Score', 'Amount.Requested', 'Interest.Intercept'] 

#  Define the logistic regression
logit = sm.Logit(loansData['Interest.IR_TF'], loansData[ind_vars])

#  Fit the model
result = logit.fit()

#  Return the  fitted coefficient and print the results summary information
coeff = result.params
print(result.summary())

##################################################################################
#  What is the probability of getting a loan from the Lending Club for $10,000   #
#  at an interest rate < 12% with a FICO score of 720?                           #
##################################################################################
f = 720
la = 10000
p = logistic_function(coeff, f, la) * 100.0
print("The probability of getting a loan from the Lending Club for $10,000 at an interest rate less than 12% with a FICO score of 720 is {0:.2f}%." .format(p))

#  Plot the data to visualize the perdiction
tick = [(720, 0), (720, 1.2)]
plt.figure()
for x in list(range(550, 951)):
    y = logistic_function(coeff, x, la)
    lf_plot = plt.scatter(x, y, color='blue')
fico_plot= plt.scatter(x = ficoScore, y = IR_TF, color='red')
fico720_plot =  plt.axvline(x=720, ymin=-0.2, ymax=1.2, color='black')
plt.savefig("Logistic_Function_Plot.png")



