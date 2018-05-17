import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# load the data amd splitting to X,Y
voice = pd.read_csv("./voice.csv")
Y = voice['label']
X = voice.drop(['label'], axis=1)

# checking distribution between male and female examples
print (Y.value_counts())

# converting voice from wide to long format
voice_normalized = pd.DataFrame(MinMaxScaler().fit_transform(X))
voice_normalized = pd.concat([voice_normalized, Y], axis=1)
voice_normalized.columns = voice.columns
voice_long = pd.melt(voice_normalized, id_vars=voice_normalized.drop(['meanfreq','meanfun','meandom'], axis=1), 
                    value_vars=['meanfreq','meanfun','meandom'], 
                    var_name='typeofmean', value_name='valueofmean')

# facet grid scatterplot on mean features
voice_reduced = voice[['meanfreq','meanfun','meandom','label']]
sns.pairplot(voice_reduced, 
            hue='label',
            plot_kws=dict(s=15, alpha=0.5))
plt.show()

# facet grid boxplot on mean features
plt.gcf().clear()
boxplot = sns.factorplot(x='label', 
                        y='valueofmean', 
                        hue='label', 
                        col='typeofmean', 
                        data=voice_long, 
                        kind='box', 
                        legend=True)
plt.show()