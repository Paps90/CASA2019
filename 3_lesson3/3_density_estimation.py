import numpy
import pandas
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pandas.read_csv('C:/Udemy/SKLEARN-Python/004_visits_per_day.csv', index_col=False, header=0);
X_plot =  numpy.linspace(-2, 2, 1000)[:, numpy.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(df.values)
log_dens = kde.score_samples(X_plot)

#plt.hist(df.values, bins=numpy.linspace(0, 1, 20), fc='#AAAAFF', normed=True)

plt.hist(df.values, bins=numpy.linspace(0, 1, 10), fc='#AAAAFF', normed=True)


plt.plot(X_plot, numpy.exp(log_dens))
plt.show()


#########################################################################################


df = pandas.read_csv('C:/Udemy/SKLEARN-Python/004_visits_per_day.csv', index_col=False, header=0);
X_plot =  numpy.linspace(-20, 1, 1000)[:, numpy.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=1.4).fit(numpy.log(df.values))
# here I needed to change the bandwidth value to 1.4
log_dens = kde.score_samples(X_plot)

plt.hist(numpy.log(df.values), bins=numpy.linspace(-20, 1, 10), fc='#AAAAFF', normed=True)


plt.plot(X_plot, numpy.exp(log_dens))
plt.show()

