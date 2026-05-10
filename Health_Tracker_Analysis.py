import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Project: Synthetic Health Data Analysis

# Creating sample data set using np.random.normal
mean = 70
std_dev = 10
size = 1000 # size of the sample of people i am going to take as my patients
sample = np.random.normal(mean, std_dev, size) # not using randint() as that is a uniform distribution
sample = np.round(sample).astype(int) # As in real life we rarely see heart beats to a decimal value
print(f'The mean of this data set is {np.mean(sample)}') # to check if the sample mean is close to the actual value
print(f'The standard deviation of this data set is {np.std(sample)}') # to check if the sample standard deviation is close to the actual value
print(f'The median of this data set is {np.median(sample)}')

# Displaying the data as a graph
sns.displot(sample, kde = True) # the line shows the probability trend
plt.title("Heart Rate Distribution")
plt.xlabel("Heart Rate")
plt.ylabel("Frequency")
plt.show()

# Smart Filter
# Finding out all the outliers
lowerlimit = mean - 2 * std_dev
upperlimit = mean + 2 * std_dev

unhealthy_patients = sample[(sample < lowerlimit) | (sample > upperlimit)]
print(f'The number of unhealthy is {len(unhealthy_patients)}')
print(unhealthy_patients)
# I am using 2 standard deviations to find patients which are at risk
outlier_indexes = np.where((sample < lowerlimit) | (sample > upperlimit))[0]
print(f'The location of the unhealthy patients are {outlier_indexes}')
# Normalizing the data
# Prepare the data for a future "AI Model."
# Find the boundaries
lowest = np.min(sample)
highest = np.max(sample)
normalized_data = (sample - lowest) / (highest - lowest)

# Proving it worked
print(f"Lowest value is now: {np.min(normalized_data)}") # Should be 0.0
print(f"Highest value is now: {np.max(normalized_data)}") # Should be 1.0