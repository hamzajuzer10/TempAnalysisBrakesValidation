import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

# suppress scientific float notation
np.set_printoptions(precision=5, suppress=True)

# Update path depending where raw data is stored
PATH = 'C:\Users\hamzajuzer\PycharmProjects\TempAnalysisBrakesValidation\A_temp.csv'

# import data (Each top level customers has unique row)
data = pd.read_csv(PATH)

# check data makes sense
print(data.shape)
print(data.head())

# clean data if necessary
# TODO


# One - hot encode categorical data
# inc. Sector, Region, Food Technical Dept, Kitchen Skill Level
data = pd.concat([data, pd.get_dummies(data['Sector'], prefix='Sector')], axis=1)
# data = pd.concat([data, pd.get_dummies(data['Region'], prefix='Region')], axis=1)
data = pd.concat([data, pd.get_dummies(data['Do they have a food technical department'], prefix='FoodTech')], axis=1)
data = pd.concat([data, pd.get_dummies(data['General Unit Kitchen Skill level'], prefix='KSkillLvl')], axis=1)

# Set all 0s in categories which are null to null
data.loc[data['Sector'].isnull(), data.columns.str.startswith("Sector_")] = np.nan
# data.loc[data['Region'].isnull(), data.columns.str.startswith("Region_")] = np.nan
data.loc[data['Do they have a food technical department'].isnull(), data.columns.str.startswith("FoodTech_")] = np.nan
data.loc[data['General Unit Kitchen Skill level'].isnull(), data.columns.str.startswith("KSkillLvl_")] = np.nan

# convert ordered categorical data
# inc. Size of proc team, Size of food dev team


def if_team(col):

    if col == '0-5':
        return 2.5
    if col == '5-10':
        return 7.5
    # for some stupid reason, csv always auto-formats 0-5 to 05-Oct
    if col == '05-Oct':
        return 7.5
    if col == '10+':
        return 15


data['pTeamSize'] = data['Size of Food development team'].apply(if_team)
data['fdTeamSize'] = data['Size of procurement team'].apply(if_team)

# drop original categorical columns
data.drop(['Sector'], axis=1, inplace=True)
# data.drop(['Region'], axis=1, inplace=True)
data.drop(['Do they have a food technical department'], axis=1, inplace=True)
data.drop(['General Unit Kitchen Skill level'], axis=1, inplace=True)
data.drop(['Size of Food development team'], axis=1, inplace=True)
data.drop(['Size of procurement team'], axis=1, inplace=True)

# drop top level customer column
data.drop(['Top Level Customer'], axis=1, inplace=True)

# Cleanse all data with NaNs - significant since behaviour data only accounts for 30% of data-set
# Use mean imputation
data = data.fillna(data.mean())

# TODO - Look into other imputation methods (or come up with reasonable
# TODO - estimates for behaviour data based on TLC revenue)

# Normalize the data, don't actually need to do it as all data lie within 0-1 but just to be safe...
data = (data-data.min())/(data.max()-data.min())

# Run agglomerative clustering.....

# generate the linkage matrix
# May need to play around with other linkage methods such as single, etc and distance metrics
Z = linkage(data, 'ward')


# Check the Cophenetic Correlation Coefficient
# This compares (correlates) the actual pairwise distances
# of all samples to those implied by the
# hierarchical clustering.
# The closer the value is to 1,
# the better the clustering preserves the original distances,

c, coph_dists = cophenet(Z, pdist(data))

print c

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')

# set cut-off to 50
max_distance = 50  # max_d as in max_distance

dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    # show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12,
    show_contracted=True,  # to get a distribution impression in truncated branches

)
plt.show()

# TODO - Show most frequent features at every leaf node
