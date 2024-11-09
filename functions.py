import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency

def basic_hist(ax, data, name, color):
    '''Creates a very basic histogram'''
    ax.hist(data.dropna(), bins=30, color=color, edgecolor='black')
    ax.set_title('Histogram of '+name, fontsize=16)
    ax.set_xlabel(name, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)

def advanced_hist(data, name, n_bins):
    '''Creates a histogram with kde, mean and median'''
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=n_bins, kde=True, color='skyblue', alpha=0.7)

    mean = data.mean()
    median = data.median()

    plt.axvline(mean, color='red', linestyle='--', label='Mean '+name)
    plt.axvline(median, color='green', linestyle='--', label='Median '+name)

    plt.title('Distribution of '+name, fontsize=16)
    plt.xlabel(name, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()

    plt.show()

def basic_bar(ax, data, name):
    '''Creates a barplot on a given axis'''
    value_counts = data.value_counts()
    sns.countplot(x=data, ax=ax, hue=data, legend=False)
    ax.set_title(name, fontsize=16)
    ax.set_ylabel('Count', fontsize=14)

def run_ttest(group_a, group_b, name, alpha):
    '''Runs a Welsch t-test between two given arrays'''
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    
    t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)

    print(f"Mean {name} for No Stroke: {mean_a:.2f}")
    print(f"Mean {name} for Stroke: {mean_b:.2f}")
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.6f}")
    if p_value < alpha:
        print(f"We reject H0. The difference between groups is statistically significant.")
    else:
        print(f"We can't reject H0. The difference between groups is not statistically significant.")

def test_category(X, y, a):
    '''Runs a chi2 test between two categories'''
    contingency_table = pd.crosstab(X, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"H0: Feature is statistically insignificant")
    print(f"H1: Feature is statistically significant\n\n")
    print(f"\nChi2 Statistic: {chi2}")
    print(f"P-value: {p}")

    if p < a:
        print("We reject the null hypothesis, the feature is statistically significant.")
    else:
        print("We can't reject the null hypothesis - the feature is statistically insignificant.")

def print_crosstab(X, y):
    '''Return a crosstab of two different categories'''
    cross_tab = pd.crosstab(X, y, margins=True, normalize='index')*100
    cross_tab = cross_tab.round(1)
    return(cross_tab)

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores