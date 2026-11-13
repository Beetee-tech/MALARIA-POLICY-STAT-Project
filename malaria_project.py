# Import pandas to work with data tables
import pandas as pd
# Import numpy for numerical calculations (we will need this for Part C)
import numpy as np
# Import scipy.stats for probability distributions and statistical tests
from scipy import stats
# Import matplotlib to draw and save charts
import matplotlib.pyplot as plt

# Load the 600 survey households from the CSV file into a table named df
df = pd.read_csv('malaria_programme_data.csv')

# Check for any missing values across the entire dataset (0 means no missing data)
print("Total missing values:", df.isna().sum().sum())

# --- PART A, TASK A1: SUMMARY STATISTICS ---

# Create a list of the four continuous/count columns we need to summarize
cols = ['haemoglobin', 'clinic_visits', 'days_malaria_free', 'distance_km']

# Calculate the sample mean 
means = df[cols].mean()

# Calculate the median 
medians = df[cols].median()

# Calculate the sample variance using n - 1 (ddof=1) for each of the four columns
variances = df[cols].var(ddof=1)

# Calculate the sample standard deviation using n - 1 (ddof=1) for each of the four columns
std_devs = df[cols].std(ddof=1)

# Combine all four summary metrics into one clean, easy-to-read table
summary_a1 = pd.DataFrame({
    'Mean': means,
    'Median': medians,
    'Variance (n-1)': variances,
    'Std Dev (n-1)': std_devs
})

# Print a section header so the terminal output is easy to read
print("\n--- TASK A1: SUMMARY STATISTICS ---")

# Print the summary table rounded to 3 decimal places
print(summary_a1.round(3))

# Find the most frequent value (mode) for clinic_visits; [0] selects the first value
clinic_mode = df['clinic_visits'].mode()[0]

# Print the mode for clinic_visits
print("\nMode for clinic_visits:", clinic_mode)


#------------------------------------------------------------------

# --- PART A, TASK A2: OUTLIER ANALYSIS FOR distance_km ---

# Select the distance_km column so it is easy to reference
d = df['distance_km']

# Calculate the 25th percentile (Q1: 25% of households are closer than this)
q1 = d.quantile(0.25)

# Calculate the 75th percentile (Q3: 75% of households are closer than this)
q3 = d.quantile(0.75)

# Calculate the Interquartile Range (spread of the middle 50% of households)
iqr = q3 - q1

# Calculate the lower cutoff fence for outliers
lower_fence = q1 - 1.5 * iqr

# Calculate the upper cutoff fence for outliers
upper_fence = q3 + 1.5 * iqr

# Print a section header for Task A2
print("\n--- TASK A2: OUTLIER ANALYSIS (distance_km) ---")

# Print the calculated lower and upper outlier fences
print(f"Lower Fence: {lower_fence:.2f} km | Upper Fence: {upper_fence:.2f} km")

# Filter the dataset to find all rows falling outside the fences
outliers = df[(d < lower_fence) | (d > upper_fence)]

# Print how many households were flagged by the 1.5 x IQR rule
print("Total households flagged by 1.5 x IQR rule:", len(outliers))

# Sort the dataset from largest to smallest distance and show the top 6 rows
print("\nTop 6 largest distances in the dataset:")
print(df[['household_id', 'lga', 'distance_km']].sort_values('distance_km', ascending=False).head(6))

# Calculate and print Mean, Median, and Skewness BEFORE removing extreme values
print("\nBEFORE removal -> Mean:", round(d.mean(), 3), "| Median:", round(d.median(), 3), "| Skewness:", round(d.skew(), 3))

# Create a cleaned version of distance_km by keeping only realistic distances (30 km or less)
d_clean = d[d <= 30]

# Calculate and print Mean, Median, and Skewness AFTER removing extreme values
print("AFTER removal  -> Mean:", round(d_clean.mean(), 3), "| Median:", round(d_clean.median(), 3), "| Skewness:", round(d_clean.skew(), 3))




# PART A, TASK A3: SKEWNESS OF days_malaria_free 

# Compute the sample skewness of the days_malaria_free column
days_skew = df['days_malaria_free'].skew()

# Print a section header for Task A3
print("\n--- TASK A3: SKEWNESS OF days_malaria_free ---")

# Print the skewness value rounded to 3 decimal places
print("Skewness of days_malaria_free:", round(days_skew, 3))



# ==============================================================================
# PART B: PROBABILITY DISTRIBUTIONS
# ==============================================================================

# TASK B1: BINOMIAL DISTRIBUTION (uses_net) 
print("\n--- TASK B1: BINOMIAL DISTRIBUTION ---")

# Calculate the observed proportion (p) of households that sleep under a bed net
p_net = (df['uses_net'] == 'Yes').mean()
print("Observed net-use proportion (p):", round(p_net, 4))

# Calculate P(X >= 15) out of n=20 households using survival function P(X > 14)
prob_at_least_15 = stats.binom.sf(14, n=20, p=p_net)
print("Probability that at least 15 of 20 households use a net:", round(prob_at_least_15, 4))


# TASK B2: POISSON DISTRIBUTION (clinic_visits) 
print("\n--- TASK B2: POISSON DISTRIBUTION ---")

# Estimate the average monthly clinic visits per household (lambda)
lam = df['clinic_visits'].mean()

# Calculate the sample variance (n - 1) to check if Mean is close to Variance
visits_var = df['clinic_visits'].var(ddof=1)
print(f"Clinic visits Mean: {lam:.3f} | Variance: {visits_var:.3f}")

# Calculate probability of more than 3 clinic visits in a month: P(X > 3)
prob_more_than_3 = stats.poisson.sf(3, mu=lam)
print("Probability of more than 3 clinic visits:", round(prob_more_than_3, 4))

# Calculate probability of exactly 0 clinic visits in a month: P(X = 0)
prob_zero_visits = stats.poisson.pmf(0, mu=lam)
print("Probability of 0 clinic visits:", round(prob_zero_visits, 4))


# TASK B3: EXPONENTIAL DISTRIBUTION (days_malaria_free) 
print("\n--- TASK B3: EXPONENTIAL DISTRIBUTION ---")

# Estimate average waiting time until next malaria episode
mean_days = df['days_malaria_free'].mean()

# Calculate probability of staying malaria-free for more than 90 days: P(T > 90)
prob_over_90 = stats.expon.sf(90, scale=mean_days)
print("Probability of staying malaria-free > 90 days:", round(prob_over_90, 4))

# Calculate probability of experiencing a malaria episode within 30 days: P(T <= 30)
prob_within_30 = stats.expon.cdf(30, scale=mean_days)
print("Probability of malaria episode within 30 days:", round(prob_within_30, 4))


# TASK B4: SHAPE OF HAEMOGLOBIN 
print("\n--- TASK B4: SHAPE OF HAEMOGLOBIN ---")

# Calculate sample skewness of haemoglobin
hb_skew = df['haemoglobin'].skew()
print("Haemoglobin skewness:", round(hb_skew, 3))

# Create a figure and draw a histogram with 20 bars to inspect the shape
plt.figure(figsize=(8, 5))
plt.hist(df['haemoglobin'], bins=20, edgecolor='black', color='steelblue')

# Add title and axis labels with units
plt.title('Distribution of Household Haemoglobin Levels')
plt.xlabel('Haemoglobin (g/dL)')
plt.ylabel('Number of Households')

# Save the histogram as an image file in your project folder
plt.savefig('haemoglobin_histogram.png', bbox_inches='tight')

# Display the histogram window on screen
#plt.show()



# ==============================================================================
# PART C: HYPOTHESIS TESTS AND CONFIDENCE INTERVALS
# ==============================================================================

#  TASK C1: ONE-SAMPLE Z-TEST (haemoglobin vs 11.5 benchmark) 
# H0: mu = 11.5 g/dL (Mean haemoglobin equals the national benchmark)
# H1: mu < 11.5 g/dL (Mean haemoglobin is below the national benchmark)
# Tail: Left-tailed, because we are testing whether the mean is LOWER (<) than 11.5
# Significance level: alpha = 0.05

print("\n--- TASK C1: ONE-SAMPLE Z-TEST ---")

# Extract the haemoglobin column
hb = df['haemoglobin']

# Set the national benchmark (mu0) and known population standard deviation (sigma)
mu0 = 11.5
sigma = 1.5

# Calculate sample size (n) and sample mean (xbar)
n_hb = len(hb)
xbar = hb.mean()

# Calculate standard error: population SD divided by square root of sample size
se_z = sigma / np.sqrt(n_hb)

# Calculate the z-test statistic
z_stat = (xbar - mu0) / se_z

# Calculate the left-tailed p-value using the standard normal cumulative distribution
p_val_c1 = stats.norm.cdf(z_stat)

# Find the critical z-value for a 95% two-sided confidence interval (1.96)
z_crit = stats.norm.ppf(0.975)

# Calculate the 95% confidence interval around the sample mean
ci_c1_low = xbar - z_crit * se_z
ci_c1_high = xbar + z_crit * se_z

# Print all Task C1 results
print(f"Sample Mean: {xbar:.3f} g/dL | Standard Error: {se_z:.4f}")
print(f"Z-statistic: {z_stat:.4f} | Left-tailed p-value: {p_val_c1:.6e}")
print(f"95% Confidence Interval: [{ci_c1_low:.3f}, {ci_c1_high:.3f}]")


# TASK C2: TWO-SAMPLE T-TEST (Programme Yes vs No) 
# H0: mu_yes = mu_no (Programme and non-programme households have the same mean haemoglobin)
# H1: mu_yes != mu_no (Programme and non-programme households differ in mean haemoglobin)
# Tail: Two-tailed, because we are asking whether the two groups DIFFER (!=)
# Significance level: alpha = 0.05

print("\n--- TASK C2: TWO-SAMPLE WELCH T-TEST ---")

# Split haemoglobin values into programme ('Yes') and non-programme ('No') groups
hb_yes = df.loc[df['programme'] == 'Yes', 'haemoglobin']
hb_no = df.loc[df['programme'] == 'No', 'haemoglobin']

# Run Welch's two-sample t-test (equal_var=False does not assume equal group variances)
t_res = stats.ttest_ind(hb_yes, hb_no, equal_var=False)

# Calculate the difference in group means (Yes minus No) in real units (g/dL)
mean_diff = hb_yes.mean() - hb_no.mean()

# Extract the 95% confidence interval for the difference in means
ci_c2 = t_res.confidence_interval(confidence_level=0.95)

# Calculate Cohen's d to measure practical effect size in standard deviation units
pooled_sd = np.sqrt((hb_yes.var(ddof=1) + hb_no.var(ddof=1)) / 2)
cohen_d = mean_diff / pooled_sd

# Print all Task C2 results
print(f"Programme Mean: {hb_yes.mean():.3f} | Non-Programme Mean: {hb_no.mean():.3f}")
print(f"Difference in Means: {mean_diff:.3f} g/dL")
print(f"T-statistic: {t_res.statistic:.4f} | Degrees of Freedom: {t_res.df:.2f} | Two-tailed p-value: {t_res.pvalue:.6f}")
print(f"95% CI for Difference: [{ci_c2.low:.3f}, {ci_c2.high:.3f}]")
print(f"Cohen's d: {cohen_d:.3f}")



# TASK C3: CHI-SQUARE TEST 1 (LGA vs Bed-Net Use)
# H0: Bed-net use is independent of LGA (no relationship between LGA and using a net)
# H1: Bed-net use is related to LGA (net use differs across LGAs A, B, and C)
# Tail: Right-tailed (chi-square test statistics are always positive, testing upper tail)
# Significance level: alpha = 0.05

print("\n--- TASK C3: CHI-SQUARE TEST (LGA vs uses_net) ---")

# Create a contingency table (cross-tabulation) of counts: rows = LGA, columns = uses_net
tab_lga = pd.crosstab(df['lga'], df['uses_net'])
print("Observed Counts (LGA vs uses_net):")
print(tab_lga)

# Run the chi-square test of independence
chi2_c3, p_c3, dof_c3, expected_c3 = stats.chi2_contingency(tab_lga)

# Print test statistic, p-value, and degrees of freedom
print(f"\nChi-square statistic: {chi2_c3:.4f} | p-value: {p_c3:.6e} | df: {dof_c3}")

# Check whether all expected cell counts are at least 5 (assumption check)
print("All expected counts >= 5:", (expected_c3 >= 5).all())

# Print expected counts rounded to 1 decimal place for verification
print("\nExpected Counts:")
print(pd.DataFrame(expected_c3, index=tab_lga.index, columns=tab_lga.columns).round(1))

# Calculate row percentages (share of Yes/No within each LGA) to identify the problem LGA
print("\nRow Percentages (Net Use Share within each LGA):")
print(pd.crosstab(df['lga'], df['uses_net'], normalize='index').round(3))


# TASK C4: CHI-SQUARE TEST 2 (Programme vs Bed-Net Use)
# H0: Bed-net use is independent of taking part in the programme
# H1: Bed-net use is related to taking part in the programme
# Tail: Right-tailed (chi-square goodness-of-fit/independence upper tail)
# Significance level: alpha = 0.05

print("\n--- TASK C4: CHI-SQUARE TEST (programme vs uses_net) ---")

# Create a 2x2 contingency table: rows = programme, columns = uses_net
tab_prog = pd.crosstab(df['programme'], df['uses_net'])
print("Observed Counts (programme vs uses_net):")
print(tab_prog)

# Run chi-square test WITH Yates' continuity correction (default in SciPy for 2x2 tables)
chi2_yates, p_yates, dof_yates, _ = stats.chi2_contingency(tab_prog, correction=True)
print(f"\nWith Yates correction    -> Chi2: {chi2_yates:.4f} | p-value: {p_yates:.6e} | df: {dof_yates}")

# Run chi-square test WITHOUT Yates' continuity correction (standard textbook formula)
chi2_noyates, p_noyates, dof_noyates, _ = stats.chi2_contingency(tab_prog, correction=False)
print(f"Without Yates correction -> Chi2: {chi2_noyates:.4f} | p-value: {p_noyates:.6e} | df: {dof_noyates}")


#  TASK C5: 95% CONFIDENCE INTERVAL FOR NET-USE PROPORTION ---
print("\n--- TASK C5: 95% CI FOR OVERALL NET-USE PROPORTION ---")

# Create a boolean series where True = household uses a net
net_used = df['uses_net'] == 'Yes'

# Calculate sample proportion (p_hat) and sample size (n)
p_hat = net_used.mean()
n_net = len(net_used)

# Calculate the standard error of the sample proportion
se_prop = np.sqrt(p_hat * (1 - p_hat) / n_net)

# Calculate the lower and upper bounds of the 95% confidence interval
z_crit_prop = stats.norm.ppf(0.975)
ci_prop_low = p_hat - z_crit_prop * se_prop
ci_prop_high = p_hat + z_crit_prop * se_prop

# Print sample proportion, 95% CI, and whether the 80% (0.80) state target is inside
print(f"Sample Proportion (p_hat): {p_hat:.4f} ({p_hat*100:.1f}%)")
print(f"95% Confidence Interval: [{ci_prop_low:.4f}, {ci_prop_high:.4f}] ({ci_prop_low*100:.1f}% to {ci_prop_high*100:.1f}%)")
print("Is the 80% (0.80) state target inside the interval?:", ci_prop_low <= 0.80 <= ci_prop_high)


# --- TASK C6: TYPE I AND TYPE II ERRORS (Using Task C2: Two-Sample T-Test on Haemoglobin) ---
# Test chosen: Task C2 (comparing mean haemoglobin between programme and non-programme households).
# H0: Programme and non-programme households have the same mean haemoglobin (no real health benefit).
# H1: Programme and non-programme households differ in mean haemoglobin.
#
# Type I Error (False Positive): Rejecting H0 when it is actually true.
# Meaning for the Commissioner: Concluding that the malaria programme improves haemoglobin levels
# when in reality it produces no real difference, leading the Ministry to waste public funds
# scaling up an ineffective health intervention across the entire state.
#
# Type II Error (False Negative): Failing to reject H0 when H1 is actually true.
# Meaning for the Commissioner: Concluding that the programme has no effect on haemoglobin when
# it genuinely helps households, leading the Ministry to cancel or delay a beneficial programme
# and leaving families across the state unprotected against malaria-related anaemia.