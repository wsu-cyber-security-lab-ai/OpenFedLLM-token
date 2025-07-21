import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom organization order
custom_org_order = [
    "FirstCompany",
    "SecondCompany",
    "ThirdOrg",
    "FourthInc",
    "FifthLLC",
    "SixthGroup",
    "SeventhEnterprises",
    "EighthPartners",
    "NinthCorp",
    "TenthSolutions"
]

org_map = {name: f"Organization {i+1}" for i, name in enumerate(custom_org_order)}

# Load data (all situations, no filter)
df = pd.read_csv("org_summery_wrong.csv")

# Compute F1 for each row before grouping
for datatype in ['Overall', 'Email', 'SSN', 'Phone']:
    p_col = f"{datatype} Precision"
    r_col = f"{datatype} Recall"
    f1_col = f"{datatype} F1"
    # Set F1 to 0 if both precision and recall are 0, else use harmonic mean
    df[f1_col] = np.where(
        (df[p_col] == 0) & (df[r_col] == 0),
        0,
        2 * df[p_col] * df[r_col] / (df[p_col] + df[r_col])
    )


metrics_extended = [
    "Overall Precision", "Overall Recall", "Overall F1",
    "Email Precision", "Email Recall", "Email F1",
    "SSN Precision", "SSN Recall", "SSN F1",
    "Phone Precision", "Phone Recall", "Phone F1"
]

# Group by organization and average all metrics
df_grouped = df.groupby("Organization")[metrics_extended].mean().reset_index()

# Apply mapping and ordering
df_grouped["OrganizationRenamed"] = df_grouped["Organization"].map(org_map)
cat_type = pd.CategoricalDtype(
    categories=[f"Organization {i+1}" for i in range(10)],
    ordered=True
)
df_grouped["OrganizationRenamed"] = df_grouped["OrganizationRenamed"].astype(cat_type)
df_grouped = df_grouped.sort_values("OrganizationRenamed")

# Prepare DataFrame for heatmap
heatmap_df = df_grouped.set_index("OrganizationRenamed")[metrics_extended] * 100

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={'label': 'Percentage (%)'}
)
plt.title("Precision, Recall, and F1-score by Organization and Data Type (Averaged)", fontsize=16)
plt.xlabel("Metric", fontsize=13)
plt.ylabel("Organization", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("precision_recall_f1_heatmap_customorder_avg.png", dpi=300)
# plt.show()
