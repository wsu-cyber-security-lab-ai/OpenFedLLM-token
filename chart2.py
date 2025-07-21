import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom order: the exact desired order of your raw organization names
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

# Map each original name to 'Organization 1' ... 'Organization 10'
org_map = {name: f"Organization {i+1}" for i, name in enumerate(custom_org_order)}

# Load data and filter
df = pd.read_csv("org_summery_correct_code_correct_org.csv")
df_situation = df[df["situation"] == "correct_code_correct_org"].copy()
df_situation["OrganizationRenamed"] = df_situation["Organization"].map(org_map)

# (Re-)order your DataFrame with a categorical type for the desired order
cat_type = pd.CategoricalDtype(
    categories=[f"Organization {i+1}" for i in range(10)],
    ordered=True
)
df_situation["OrganizationRenamed"] = df_situation["OrganizationRenamed"].astype(cat_type)
df_situation = df_situation.sort_values("OrganizationRenamed")

# Example: Use in your heatmap setup
# ---------------------------------------------------------
metrics = [
    "Overall Precision", "Overall Recall", "Email Precision", "Email Recall",
    "SSN Precision", "SSN Recall", "Phone Precision", "Phone Recall"
]

for datatype in ['Overall', 'Email', 'SSN', 'Phone']:
    p_col = f"{datatype} Precision"
    r_col = f"{datatype} Recall"
    f1_col = f"{datatype} F1"
    df_situation[f1_col] = 2 * df_situation[p_col] * df_situation[r_col] / (
        df_situation[p_col] + df_situation[r_col]
    )

metrics_extended = [
    "Overall Precision", "Overall Recall", "Overall F1",
    "Email Precision", "Email Recall", "Email F1",
    "SSN Precision", "SSN Recall", "SSN F1",
    "Phone Precision", "Phone Recall", "Phone F1"
]

# Now build the heatmap DataFrame in the chosen order
heatmap_df = (
    df_situation
    .set_index("OrganizationRenamed")[metrics_extended]
    .copy()
    * 100
)

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
plt.title("Precision, Recall, and F1-score by Organization and Data Type", fontsize=16)
plt.xlabel("Metric", fontsize=13)
plt.ylabel("Organization", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("precision_recall_f1_heatmap_customorder.png", dpi=300)
# plt.show()
