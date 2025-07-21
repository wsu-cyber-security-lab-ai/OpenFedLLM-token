import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv("org_summery.csv")

# Filter for correct situation and assign organization names
df_situation = df[df["situation"] == "correct_code_correct_org"].copy()
metrics = ["Overall Precision", "Overall Recall", "Email Precision", "Email Recall"]
org_names = sorted(df_situation["Organization"].unique())
org_map = {orig: f"Organization {i + 1}" for i, orig in enumerate(org_names)}
df_situation["OrganizationRenamed"] = df_situation["Organization"].map(org_map)

# Reshape for plotting
df_melted = df_situation.melt(
    id_vars="OrganizationRenamed",
    value_vars=metrics,
    var_name="Metric",
    value_name="Value"
)
df_melted["Percent"] = df_melted["Value"] * 100

plt.figure(figsize=(20, 8))
ax = sns.barplot(
    data=df_melted,
    x="OrganizationRenamed",
    y="Percent",
    hue="Metric",
    width=0.7
)

ax.set_ylim(0, 107)
# Set axis labels using matplotlib or ax.set()[1][2][3]
ax.set(xlabel="Organizations", ylabel="Percentage %", title="Metrics by Organization for Situation: Correct Code & Correct Org")

plt.xticks(rotation=45, ha="right")
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Annotate directly above each bar, no x-offset, small font
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=5,
            color="black"
        )

plt.savefig("correct_code_correct_org_barplot_nooffset.png", dpi=300)
# plt.show()
