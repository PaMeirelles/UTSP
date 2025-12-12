import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('old_runs/final_benchmark_results.csv')

# --- Initial Data Prep ---

# Define the success column (True for 'SUCCESS', False otherwise)
df['is_success'] = df['status'] == 'SUCCESS'

# Define the columns for the first plot
grouping_cols = ['instance_type']
titles = [
    'Success Rate by Instance Type'
]
xlabels = ['Instance Type']
file_names = ['success_rate_instance_type.png']

# --- General Calculation/Plotting Functions ---

def calculate_success_percentage(data, group_col):
    # Ensure the grouping column is treated as a string
    data[group_col] = data[group_col].astype(str)
    success_rate = data.groupby(group_col)['is_success'].mean() * 100
    success_df = success_rate.reset_index(name='Success_Percentage')
    success_df = success_df.sort_values(by='Success_Percentage', ascending=False)
    return success_df

def plot_success_rate(data_df, group_col, title, xlabel, file_name):
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=group_col,
        y='Success_Percentage',
        data=data_df,
        palette='viridis',
        order=data_df[group_col].astype(str).tolist()
    )
    max_height = data_df['Success_Percentage'].max()
    plt.ylim(0, 100 + max_height * 0.08)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Success Percentage (%)')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


# --- Execution of Success Rate Plots ---

# 1. Success rate per 'instance_type'
instance_type_df = calculate_success_percentage(df.copy(), grouping_cols[0])
plot_success_rate(instance_type_df, grouping_cols[0], titles[0], xlabels[0], file_names[0])
print(f"Generated plot: {file_names[0]}")

# 2. Success rate for 'ATT' by 'topk'
att_df = df[df['instance_type'] == 'ATT'].copy()
if not att_df.empty:
    att_topk_df = calculate_success_percentage(att_df.copy(), 'topk')
    plot_success_rate(att_topk_df, 'topk', 'ATT Success Rate by Top K', 'Top K Value',
                      'old_runs/att_success_rate_topk.png')
    print(f"Generated plot: att_success_rate_topk.png")


# --- NEW Execution for Gap Analysis using 'gap_percent' ---

# 3. Mean Gap Percentage by 'topk' and 'instance_type'

gap_grouping_cols = ['instance_type', 'topk']

# Calculate the mean of the 'gap_percent' column grouped by instance_type and topk
gap_df_grouped = df.groupby(gap_grouping_cols)['gap_percent'].mean().reset_index(name='Mean_Gap_Percent')

# Convert 'topk' to string for categorical plotting
gap_df_grouped['topk'] = gap_df_grouped['topk'].astype(str)

# Sort for better visualization: by instance type, then by topk
gap_df_grouped = gap_df_grouped.sort_values(by=['instance_type', 'topk'], ascending=[True, True])

# Plotting the mean gap using seaborn's catplot for a grouped bar chart
g = sns.catplot(
    data=gap_df_grouped,
    kind="bar",
    x="topk",
    y="Mean_Gap_Percent",
    hue="instance_type",
    palette="dark",
    height=6,
    aspect=1.5
)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Mean Gap Percentage by Top K and Instance Type')
g.set_axis_labels("Top K Value", "Mean Gap Percentage (%)")

# Add a horizontal line at 0 for reference
g.refline(y=0, color='grey', linewidth=0.8)

# Add value labels on top of the bars
for ax in g.axes.flat:
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:,.2f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center',
                    va='center',
                    xytext=(0, 5 if height >= 0 else -10),
                    textcoords='offset points',
                    fontsize=8)

g.savefig('mean_gap_percent_topk_instance_type.png')
plt.close(g.fig)
print("Generated plot: mean_gap_percent_topk_instance_type.png")