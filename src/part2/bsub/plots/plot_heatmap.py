import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the data you provided is in a CSV file and the delimiter is a tab
csv_file_path = '../output/omp_teams_thread_limit_19952410.err'  # Update this path to your CSV file

# Read the CSV file
data_from_csv = pd.read_csv(csv_file_path)
column_names = data_from_csv.columns.tolist()
# print(column_names)

# Pivot the DataFrame for the heatmap
heatmap_data = data_from_csv.pivot(index='num_teams', columns='thread_limit', values='iterations_per_sec')

# Plotting
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)

# Labels and title
plt.xlabel('Thread Limit')
plt.ylabel('Number of Teams')
plt.title('Heatmap of Iterations per Second')
# plt.show()
plt.savefig('../../../../plots/heatmap_num_teams_thread_limit.png', dpi=300, bbox_inches='tight')

# Show plot
