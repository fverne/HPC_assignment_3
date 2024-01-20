
import matplotlib.pyplot as plt
import numpy as np

grid_sizes = [100, 200, 500, 1000, 2000, 5000]

# Data for mnk_offload
iterations_mnk_offload_1_team = [20509.859,
47988.204,
69776.498,
77094.942,
49419.468,
48108.123,]
iterations_mnk_offload_2_teams = [ 24663.041,
 70951.673,
123452.899,
138657.720,
 97206.199,
 95565.830,]
iterations_mnk_offload_4_teams = [  27791.078,
111833.676,
273382.440,
359578.743,
287179.282,
250296.628,]
iterations_mnk_offload_6_teams = [ 27884.633,
118479.701,
313213.710,
438413.182,
339283.093,
328210.668,]
iterations_mnk_offload_8_teams = [  27786.375,
120088.482,
353561.108,
535264.358,
443499.429,
469930.469,]
iterations_mnk_offload_10_teams = [  26807.369,
119806.260,
304115.718,
405887.760,
502928.968,
519601.507,]
iterations_mnk_offload_12_teams = [ 27956.577,
118994.256,
340815.432,
539576.163,
611068.591,
641051.820,]
iterations_mnk_offload_14_teams = [  27905.328,
118498.032,
354076.302,
560368.649,
685407.428,
718695.406,]
iterations_mnk_offload_16_teams = [ 27740.578,
118838.207,
366376.049,
587272.729,
757389.036,
799219.300,
]

iterations_mnk_offload_18_teams = [ 22967.659,
111806.748,
364745.917,
586528.717,
755185.470,
798390.042,
]


# Data for lib
iterations_mkn_offload_1_team = [ 1699.584,
 3246.930,
 9200.600,
16915.478,
12429.744,
10377.851,]
iterations_mkn_offload_2_teams = [ 1699.623,
 3249.715,
 9020.346,
16641.352,
21505.934,
17135.599,]
iterations_mkn_offload_4_teams = [  1700.977,
 3246.732,
 9022.123,
16657.380,
21463.433,
25867.622,]
iterations_mkn_offload_6_teams = [  1700.200,
 3245.985,
 9003.471,
16663.025,
21138.517,
48436.371,]
iterations_mkn_offload_8_teams = [  1693.361,
 3240.120,
 8967.134,
16631.095,
21261.897,
48459.299,]
iterations_mkn_offload_10_teams = [  1699.021,
 3247.421,
 9020.994,
16671.617,
21244.280,
48331.813,]
iterations_mkn_offload_12_teams = [  1699.728,
 3243.250,
 9022.553,
16619.834,
21263.253,
48235.603,]
iterations_mkn_offload_14_teams = [  1711.180,
 3250.651,
 9118.211,
16717.153,
21327.443,
48428.688,]
iterations_mkn_offload_16_teams = [  1710.917,
 3252.612,
 9106.878,
16778.152,
21305.114,
48325.992,]

iterations_mkn_offload_18_teams = [  1710.906,
 3250.761,
 9121.135,
16756.823,
21466.958,
48296.668,]


# Various team numbers
team_numbers = [64, 128, 256, 512, 700, 1024, 2024, 4096, 8192, 16384]

# Get unique labels for each combination of implementation and team number
labels_mnk_offload = [f'mnk_offload' for n in team_numbers]
labels_mkn_offload = [f'mkn_offload' for n in team_numbers]
labels = labels_mnk_offload + labels_mkn_offload

# Reshape the data for bar plotting
data_mnk_offload = np.array([iterations_mnk_offload_1_team[3], iterations_mnk_offload_2_teams[3], iterations_mnk_offload_4_teams[3],
                         iterations_mnk_offload_6_teams[3], iterations_mnk_offload_8_teams[3], iterations_mnk_offload_10_teams[3],
                         iterations_mnk_offload_12_teams[3], iterations_mnk_offload_14_teams[3], iterations_mnk_offload_16_teams[3], iterations_mnk_offload_18_teams[3]])
data_mkn_offload = np.array([iterations_mkn_offload_1_team[3], iterations_mkn_offload_2_teams[3], iterations_mkn_offload_4_teams[3],
                     iterations_mkn_offload_6_teams[3], iterations_mkn_offload_8_teams[3], iterations_mkn_offload_10_teams[3],
                     iterations_mkn_offload_12_teams[3], iterations_mkn_offload_14_teams[3], iterations_mkn_offload_16_teams[3], iterations_mkn_offload_18_teams[3]])


# Plotting with shades of blue for mnk_offload and shades of red for mkn_offload
mnk_offload_colors = plt.cm.Blues(np.linspace(0.3, 1, len(team_numbers)))
mkn_offload_colors = plt.cm.Reds(np.linspace(0.3, 1, len(team_numbers)))

plt.plot(team_numbers, data_mnk_offload, label=labels_mnk_offload[0], linestyle='-', color=mnk_offload_colors[0])
plt.plot(team_numbers, data_mkn_offload, label=labels_mkn_offload[0], linestyle='--', color=mkn_offload_colors[0])

plt.xlabel('Teams')
plt.xscale("log")
plt.ylabel('MFlops/s [Seconds]')
plt.title('Comparison between teams for Threads=16, Size=1000')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(team_numbers, labels=team_numbers)  # Display thread numbers on the x-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent cropping of labels
plt.savefig("plots/q2plot2_2")
plt.show()