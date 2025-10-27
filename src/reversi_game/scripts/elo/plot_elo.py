import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the data
data = {
    'Agent': [
        'azero200', 'azero30', 'ppo_cnn', 'Mcts500', 'ppo_mlp',
        'MinMaxHuman depth dyn', 'MinMaxGA depth dyn',
        'MinMaxHuman depth 1', 'MinMaxGA depth 1',
        'Mcts100', 'Mcts30', 'ars_mlp', 'trpo_cnn', 'Random'
    ],
    'Elo Score': [
        1782, 1551, 1336, 1328, 1320,
        1311, 1252, 1191, 1153, 1140,
        997, 917, 873, 615
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create bar chart
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x='Elo Score', y='Agent', data=df, palette='viridis')

# Add score numbers at the end of each bar
for index, value in enumerate(df['Elo Score']):
    bar_plot.text(value, index, f'{value}', va='center', ha='left', fontsize=10, color='black')

# Add titles and labels
plt.title('Elo Score after 100 round tournament')
plt.xlabel('Elo Score')
plt.ylabel('Agent')

# Display the plot
plt.show()
