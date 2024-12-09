import pandas as pd
import matplotlib.pyplot as plt

annotations_path = 'annotations/ssl-dataset/segments_annotations.csv'
df = pd.read_csv(annotations_path)

true_counts = df[['is_field_boundary', 'is_field_marking', 'is_not_a_field_feature']].sum()

# Plot the bar chart
true_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Count of True Values in Columns')
plt.ylabel('Number of True Values')
plt.xticks(rotation=0)
plt.show()