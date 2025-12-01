import matplotlib.pyplot as plt

# Basic plot
plt.plot([1,2,3],[4,5,6])
plt.scatter([1,2,3],[4,5,6])
plt.bar([1,2,3],[4,5,6])

# Labels
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("My Chart")

# Subplots
fig, ax = plt.subplots(2,1)
ax[0].plot([1,2,3],[3,2,1])
ax[1].bar([1,2,3],[4,5,6])

plt.show()

# Customization
plt.plot([1,2,3],[4,5,6], color='red', linestyle='--', marker='o', label='Data 1')
plt.legend()    
plt.grid(True)
plt.xlim(0,4)
plt.ylim(0,7)
plt.savefig("my_chart.png")
plt.close()
# Histograms
data = [1,2,2,3,3,3,4,4,4,4]
plt.hist(data, bins=4, color='blue', alpha=0.7)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
plt.close()
# Pie Chart
labels = ['A', 'B', 'C']
sizes = [30, 50, 20]
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
colors=['gold', 'lightblue', 'lightgreen'])
plt.title("Pie Chart")
plt.show()
plt.close()
# Box Plot
data = [[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]]
plt .boxplot(data, vert=True, patch_artist=True,
boxprops=dict(facecolor='lightblue'))   
plt.title("Box Plot")
plt.xlabel("Groups")
plt.ylabel("Values")
plt.show()
plt.close()
# Heatmap
import numpy as np
data = np.random.rand(5,5)  # example data
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Heatmap")
plt.show()
plt.close()     
# Time Series
dates = np.arange('2023-01', '2023-02', dtype='datetime 64[D]')
values = np.random.rand(len(dates))
plt.plot(dates, values)
plt.title("Time Series")
plt.xlabel("Date")
plt.ylabel("Value") 
plt.show()
plt.close() 
# Multiple Lines
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='Sine Wave')
plt.plot(x, y2, label='Cosine Wave')
plt.legend()
plt.title("Multiple Lines")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
plt.close()
# Annotations
x = [1, 2, 3, 4, 5] 
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.title("Annotations Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
for i, txt in enumerate(y):
    plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()
plt.close()# Save Figure
plt.plot([1,2,3],[4,5,6])       
plt.savefig("figure.png")
plt.close()
# Load and Display Image
img = plt.imread("image.png")
plt.imshow(img)
plt.axis('off')  # hide axes
plt.show()
plt.close()
# Log Scale
plt.plot([1,10,100],[1,10,1000])
plt.xscale('log')
plt.yscale('log')
plt.title("Log Scale Example")
plt.xlabel("X-axis (log scale)")
plt.ylabel("Y-axis (log scale)")
plt.show()
plt.close()
# Error Bars
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
yerr = [0.5, 0.4, 0.6, 0.7, 0.5]
plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='red', capsize=5)
plt.title("Error Bars Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
plt.close()
# Multiple Plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)  
y2 = np.cos(x)
y3 = np.tan(x)
fig, axs = plt.subplots(3, 1, figsize=(6, 12))
axs[0].plot(x, y1, 'r')
axs[0].set_title('Sine Wave')   
axs[1].plot(x, y2, 'g')
axs[1].set_title('Cosine Wave')
axs[2].plot(x, y3, 'b')
axs[2].set_title('Tangent Wave')
plt.tight_layout()
plt.show()
plt.close() 
# Custom Colormap
data = np.random.rand(10,10)    
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Custom Colormap Example")
plt.show()
plt.close() 
# 3D Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
z = [1, 4, 9, 16, 25]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_title("3D Scatter Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()
plt.close()
# Filled Area Plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.fill_between(x, y1, y2, color='lightblue', alpha=0.5)
plt.title("Filled Area Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
plt.close() 
# Stem Plot
x = np.linspace(0, 10, 10)
y = np.sin(x)
plt.stem(x, y, use_line_collection=True)    
plt.title("Stem Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
plt.close()

