# Import modules
import imageio

# Generate gif file
gif = []
for filename in ['Gan'+str(i)+'.jpg' for i in range(50)]:
    gif.append(imageio.imread(filename))
imageio.mimsave('GanMNIST.gif', gif)