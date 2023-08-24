import os
import imageio

def clear_graph_folder():
    folder = './graphs'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)

def create_gifs():
    # Combine all images in graphs folder into a gif
    images = []
    for filename in os.listdir('./graphs'):
        images.append((filename, imageio.imread('./graphs/' + filename)))
    # sort the images based on number in filename
    images = sorted(images, key=lambda x: int(x[0].split('.')[0]))
    images = [image[1] for image in images]
    imageio.mimsave('./animation.gif', images, duration=50)
