import matplotlib.pyplot as plt

def retainer(x, y):
    # Reshape to 28x28
    image_2d_real = x.reshape(28, 28)

    # Display the image on screen
    plt.figure(figsize=(3, 3))                  
    plt.imshow(image_2d_real, cmap='gray')      
    plt.title(f"Label: {y}")
    plt.axis('off')                         
    plt.show()