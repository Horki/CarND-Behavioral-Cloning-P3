import matplotlib.pyplot as plt

from model import *

if __name__ == "__main__":
    correction = 0.2
    samples = load_data("{}/{}".format(DATA_PATH, DRIVING_LOG))
    sample = samples[1200]
    img_center = load_image(sample[0])
    img_left = load_image(sample[1])
    img_right = load_image(sample[2])
    img_center_c = img_center[50:IMG_HEIGHT-20, 0:IMG_WIDTH, :]
    img_left_c = img_left[50:IMG_HEIGHT-20, 0:IMG_WIDTH, :]
    img_right_c = img_right[50:IMG_HEIGHT-20, 0:IMG_WIDTH, :]
    steer = float(sample[3])
    fig, axs = plt.subplots(4, 3, figsize=(15, 15))
    axs = axs.ravel()

    axs[0].imshow(img_left)
    axs[0].set_title("Left : Steer = {}".format(steer - correction))
    axs[1].imshow(img_center)
    axs[1].set_title("Center : Steer = {}".format(steer))
    axs[2].imshow(img_right)
    axs[2].set_title("Right: Steer = {}".format(steer + correction))

    axs[3].imshow(np.fliplr(img_left))
    axs[3].set_title("Flip Left : Steer = {}".format(-(steer - correction)))
    axs[4].imshow(np.fliplr(img_center))
    axs[4].set_title("Flip Center : Steer = {}".format(-steer))
    axs[5].imshow(np.fliplr(img_right))
    axs[5].set_title("Flip Right: Steer = {}".format(-(steer + correction)))


    axs[6].imshow(img_left_c)
    axs[6].set_title("Cropped Left : Steer = {}".format(steer - correction))
    axs[7].imshow(img_center_c)
    axs[7].set_title("Cropped Center : Steer = {}".format(steer))
    axs[8].imshow(img_right_c)
    axs[8].set_title("Cropped Right: Steer = {}".format(steer + correction))

    axs[9].imshow(np.fliplr(img_left_c))
    axs[9].set_title("Cropped Flip Left : Steer = {}".format(-(steer - correction)))
    axs[10].imshow(np.fliplr(img_center_c))
    axs[10].set_title("Cropped Flip Center : Steer = {}".format(-steer))
    axs[11].imshow(np.fliplr(img_right_c))
    axs[11].set_title("Cropped Flip Right: Steer = {}".format(-(steer + correction)))


    plt.savefig("examples/dataset.png")
    plt.show()
