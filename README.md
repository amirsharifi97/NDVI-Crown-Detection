This is a simple sattelite imagery tree crown segmentation which uses NDVI treshold to segment crowns. at the end it returns crown masks which is useful for data labeling and ground truth production.

![Screenshot (689)](https://github.com/user-attachments/assets/7f9433b6-b642-4794-8454-2fd7d5f1bc42)
![Screenshot (688)](https://github.com/user-attachments/assets/b3c16b78-4ff1-4509-89f1-93cb17cd6a42)

final binary mask:

![patch_17_6_L-PNN_100](https://github.com/user-attachments/assets/807fb862-6ee9-41b4-a44d-ffd725e22ed2)



This simple code allow you to make binary masks of tree crowns. run this code in your repo and set the treshhold. after you push the save button, the final mask of that image will be saved in maks directory and the original image will be moved to masked_images directory.
