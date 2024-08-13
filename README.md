This is a simple sattelite imagery tree crown segmentation which uses NDVI treshold to segment crowns. at the end it returns crown masks which is useful for data labeling and ground truth production.

![Screenshot (689)](https://github.com/user-attachments/assets/7f9433b6-b642-4794-8454-2fd7d5f1bc42)

![Screenshot (690)](https://github.com/user-attachments/assets/76992b3e-5910-43f9-b95c-1973e60320fe)

final binary mask:


![patch_16_42_L-PNN_100](https://github.com/user-attachments/assets/0c9cdc03-ea2d-4e61-85d3-8bc235102ee0)


This simple code allow you to make binary masks of tree crowns. run this code in your repo and set the treshhold. after you push the save button, the final mask of that image will be saved in maks directory and the original image will be moved to masked_images directory.
