## NDVI Tree Crown Detection
This is a simple sattelite imagery tree crown segmentation which uses NDVI treshold to segment crowns. at the end it returns crown masks which is useful for data labeling and ground truth production.If the loaded file is inappropriate, you can push "Delete File" button and remove it from files directory. If NDVI doesn't work precise fore tree crown segmentation, you can push "Manual" button. With this "MANUAL_" will be added to begining of your file name like: "MANUAL_patchxxxx.tif" , then you can sort these files and lable them manually with other platforms. Note that by default, it shows normalized RGB image. if you wanna see images with stretched bands, you should change "rgb_image" with "R_I" in this code. stretching bands might be useful in some images.

![Screenshot (694)](https://github.com/user-attachments/assets/0fa0f8be-9808-42d6-bd09-575e7579a540)

## delinated Tree Crowns:

![Screenshot (690)](https://github.com/user-attachments/assets/76992b3e-5910-43f9-b95c-1973e60320fe)

## binary mask:


![patch_16_42_L-PNN_100](https://github.com/user-attachments/assets/0c9cdc03-ea2d-4e61-85d3-8bc235102ee0)


This simple code allows you to make binary masks of tree crowns. run this code in your repo and set the treshhold. after you push the save button, the final mask of that image will be saved in maks directory and the original image will be moved to masked_images directory.
