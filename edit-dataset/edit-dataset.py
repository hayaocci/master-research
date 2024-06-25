import os
import cv2


from const import *
import module.change_contrast


def change_contrast_main(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for i in range(sum(os.path.isfile(os.path.join(input_dir, name)) for name in os.listdir(input_dir))):
        
        img = cv2.imread(os.path.join(input_dir, str(i) + '.png'))
        
        if img is None:
            print('Could not open or find the image:', input_dir + str(i) + '.png')
            exit(0)

        new_img = module.change_contrast.change_contrast(img, 1.5, -30)
        cv2.imwrite(os.path.join(output_dir, str(i) + '.png'), new_img)

    print(f"img:{sum(os.path.isfile(os.path.join(input_dir, name)) for name in os.listdir(input_dir))} === Contrast changed images saved in {output_dir}")

if __name__ == '__main__':
    change_contrast_main(IMG_DIR, EDIT_IMG_DIR)