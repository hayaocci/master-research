import os
import cv2


from const import *
import module.process_image
def change_contrast_main(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(input_dir)
    numeric_filenames = []
    for file in files:
        name, ext = os.path.splitext(file)
        if name.isdigit():  # 名前が数値かどうかを確認
            numeric_filenames.append(int(name))
    
    if numeric_filenames:
        smallest_value = min(numeric_filenames) # ファイル名の最長値を取得

    for i in range(sum(os.path.isfile(os.path.join(input_dir, name)) for name in os.listdir(input_dir))):
        
        img = cv2.imread(os.path.join(input_dir, str(i+smallest_value) + '.png'))
        
        if img is None:
            print('Could not open or find the image:', input_dir + str(i+smallest_value) + '.png')
            exit(0)

        # new_img = module.process_image.change_contrast(img, 1.5, -30)
        new_img = module.process_image.sigmoidTone(img, 0.1, 127.5)
        cv2.imwrite(os.path.join(output_dir, str(i+smallest_value) + '.png'), new_img)

    print(f"img:{sum(os.path.isfile(os.path.join(input_dir, name)) for name in os.listdir(input_dir))} === Contrast changed images saved in {output_dir}")

def crop_main(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(input_dir)
    numeric_filenames = []
    for file in files:
        name, ext = os.path.splitext(file)
        if name.isdigit():  # 名前が数値かどうかを確認
            numeric_filenames.append(int(name))
    
    if numeric_filenames:
        smallest_value = min(numeric_filenames) # ファイル名の最長値を取得

    for i in range(sum(os.path.isfile(os.path.join(input_dir, name)) for name in os.listdir(input_dir))):
        try:
            img = cv2.imread(os.path.join(input_dir, str(i+smallest_value) + '.png'))
        except:
            img = cv2.imread(os.path.join(input_dir, str(i+smallest_value) + '.jpg'))
        
        if img is None:
            print('Could not open or find the image:', input_dir + str(i+smallest_value) + '.png')
            exit(0)

        new_img = module.process_image.crop_square(img)
        new_img = module.process_image.change_size(new_img, IMG_SIZE)
        cv2.imwrite(os.path.join(output_dir, str(i+smallest_value) + '.png'), new_img)

    print(f"img:{sum(os.path.isfile(os.path.join(input_dir, name)) for name in os.listdir(input_dir))} === Cropped images saved in {output_dir}")

def convert_to_bin(img):

    # convert grayscale img to binary img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


img = cv2.imread('edit-img/original/1.jpg')
new_img = module.process_image.crop_square(img)
new_img = module.process_image.change_size(new_img, IMG_SIZE)
cv2.imwrite('edit-img/edit/1.png', new_img)


# if __name__ == '__main__':
    # change_contrast_main(IMG_DIR, EDIT_IMG_DIR)
    # crop_main('edit-img/original/', 'edit-img/edit')

