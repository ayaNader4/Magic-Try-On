import os
import os.path
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


def resize_single(img_path, dest_folder, suffix):
    img = cv2.imread(os.path.join(img_path), cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    height = 512
    width = 384

    blank_img = np.zeros((height, width, 3), np.uint8)
    blank_img[:, :] = (255, 255, 255)
    final_img = blank_img.copy()

    h, w, c = rgb_img.shape

    if h / w > 1.3333:
        new_width = int(w * (height / h))
        img_resize = cv2.resize(rgb_img, dsize=(new_width, height), interpolation=cv2.INTER_CUBIC)
        x_offset = int((width - w * (height / h)) / 2)
        final_img[0:height, x_offset:x_offset + new_width] = img_resize.copy()

    else:
        new_height = int(h * (width / w))
        img_resize = cv2.resize(rgb_img, dsize=(width, new_height), interpolation=cv2.INTER_CUBIC)
        y_offset = int((height - new_height) / 2)

        final_img[y_offset:y_offset + new_height, 0:width] = img_resize
    print(final_img.shape)

    # rename, to img_0.jpg for example
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(dest_folder, f'{img_name}_{suffix}.jpg').replace("\\", "/")
    print(output_path)
    cv2.imwrite(output_path, final_img)

    del img, rgb_img
    return output_path

def remove_background_single(masks_dir,img_path, mask=False):
    model_name = "unet"
    session = new_session(model_name)
    input_image = Image.open(img_path)
    output = remove(input_image, only_mask=mask,session=session, bgcolor=(255, 255, 255, 255),
                    post_process_mask=True).convert('RGB')
    #os.remove(image_path)
    if mask == True:
      img_name = os.path.splitext(os.path.basename(img_path))[0]
      img_path = masks_dir + img_name + '.jpg'
    output.save(img_path)
    print(img_path)

    del input_image
    del output


def create_pairs_file_single(image_path, cloth_path, pairs_file_path):
    image_name = os.path.basename(image_path)
    cloth_path = os.path.basename(cloth_path)
    with open(pairs_file_path, 'w') as f:
      f.write(f'{image_name} {cloth_path}\n')