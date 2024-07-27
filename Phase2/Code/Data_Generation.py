import cv2
import numpy as np
import pandas as pd
import os

def count_images_in_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return len(image_files)

def GetPatches(image, patch_size=128, perturbation=32, border=42, translation = 10):

    h, w = image.shape[:2]
    min_size = patch_size + 2 * border + 1
    
    if w > min_size and h > min_size:
        end_margin = patch_size + border 

        x = np.random.randint(border, w - end_margin)
        y = np.random.randint(border, h - end_margin)
        
        translation = np.random.randint(-translation, translation)

        pts1 = np.array([[x, y], [x, patch_size + y], [patch_size + x, y], [patch_size + x, patch_size + y]])
        pts2 = np.zeros_like(pts1)

        for i, pt in enumerate(pts1):
            pts2[i][0] = pt[0] + np.random.randint(-perturbation, perturbation) + translation
            pts2[i][1] = pt[1] + np.random.randint(-perturbation, perturbation) + translation

        H_inv = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))) 
        imageB = cv2.warpPerspective(image, H_inv, (w, h))

        Patch_a = image[y:y + patch_size, x:x + patch_size]
        Patch_b = imageB[y:y + patch_size, x:x + patch_size]
        H4 = (pts2 - pts1).astype(np.float32)

        return Patch_a, Patch_b, H4, imageB, pts1, pts2
    else:
        return None, None, None, None, None

def generate_data_set(option, path, save_path):
    im_count = count_images_in_folder(path)
    patches_per_image = ['a', 'b']
    
    if not os.path.isdir(save_path):
        print(f"{save_path} was not present, creating the folder...")
        os.makedirs(save_path)

    H4_list = []
    Ca_list = []
    Cb_list = []

    print(f"Generating {option} data with {im_count} images ......")
    print("Begin Data Generation .... ")

    for a in patches_per_image:
        print(f"Doing attempt: {a}")
        for i in range(1, im_count + 1):
            image_a = cv2.imread(os.path.join(path, f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
            image_a = cv2.resize(image_a, (320, 240), interpolation=cv2.INTER_AREA)
            
            patch_a, patch_b, H4, _, Ca, Cb = GetPatches(image_a, patch_size=128, perturbation=32, border=42, translation=10)

            if patch_a is None and patch_b is None and H4 is None:
                print("encountered None return.. ignoring Image..")
            else:
                sub_directories = ['PA', 'PB', 'IA']
                for sub_dir in sub_directories:
                    sub_path = os.path.join(save_path, sub_dir)
                    if not os.path.isdir(sub_path):
                        print(f"Subdirectories inside {save_path} were not present.. creating the folders...")
                        os.makedirs(sub_path)

                path_a = os.path.join(save_path, 'PA', f"{i}{a}.jpg")
                path_b = os.path.join(save_path, 'PB', f"{i}{a}.jpg")
                im_path_a = os.path.join(save_path, 'IA', f"{i}{a}.jpg")

                cv2.imwrite(path_a, patch_a)
                cv2.imwrite(path_b, patch_b)
                cv2.imwrite(im_path_a, image_a)

                image_name = f"{i}{a}.jpg"
                H4_values = list(np.hstack((H4[:, 0], H4[:, 1])))
                H4_list.append([image_name, H4_values[0], H4_values[4], H4_values[1], H4_values[5],H4_values[2], H4_values[6], H4_values[3],H4_values[7]])
                CA_values = list(np.hstack((Ca[:, 0], Ca[:, 1])))
                Ca_list.append([image_name, CA_values[0], CA_values[4], CA_values[1], CA_values[5],CA_values[2], CA_values[6], CA_values[3],CA_values[7]])
                CB_values = list(np.hstack((Cb[:, 0], Cb[:, 1])))
                Cb_list.append([image_name, CB_values[0], CB_values[4], CB_values[1], CB_values[5],CB_values[2], CB_values[6], CB_values[3],CB_values[7]])

    df = pd.DataFrame(H4_list)
    df.to_csv(os.path.join(save_path, "H4.csv"),header=False, index=False)
    print(f"Saved H4 data in: {save_path}")
    
    df1 = pd.DataFrame(Ca_list)
    df1.to_csv(os.path.join(save_path, "Ca.csv"),header=False, index=False)
    print(f"Saved Ca data in: {save_path}")
    
    df2 = pd.DataFrame(Cb_list)
    df2.to_csv(os.path.join(save_path, "Cb.csv"),header=False, index=False)
    print(f"Saved Cb data in: {save_path}")

def main():
    train_path = 'Phase2/Data/Train/'
    train_save_path = 'Phase2/Data/Train_synthetic/'
    generate_data_set('Train', train_path, train_save_path)

    val_path = 'Phase2/Data/Val/'
    val_save_path = 'Phase2/Data/Val_synthetic/'
    generate_data_set('Val', val_path, val_save_path)

    test_path = 'Phase2/Data/Test/'
    test_save_path = 'Phase2/Data/Test_synthetic/'
    generate_data_set('Test', test_path, test_save_path)

if __name__ == '__main__':
    main()
