#!/usr/bin/evn python

import numpy as np
import cv2
from skimage.feature import peak_local_max
import os
import datetime


def ANMS(corner_img, N_best):
   
    corners_locations = peak_local_max(corner_img, min_distance=20)          
    num_corners = len(corners_locations)                                           

    r = np.ones((num_corners, 1)) * np.inf             
    ED = np.inf

    for i in range(num_corners):
        for j in range(num_corners):
            x_i = corners_locations[i][1]
            y_i = corners_locations[i][0]
            x_j = corners_locations[j][1]
            y_j = corners_locations[j][0]

            C_i = corner_img[y_i, x_i]
            C_j = corner_img[y_j, x_j]

            if C_j > C_i:                                     
                ED = ((x_j - x_i) ** 2) + ((y_j - y_i) ** 2) 

            if ED < r[i]:
                r[i] = ED

    corners_locations_np = np.array(corners_locations)
    corners_y, corners_x = np.split(corners_locations_np, 2, axis=1)
    corners_ranked = np.concatenate((corners_y, corners_x, r), axis=1)

    corners_sorted = corners_ranked[corners_ranked[:, 2].argsort()[::-1]]

    if N_best > num_corners:
        N_best = num_corners
    selected_corners = corners_sorted[0:N_best, :]

    return selected_corners

def get_features(selected_points, input_image):
    patch_size = 41
    features = np.array(np.zeros((64, 1)))
    num_selected_points, cols = selected_points.shape

    for i in range(num_selected_points):

        patch_y = selected_points[i][0]
        patch_x = selected_points[i][1]

        patch = input_image[int(patch_y - ((patch_size-1) / 2)):int(patch_y + (patch_size / 2)),
                            int(patch_x - ((patch_size - 1) / 2)):int(patch_x + (patch_size / 2))]
        
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        subsampled_patch = cv2.resize(blurred_patch, (8, 8))

        subsampled_patch = subsampled_patch.reshape(64, 1)

        subsampled_patch = (subsampled_patch - np.mean(subsampled_patch)) / np.std(subsampled_patch)

        features = np.dstack((features, subsampled_patch))
        
    return features[:, :, 1:]

def match_features(features_set1, features_set2, corners_set1, corners_set2):

    a, b, num_features_set1 = features_set1.shape
    c, d, num_features_set2 = features_set2.shape
    min_features = int(min(num_features_set1, num_features_set2))
    max_features = int(max(num_features_set1, num_features_set2))

    tolerance_ratio = 0.7
    feature_match_pairs = []

    for i in range(min_features):
        feature_matches = {}

        for j in range(max_features):

            feature1 = features_set1[:, :, i]
            feature2 = features_set2[:, :, j]
            corner1 = corners_set1[i, :]
            corner2 = corners_set2[j, :]

            ssd = np.linalg.norm((feature1 - feature2)) ** 2
            feature_matches[ssd] = [corner1, corner2]

        sorted_feature_matches = sorted(feature_matches)

        if sorted_feature_matches[0] / sorted_feature_matches[1] < tolerance_ratio:
            pairs = feature_matches[sorted_feature_matches[0]]
            feature_match_pairs.append(pairs)

    return feature_match_pairs

def visualize_matches(image1, image2, matched_pairs):
    height1, width1, depth1 = image1.shape if len(image1.shape) == 3 else image1.shape + (1,)
    height2, width2, depth2 = image2.shape if len(image2.shape) == 3 else image2.shape + (1,)

    shape = (max(height1, height2), width1 + width2, max(depth1, depth2))

    image_combined = np.zeros(shape, dtype=image1.dtype)

    image_combined[0:height1, 0:width1] = image1
    image_combined[0:height2, width1:width1 + width2] = image2

    image_12 = image_combined.copy()

    circle_size = 4
    red = [0, 0, 255]
    cyan = [255, 255, 0]
    yellow = [0, 255, 255]

    for pair in matched_pairs:
        for i in range(2):
            corner_x = pair[i][1] + i * image1.shape[1]
            corner_y = pair[i][0]
            color = cyan if i == 0 else yellow

            cv2.circle(image_12, (int(corner_x), int(corner_y)), circle_size, color, 1)

        start_point = (int(pair[0][1]), int(pair[0][0]))
        end_point = (int(pair[1][1] + image1.shape[1]), int(pair[1][0]))
        cv2.line(image_12, start_point, end_point, red, 1)

    return image_12

def Homography(pair1, pair2):
    pts1 = pair1.reshape(-1, 2)
    pts2 = pair2.reshape(-1, 2)

    num_points = len(pts1)
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        A[2*i, :] = [-pts1[i, 0], -pts1[i, 1], -1, 0, 0, 0, pts1[i, 0]*pts2[i, 0], pts1[i, 1]*pts2[i, 0], pts2[i, 0]]
        A[2*i + 1, :] = [0, 0, 0, -pts1[i, 0], -pts1[i, 1], -1, pts1[i, 0]*pts2[i, 1], pts1[i, 1]*pts2[i, 1], pts2[i, 1]]

    _, _, V = np.linalg.svd(A)

    H = V[-1, :].reshape((3, 3))
    
    return H

def ransac(allmatches):
    max_iterations = 10000
    threshold = 30
    num_matches = len(allmatches)
    min_inliers_ratio = 0.9
    refined_homography = np.zeros((3, 3))
    maximum = 0

    for iteration in range(max_iterations):
        if num_matches >= 4:
            sample_indices = np.random.choice(num_matches, 4, replace=False)

            sample_pt1 = [np.flip(allmatches[i][0][0:2]) for i in sample_indices]
            sample_pt2 = [np.flip(allmatches[i][1][0:2]) for i in sample_indices]

            sample_pts1 = np.array([sample_pt1])
            sample_pts2 = np.array([sample_pt2])

            homography = Homography(sample_pts1, sample_pts2)

            good_matches = 0
            inliers_indices = []

            for i in range(num_matches):
                pts_y, pts_x = allmatches[i][0][0], allmatches[i][0][1]
                pts_y_h, pts_x_h = allmatches[i][1][0], allmatches[i][1][1]
                image2_pts = np.array([pts_x_h, pts_y_h])
                image1_pts = np.array([pts_x, pts_y, 1])
                transformed_pts = np.matmul(homography, image1_pts)

                if transformed_pts[2] == 0:
                    transformed_pts[2] = 0.000001

                x_transformed_pts = transformed_pts[0] / transformed_pts[2]
                y_transformed_pts = transformed_pts[1] / transformed_pts[2]
                transformed_pts = np.array([x_transformed_pts, y_transformed_pts], np.float32)
                ssd_ransac = np.linalg.norm((image2_pts - transformed_pts)) ** 2

                if ssd_ransac < threshold:
                    good_matches += 1
                    inliers_indices.append(i)

            inliers_pts1 = [np.flip(allmatches[ind][0][0:2]) for ind in inliers_indices]
            inliers_pts2 = [np.flip(allmatches[ind][1][0:2]) for ind in inliers_indices]

            if good_matches > maximum:
                maximum = good_matches
                refined_homography = Homography(np.array([inliers_pts1]), np.array([inliers_pts2]))

                if good_matches > min_inliers_ratio * num_matches:
                    break
        else:
            print('Num_matches < 4 , cannot found ransac feature')
            
    best_inlier_pairs = [allmatches[i] for i in inliers_indices]
    return refined_homography, best_inlier_pairs

def stitchImagePairs(img0, img1, H):
    image0 = img0.copy()
    image1 = img1.copy()

    h0, w0, _ = image0.shape
    h1, w1, _ = image1.shape

    points_on_image0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
    points_on_image0_transformed = cv2.perspectiveTransform(points_on_image0, H)

    points_on_image1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    points_on_merged_images = np.concatenate((points_on_image0_transformed, points_on_image1), axis=0)

    points_on_merged_images_ = []
    for p in range(len(points_on_merged_images)):
        points_on_merged_images_.append(points_on_merged_images[p].ravel())
    points_on_merged_images_ = np.array(points_on_merged_images_)

    x_min, y_min = np.int0(np.min(points_on_merged_images_, axis=0))
    x_max, y_max = np.int0(np.max(points_on_merged_images_, axis=0))

    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    image0_transformed_and_stitched = cv2.warpPerspective(image0, np.dot(H_translate, H), (x_max - x_min, y_max - y_min))

    images_stitched = image0_transformed_and_stitched.copy()
    images_stitched[-y_min:-y_min + h1, -x_min: -x_min + w1] = image1

    indices = np.where(image1 == [0, 0, 0])
    y = indices[0] + -y_min
    x = indices[1] + -x_min

    images_stitched[y, x] = image0_transformed_and_stitched[y, x]

    return images_stitched

def cropImageRect(image):
    
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    _, binary_threshold = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    morphological_kernel = np.ones((5, 5), np.uint8)

    closed_image = cv2.morphologyEx(binary_threshold, cv2.MORPH_CLOSE, morphological_kernel)

    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, morphological_kernel)
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, width, height = cv2.boundingRect(contours[len(contours) - 1])
    cropped_image = image_copy[y:y+height, x:x+width]

    return cropped_image

def save_image_with_timestamp(image, output_folder, filename_prefix):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{filename_prefix}_{timestamp}.png"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)
    return output_path

def MergeImages(Imagelist):
    
    Number = len(Imagelist)
    img0 = Imagelist[0]
    j = 0
    
    for i in range(1,Number):
        j = j+1
        img1 = Imagelist[i]
        
        Pair = [img0, img1]
        
        ANMS_list = []
        feature_descriptor_list = []
        for k in Pair:
            
            """
            Corner Detection
            Save Corner detection output as corners.png
            """
            image_corner = k.copy()
            image_anms = k.copy()
            
            image_grey = cv2.cvtColor(image_corner, cv2.COLOR_BGR2GRAY)
            
            dst = cv2.cornerHarris(image_grey, blockSize=2, ksize=3, k=0.04)
            threshold = 0.01 * dst.max()
            corner_coordinates = np.where(dst > threshold)

            for y, x in zip(*corner_coordinates):
                cv2.circle(image_corner, (x, y), radius=1, color=[0, 0, 255], thickness=-1)

            cv2.imshow('Harris Corner Detection', image_corner)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            output_path_corner = save_image_with_timestamp(image_corner, "Phase1/Results/Corner_detection", "corners")
            print(f"Harris Corner Detection saved at: {output_path_corner}")
            
            """
            Perform ANMS: Adaptive Non-Maximal Suppression
            Save ANMS output as anms.png
            """
            num_best = 100
            image_corners_best = ANMS(dst, num_best)
            
            for corner in image_corners_best:
                cv2.circle(image_anms, (int(corner[1]), int(corner[0])), radius = 1, color=[0, 255, 0], thickness = -1)

            cv2.imshow('ANMS', image_anms)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            output_path_anms = save_image_with_timestamp(image_anms, "Phase1/Results/ANMS", "ANMS")
            print(f"ANMS saved at: {output_path_anms}")
            ANMS_list.append(image_corners_best)
            
            """
            Feature Descriptors
            Save Feature Descriptor output as FD.png
            """
            features_descriptor = get_features(image_corners_best, image_grey)
            feature_descriptor_list.append(features_descriptor)

        """
	    Feature Matching
	    Save Feature Matching output as matching.png
	    """
        matched_i_iplus1 = match_features(feature_descriptor_list[0], feature_descriptor_list[1], ANMS_list[0], ANMS_list[1])
        output_image_matching = visualize_matches(Pair[0], Pair[1], matched_i_iplus1)
        cv2.imshow('Matching', output_image_matching)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        output_path_matching = save_image_with_timestamp(output_image_matching, "Phase1/Results/Feature_matching", "Matching")
        print(f"Feature Matching saved at: {output_path_matching}")
        
        """	
        Refine: RANSAC, Estimate Homography
        """
        H_matrix, matches = ransac(matched_i_iplus1)
        output_image_ransac = visualize_matches(Pair[0], Pair[1], matches)
        
        cv2.imshow('RANSAC', output_image_ransac)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        output_path_ransac = save_image_with_timestamp(output_image_ransac, "Phase1/Results/RANSAC", "RANSAC")
        print(f"RANSAC saved at: {output_path_ransac}")
        
        """
	    Image Warping + Blending
	    Save Panorama output as mypano.png
	    """
     
        stitch_image = stitchImagePairs(Pair[0], Pair[1], H_matrix)
        stitch_image = cropImageRect(stitch_image)

        # Display and save the final stitched image
        cv2.imshow('Final Stitched Image', stitch_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path_stitched = save_image_with_timestamp(stitch_image, "Phase1/Results/Panoroma", "Stitched_Image")
        print(f"Panorama saved at: {output_path_stitched}")
        
        First_Image = stitch_image 
        
    return First_Image

def main():

    """
    Read a set of images for Panorama stitching
    """
    folder_path = "Phase1/Data/Set1"
    
    output_folder_Results = "Phase1/Results"
    output_folder_Corner = "Phase1/Results/Corner_detection"
    output_folder_ANMS = "Phase1/Results/ANMS"
    output_folder_feature_match = "Phase1/Results/Feature_matching"
    output_folder_RANSAC = "Phase1/Results/RANSAC"
    output_folder_Panoroma = "Phase1/Results/Panoroma"

    os.makedirs(output_folder_Results, exist_ok=True)
    os.makedirs(output_folder_Corner, exist_ok=True)
    os.makedirs(output_folder_ANMS, exist_ok=True)
    os.makedirs(output_folder_feature_match, exist_ok=True)
    os.makedirs(output_folder_RANSAC, exist_ok=True)
    os.makedirs(output_folder_Panoroma, exist_ok=True)
    
    Image_dataset = []
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
    else:
        files = os.listdir(folder_path)

        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    img = cv2.imread(file_path)
                    Image_dataset.append(img)
                except Exception as e:
                    print(f"Error reading image '{file_name}': {e}")
                    
    No_of_images = len(Image_dataset)
    First_Image = Image_dataset[0]
    
    for n in range(No_of_images - 1):
        Img_list = [First_Image, Image_dataset[n+1]]
        First_Image = MergeImages(Img_list)
        
    output_path_stitched = save_image_with_timestamp(First_Image, "Phase1/Results/Panoroma", "Final_Stitched_Image")
    print(f"Panorama saved at: {output_path_stitched}")

if __name__ == "__main__":
    main()
