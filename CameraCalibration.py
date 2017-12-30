import cv2
import matplotlib.pyplot as plt
import CameraManager

def undistort_images(src_name, tgt_name):
    # Test undistortion on an image
    img = cv2.imread(src_name)
    dst = camera.undistort(img)
    cv2.imwrite(tgt_name, dst)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize = 30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize = 30)
    plt.show()

camera = CameraManager.CameraManager('center')
res = camera.calibrateOnChessboard(
    './camera_cal',
    'calibration*.jpg',
    (9, 5))

if res:
    undistort_images('./camera_test/calibration1.jpg', './camera_test/dist_calibration1.jpg')
    undistort_images('./camera_test/calibration2.jpg', './camera_test/dist_calibration2.jpg')
    undistort_images('./camera_test/calibration3.jpg', './camera_test/dist_calibration3.jpg')
    undistort_images('./camera_test/calibration4.jpg', './camera_test/dist_calibration4.jpg')
    undistort_images('./camera_test/calibration5.jpg', './camera_test/dist_calibration5.jpg')
    undistort_images('./camera_test/calibration6.jpg', './camera_test/dist_calibration6.jpg')
    undistort_images('./camera_test/calibration7.jpg', './camera_test/dist_calibration7.jpg')
    undistort_images('./camera_test/calibration8.jpg', './camera_test/dist_calibration8.jpg')
    undistort_images('./camera_test/calibration9.jpg', './camera_test/dist_calibration9.jpg')
    undistort_images('./camera_test/calibration17.jpg', './camera_test/dist_calibration17.jpg')
    undistort_images('./camera_test/calibration18.jpg', './camera_test/dist_calibration18.jpg')
    undistort_images('./camera_test/test1.jpg', './camera_test/dist_test1.jpg')
    undistort_images('./camera_test/test6.jpg', './camera_test/dist_test6.jpg')


    