import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import math
import numpy as np
import random
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

'''
::COCO Dataset Body parts Keypoints::
Nose – 0,
Neck – 1,
Right Shoulder – 2,
Right Elbow – 3,
Right Wrist – 4,
Left Shoulder – 5,
Left Elbow – 6,
Left Wrist – 7,
Right Hip – 8,
Right Knee – 9,
Right Ankle – 10,
Left Hip – 11,
Left Knee – 12,
LAnkle – 13,
Right Eye – 14,
Left Eye – 15,
Right Ear – 16,
Left Ear – 17,
Background – 18
'''

BODY_PARTS_1 = { "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15 , "16": 16, "17": 17}

POSE_PAIRS_1 = [ ["0", "1"], ["1", "2"], ["1", "5"], ["2", "3"], ["3", "4"], ["5", "6"], ["6", "7"], ["1", "8"], ["1", "11"], ["8", "9"], ["9", "10"], ["11", "12"], ["12", "13"]]

def visualize_output(pose1, pose2, size):
    # Initialize blank canvas
    canvas = np.ones(size)
    # Plot points on images
    for pair in POSE_PAIRS_1:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS_1)
        assert(partTo in BODY_PARTS_1)
        idFrom = int(BODY_PARTS_1[partFrom])
        idTo = int(BODY_PARTS_1[partTo])
        if pose1[idFrom].all() and pose1[idTo].all():
            cv2.line(canvas, pose1[idFrom], pose1[idTo], (0, 255, 0), 5)
            cv2.ellipse(canvas, pose1[idFrom], (4, 4), 0, 0, 360, (0, 255, 0), cv2.FILLED)
            cv2.ellipse(canvas, pose1[idTo], (4, 4), 0, 0, 360, (0, 255, 0), cv2.FILLED)

    for pair in POSE_PAIRS_1:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS_1)
        assert(partTo in BODY_PARTS_1)
        idFrom = int(BODY_PARTS_1[partFrom])
        idTo = int(BODY_PARTS_1[partTo])
        if pose2[idFrom].all() and pose2[idTo].all():
            cv2.line(canvas, pose2[idFrom], pose2[idTo], (255, 0, 0), 5)
            cv2.ellipse(canvas, pose2[idFrom], (4, 4), 0, 0, 360, (255, 0, 0), cv2.FILLED)
            cv2.ellipse(canvas, pose2[idTo], (4, 4), 0, 0, 360, (255, 0, 0), cv2.FILLED)
    return canvas

# Cosine Distance Metric
# Input: L2 normized pose vectors
# Output: Cosine distance between the two vectors
def cosine_distance(pose1, pose2):
	# Find the cosine similarity
	cossim = pose1.dot(np.transpose(pose2)) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))
	# Find the cosine distance
	cosdist = (1 - cossim)
	return cosdist

# Weighted Distance Metric
# Input: L2 normized pose vectors, and confidence scores for each point in pose 1
# Output: Weighted distance between the two vectors
def weight_distance(pose1, pose2, conf1):
	# D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||)
	#		 = sum1 * sum2
	# Compute first summation
	sum1 = 1 / np.sum(conf1)

	# Compute second summation
	sum2 = 0
	for i in range(len(pose1)):
		conf_ind = math.floor(i / 2) # each index i has x and y that share same confidence score
		sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])
	weighted_dist = sum1 * sum2

	return weighted_dist


# similarity_score Function
# Input: L2 normized pose vectors
# Output: Prints the cossim distance between the two poses
def similarity_score(pose1, pose2, cur_conf):
    p1 = []
    p2 = []
    pose_1 = np.array(pose1, dtype=np.float)
    pose_2 = np.array(pose2, dtype=np.float)

    # Normalize coordinates
    pose_1[:,0] = pose_1[:,0] / max(pose_1[:,0])
    pose_1[:,1] = pose_1[:,1] / max(pose_1[:,1])
    pose_2[:,0] = pose_2[:,0] / max(pose_2[:,0])
    pose_2[:,1] = pose_2[:,1] / max(pose_2[:,1])

    # Turn (16x2) into (32x1)
    for joint in range(pose_1.shape[0]):
        x1 = pose_1[joint][0]
        y1 = pose_1[joint][1]
        x2 = pose_2[joint][0]
        y2 = pose_2[joint][1]

        p1.append(x1)
        p1.append(y1)
        p2.append(x2)
        p2.append(y2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Looking to minimize the distance if there is a match
    # Computing two different distance metrics
    scoreA = cosine_distance(p1, p2)
    logger.info('***************')
    logger.info('Cosine_Distance')
    logger.info(scoreA)
    
    scoreB = weight_distance(p1, p2, cur_conf)
    logger.info('***************')
    logger.info('Weighted_Distance')
    logger.info(scoreB)
    return scoreA, scoreB

def adjust_the_pose(pose1, pose2):
    pose1_new = np.array(pose1)
    pose2_new = np.array(pose2)
    pose1_new[:,0] = pose1_new[:,0] - min(pose1_new[:,0]) + 40
    pose1_new[:,1] = pose1_new[:,1] - min(pose1_new[:,1])
    pose2_new[:,0] = pose2_new[:,0] - min(pose2_new[:,0]) + 40
    pose2_new[:,1] = pose2_new[:,1] - min(pose2_new[:,1])
    
    resize_x = max(pose2_new[:,0])/max(pose1_new[:,0])
    resize_y = max(pose2_new[:,1])/max(pose1_new[:,1])
    pose1_new[:,0] = pose1_new[:,0] * resize_x
    pose1_new[:,1] = pose1_new[:,1] * resize_y
    
    pose1_resized = tuple(map(tuple, pose1_new))
    pose2_resized = tuple(map(tuple, pose2_new))
    pose1_r = np.array(pose1_resized)
    pose2_r = np.array(pose2_resized)
    pose1_resized = pose1_r.astype(int)
    pose2_resized = pose2_r.astype(int)
    
    # Get dimensions of output window
    pose_1 = np.array(pose1_new)
    pose_2 = np.array(pose2_new)
    max_y = max(max(pose_1[:,0]), max(pose_2[:,0]))
    max_x = max(max(pose_1[:,1]), max(pose_2[:,1]))
    dim = (432, 368, 3) 
    new_image = visualize_output(pose1_resized, pose2_resized, dim)
    return new_image

def choose_a_random_image(image_loc, e):
    image = common.read_imgfile(image_loc, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans_detected = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    logger.info(humans_detected)
    elapsed = time.time() - t
    logger.info('inference image: %s in %.4f seconds.' % (image_loc, elapsed))
    image_orig = TfPoseEstimator.draw_humans(image, humans_detected, imgcopy=False)
    plt.close('all')
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Target')
    plt.imshow(cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB))
    canvas = np.ones((432, 368, 3))
    canvas [:, :] = (210, 0, 0)
    b = fig.add_subplot(1, 2, 2)
    b.set_title('Pose not matched yet')
    plt.imshow(canvas)
    plt.show(block=False)
    
    if humans_detected:
        keypoints_orig = str(str(str(humans_detected[0]).split('BodyPart:')[1:]).split('-')).split(' score=')
        conf_orig = []
        keypoints_list_orig=[]
        for i in range (len(keypoints_orig)-1): 
            cur_body_part = keypoints_orig[i][-19:-16].split("'")[1]
            while (len(keypoints_list_orig)) != int(cur_body_part):
                keypoints_list_orig.append(tuple([0,0]))
            pnt = keypoints_orig[i][-11:-1]
            pnt = tuple(map(float, pnt.split(', ')))
            keypoints_list_orig.append(pnt)
            if (i>0):
                while int(cur_body_part) <= len(conf_orig):
                    conf_orig.append(0)
                conf_orig.append(float(keypoints_orig[i].split(" ")[0]))
        while (len(keypoints_list_orig)) != 18:
            keypoints_list_orig.append(tuple([0,0]))
        while (len(conf_orig)) != 18:
            conf_orig.append(0)
        keypts_array = np.array(keypoints_list_orig)
        keypts_array = keypts_array*(image.shape[1],image.shape[0])
        orig_keypoints = keypts_array.astype(int)
    
    return orig_keypoints, conf_orig, image_orig

def display_result_image(result_received, target_pose):
    plt.close('all')
    if int(result_received):
        # Fill image with Green color
        canvas = common.read_imgfile('.\images\green_pic.jpg', None, None)
        fig = plt.figure()
        a = fig.add_subplot(1, 1, 1)
        a.set_title('Pose-Matched')
    else:
        # Fill image with red color
        canvas = common.read_imgfile('.\images\red_pic.jpg', None, None)
        fig = plt.figure()
        a = fig.add_subplot(1, 1, 1)
        a.set_title('Pose Not-Matched')
    time.sleep(0.5)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.show()
    time.sleep(2)
    plt.close()
    

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
fps_time = 0

if __name__ == '__main__':
    # Define the arguements required from command line
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()
    
    # Define the pose estimator class based on the w and h values
    # Default --> 0, 0
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    # estimate human poses from a single image
    image_loc = args.image
    orig_keypoints, orig_confidence, target_pose = choose_a_random_image(image_loc, e)
    # Define default camera port '0'
    cam = cv2.VideoCapture(0)
    # Define the threshold values per image.
    # Since we have 8 test images in the image folder, we have array with 8 threshold values.
    cosine_thr_array = [0.006, 0.011, 0.006, 0.025, 0.014, 0.005, 0.008, 0.01]
    weighted_thr_array = [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]
    global_testpic_idx = '6'
    rand_int = 6
    while True:
        # Break the whole program at any point if we hit escape
        # ASCII code for 'Esc' button --> 27 
        if cv2.waitKey(1) == 27:
            break
        # Read the image from Camera port
        ret_val, image = cam.read()
        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #logger.debug('postprocess+')
        dark_image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image , humans, imgcopy=False)
        cv2.imshow('Original Image', image )
        #logger.debug('show+')
        image_1 = cv2.resize(image, (432,368))
        # If humans detected in the image capture the keypoints locations
        if humans:
            # Split the humans string which has all the bodypart location and its confidence
            # Create a list of (X,Y) co ordinates for each keypoint.
            # Create an array with respective confidence values.
            keypoints = str(str(str(humans[0]).split('BodyPart:')[1:]).split('-')).split(' score=')
            keypoints_list=[]
            current_conf = []
            for i in range (len(keypoints)-1):
                cur_body_part = keypoints[i][-19:-16].split("'")[1]
                while (len(keypoints_list)) != int(cur_body_part):
                    keypoints_list.append(tuple([0,0]))
                pnt = keypoints[i][-11:-1]
                pnt = tuple(map(float, pnt.split(', ')))
                keypoints_list.append(pnt)
                if (i>0):
                    while int(cur_body_part) <= len(current_conf):
                        current_conf.append(0)
                    current_conf.append(float(keypoints[i].split(" ")[0]))
            while (len(keypoints_list)) != 18:
                keypoints_list.append(tuple([0,0]))
            while (len(current_conf)) != 18:
                current_conf.append(0.0)
            keypts_array = np.array(keypoints_list)
            keypts_array = keypts_array*(image_1.shape[1],image_1.shape[0])
            keypts_array = keypts_array.astype(int)
            cur_keypoints = keypts_array
        else:
            keypoints_list=[]
            while (len(keypoints_list)) != 18:
                keypoints_list.append(tuple([0,0]))
            keypts_array = np.array(keypoints_list)
            keypts_array = keypts_array.astype(int)
            cur_keypoints = keypts_array
        cv2.putText(image_1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        fps_time = time.time()
        returned_image = visualize_output(orig_keypoints, cur_keypoints, (432,368,3))
        cv2.imshow('tf-overlapped result', returned_image )
        # Define and set the variables to '0'
        # These variables holds the values of number of '0's in a human pose.
        # '0' represents a body part not detected.
        null_val = 0
        orig_count = 0
        cur_count = 0
        cur_and_orig_count = 0
        orig_keypoints_array = [item for t in orig_keypoints[0:13] for item in t]
        cur_keypoints_array = [item for t in cur_keypoints[0:13] for item in t]
        for i in range(len(orig_keypoints_array)):
            if cur_keypoints_array[i] == null_val:
                cur_count = cur_count + 1
                if orig_keypoints_array[i] == null_val:
                    orig_count = orig_count + 1
            if (orig_keypoints_array[i] == null_val) and (cur_keypoints_array[i] == null_val):
                cur_and_orig_count = cur_and_orig_count + 1
        is_all_zero = np.all((cur_keypoints_array == 0))
        if (orig_count == cur_count) and (orig_count == cur_and_orig_count) and (not is_all_zero):
            cur_cosine_score, cur_weighted_score = similarity_score(cur_keypoints[:-5], orig_keypoints[:-5], current_conf[:-5])
            logger.info(cur_cosine_score)
            logger.info(cosine_thr_array[int(rand_int)-1])
            logger.info(cur_weighted_score)
            logger.info(weighted_thr_array[int(rand_int)-1])
            # check if the cosine score and weighted score are under threshold values
            # If yes, print the poase matched to the debug logs and load the next image
            if (cur_cosine_score < cosine_thr_array[int(rand_int)-1]) & (cur_weighted_score < weighted_thr_array[int(rand_int)-1]):
                size_adjusted_image = adjust_the_pose(orig_keypoints, cur_keypoints)
                cv2.imshow('Final_result', size_adjusted_image )
                logger.debug('Pose Matched with cosine & weight threshold conditions')
                display_result_image(1, target_pose)
                # Generate a random number for image index
                rand_int = str(random.randint(1,8))
                image_loc = image_loc.replace(global_testpic_idx, rand_int)
                global_testpic_idx = rand_int
                # After replacing the index, infer humans from the picture
                # Replace the new humans results in 'orig_keypoints' variable.
                orig_keypoints, orig_confidence, target_pose = choose_a_random_image(image_loc, e)
        # logger.debug('finished+')
    cv2.destroyAllWindows()
