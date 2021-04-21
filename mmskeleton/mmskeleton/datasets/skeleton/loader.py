import os
import numpy as np
import json
import torch
import csv
import math

class SkeletonLoader(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to data folder
        num_track: number of skeleton output
        pad_value: the values for padding missed joint
        repeat: times of repeating the dataset
    """
    def __init__(self, data_dir, num_track=1, repeat=1, num_keypoints=-1, outcome_label='UPDRS_gait', missing_joint_val=0, csv_loader=False):
        self.data_dir = data_dir
        self.num_track = num_track
        self.num_keypoints = num_keypoints
        self.files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
        ] * repeat
        self.outcome_label = outcome_label
        self.missing_joint_val = missing_joint_val
        self.csv_loader = csv_loader
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
# {
    # "info":
        # {
            # "video_name": "skateboarding.mp4",
            # "resolution": [340, 256],
            # "num_frame": 300,
            # "num_keypoints": 17,
            # "keypoint_channels": ["x", "y", "score"],
            # "version": "1.0"
        # },
    # "annotations":
        # [
            # {
                # "frame_index": 0,
                # "id": 0,
                # "person_id": null,
                # "keypoints": [[x, y, score], [x, y, score], ...]
            # },
            # ...
        # ],
    # "category_id": 0,
# }
        if self.csv_loader:
            # print("getting itemmmmm", self.outcome_label)
            # print(self.files[index])
            data_struct = {} 
            with open(self.files[index]) as f:        
                data = csv.reader(f)
                csvreader = csv.DictReader(f)
                for row in csvreader:
                    for colname in row:
                        if colname not in data_struct:
                            try:
                                data_struct[colname] = [float(row[colname])]
                            except ValueError as e:
                                data_struct[colname] = [row[colname]]

                        else:
                            try:
                                data_struct[colname].append(float(row[colname]))
                            except ValueError as e:
                                data_struct[colname].append(row[colname])


            
            info_struct = {
                "video_name": data_struct['walk_name'][0],
                "resolution": [1920, 1080],
                "num_frame": len(data_struct['time']),
                "num_keypoints": 17,
                "keypoint_channels": ["x", "y", "score"],
                "version": "1.0"
            }


            order_of_keypoints = {'Nose', 
                'RShoulder', 'RElbow', 'RWrist', 
                'LShoulder', 'LElbow', 'LWrist', 
                'RHip', 'RKnee', 'RAnkle', 
                'LHip', 'LKnee', 'LAnkle', 
                'REye', 'LEye', 'REar', 'LEar'}

            annotations = []
            for ts in range(len(data_struct['time'])):

                ts_keypoints = []
                for kp in order_of_keypoints:
                    if kp == "Neck":
                        RShoulder = [data_struct['RShoulder_x'][ts], data_struct['RShoulder_y'][ts], data_struct['RShoulder_conf'][ts]]   
                        LShoulder = [data_struct['LShoulder_x'][ts], data_struct['LShoulder_y'][ts], data_struct['LShoulder_conf'][ts]]   
                        print(RShoulder, LShoulder)
                        x = ( RShoulder[0] +  LShoulder[0] ) / 2
                        y = ( RShoulder[1] +  LShoulder[1] ) / 2
                        try:
                            conf = ( RShoulder[2] +  LShoulder[2] ) / 2
                        except:
                            conf = 0
                    else:
                        x = data_struct[kp + '_x'][ts]          
                        y = data_struct[kp + '_y'][ts]          
                        conf = data_struct[kp + '_conf'][ts]      
                    
                    # missing confidence = 0
                    try:
                        conf = float(conf)
                    except:                    
                        conf = 0

                    # missing actual joint coordinates
                    try:
                        x = float(x)
                        y = float(y)
                    except:                    
                        x = self.missing_joint_val
                        y = self.missing_joint_val

                    if math.isnan(conf):
                        conf = 0
                    if math.isnan(x) or math.isnan(y):
                        x = self.missing_joint_val
                        y = self.missing_joint_val
    
                    ts_keypoints.append([x, y, conf])

                cur_ts_struct = {'frame_index': ts,
                                'id': 0, 
                                'person_id': 0,
                                'keypoints': ts_keypoints}
                annotations.append(cur_ts_struct)

            outcome_cat = data_struct[self.outcome_label][0]
            try:
                outcome_cat = float(outcome_cat)
                outcome_cat = int(outcome_cat)   
            except:
                outcome_cat = -1

            
            data = {'info': info_struct, 
                        'annotations': annotations,
                        'category_id': outcome_cat}
        
        else: # original loader 
            with open(self.files[index]) as f:
                data = json.load()
        # # print("we got: ", data_arr[0])

        info = data['info']
        annotations = data['annotations']
        num_frame = info['num_frame']
        num_keypoints = info[
            'num_keypoints'] if self.num_keypoints <= 0 else self.num_keypoints
        channel = info['keypoint_channels']
        num_channel = len(channel)

        # # get data
        data['data'] = np.zeros(
            (num_channel, num_keypoints, num_frame, self.num_track),
            dtype=np.float32)

        for a in annotations:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data['data'][:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()
        
        # print(data['data'])
        return data
