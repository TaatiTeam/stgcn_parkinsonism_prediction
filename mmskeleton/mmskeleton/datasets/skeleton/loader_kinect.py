import os
import numpy as np
import json
import torch
import csv
import math
import pandas as pd
import copy
from sklearn import preprocessing


class SkeletonLoaderKinect(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to data folder
        num_track: number of skeleton output
        pad_value: the values for padding missed joint
        repeat: times of repeating the dataset
    """
    def __init__(self, data_dir, num_track=1, repeat=1, num_keypoints=-1, 
                outcome_label='UPDRS_gait', missing_joint_val=0, csv_loader=False, 
                cache=False, layout='kinect_coco_simplified_head', flip_skels=False, 
                belmont_data_mult = 0, use_gait_feats=False, fit_scaler=None, scaler=None, export_2d=False,
                extrema_range=None):
        self.data_dir = data_dir
        self.num_track = num_track
        self.num_keypoints = num_keypoints
        self.export_2d = export_2d
        self.files = data_dir * repeat
        
        # Look for belmont data and repeat it if necessary
        self.belmont_data = None

        self.outcome_label = outcome_label
        self.missing_joint_val = missing_joint_val
        self.csv_loader = csv_loader
        self.interpolate_with_mean = False
        self.layout = layout
        
        self.flip_skels = flip_skels

        if self.missing_joint_val == 'mean':
            self.interpolate_with_mean = True
            self.missing_joint_val = 0

        self.extrema_range = extrema_range
        if not self.extrema_range:
            if self.outcome_label == "SAS_gait":
                self.extrema_range = 3
            else:
                self.extrema_range = 2

        self.class_dist = {}
        for i in range(self.extrema_range + 1):
            self.class_dist[i] = 0 

        self.cache = cache
        self.cached_data = {}

     
        self.sample_extremes = False
        self.cached_extreme_inds = []

        # Load in the gait features if available
        self.gait_feats = None
        self.num_gait_feats = 0
        self.gait_feats_names = []
        self.use_gait_feats = use_gait_feats
        self.fit_min_max_scaler = fit_scaler
        self.min_max_scaler = scaler
        if self.use_gait_feats:
            try:
                base, _ = os.path.split(self.data_dir[0])
                gait_file = os.path.join(base, "gait_feats", "gait_features.csv")
                df = pd.read_csv(gait_file, engine='python')
                gait_feature_names_2d = ['cadence','avgMOS','avgminMOS','timeoutMOS',  'avgstepwidth',  'CVstepwidth',  'CVsteptime', 'SIStepTime', 'stepsofwalk']

                gait_feature_names_3d = ['walk_speed', 'cadence', 'step_time','step_length','step_width','CV_step_time','CV_step_length','CV_step_width','Symmetry_step_time','Symmetry_step_length' ,'Symmetry_step_width'	,'MOS_average', 'MOS_minimum']

                try:
                    x = df[gait_feature_names_2d].values
                    self.gait_feats_names = gait_feature_names_2d
                except:
                    x = df[gait_feature_names_3d].values
                    self.gait_feats_names = gait_feature_names_3d
                    
                if self.fit_min_max_scaler is None:
                    self.min_max_scaler = preprocessing.MinMaxScaler()
                    self.fit_min_max_scaler = self.min_max_scaler.fit_transform(x)
                df_temp = pd.DataFrame(self.fit_min_max_scaler, columns=self.gait_feats_names, index = df.index)
                df[self.gait_feats_names] = df_temp
                df.fillna(0, inplace=True)

                # For kinect also need to clean up the filenames
                def cleanup_names(name):
                    # max length is 42
                    temp_name = name[6:42]
                    # Now split this and remove trailing Skel
                    parts = temp_name.split("_")
                    if (parts[-1]).isalpha():
                        parts = parts[0:-1]
                    elif ((parts[-1]).isnumeric() and (parts[-2]).isnumeric()):
                        parts.insert(-1, '')

                    return "_".join(parts)

                df['walk_name'] = df['walk_name_full'].apply(cleanup_names)
    

                self.gait_feats = df
                self.num_gait_feats = len(self.gait_feats_names)


            except:
                self.gait_feats = None

        if self.cache:
            print("loading data to cache...")
            for index in range(self.__len__()):
                self.get_item_loc(index)

    def get_class_dist(self):
        if self.sample_extremes:
            extrema_dist = copy.deepcopy(self.class_dist)
            for i in range(1, self.extrema_range):
                extrema_dist[i] = 0

            return extrema_dist
            
        return self.class_dist

    def __len__(self):
        if self.flip_skels:
            return len(self.files)*2
        return len(self.files)

    def extremaLength(self):
        return len(self.cached_extreme_inds)

    def get_fit_scaler(self):
        return self.fit_min_max_scaler

    def get_scaler(self):
        return self.min_max_scaler

    def get_num_gait_feats(self):
        return self.num_gait_feats

    def relabelItem(self, index, newLabel):
        if index not in self.cached_data:
            print("Don't have this data, skipping relabel...", index)
            return
        if self.cached_data[index]['have_true_label']:
            return

        # Make sure that the label is within the admissible range of [0, 4]
        if newLabel < 0:
            newLabel = 0
        if newLabel > 3 and self.outcome_label == "SAS_gait":
            newLabel = 3
        elif newLabel > 2 and self.outcome_label == "UPDRS_gait":
            newLabel = 2

        
        # Update the class distributions
        old_label = int(round(self.cached_data[index]['category_id']))
        self.cached_data[index]['category_id'] = newLabel

        if old_label >= 0:
            self.class_dist[old_label] -= 1

        roundedLabel = int(round(newLabel))
        if roundedLabel in self.class_dist:
            self.class_dist[roundedLabel] += 1 
        else:   
            self.class_dist[roundedLabel] = 1

        # print(self.cached_data[index])


        # Add to extrema map if needed
        # old and new are both extrema, so don't need to do anything
        # old was extrema, remove it from the extrema list
        if self.isExtrema(old_label) and not self.isExtrema(roundedLabel):
            inds = [i for i,x in enumerate(self.cached_extreme_inds) if x==index]
            inds.sort(reverse = True)
            for ind in inds:
                del self.cached_extreme_inds[ind]

        # new is extrema, append to extrema list
        if not self.isExtrema(old_label) and self.isExtrema(roundedLabel):
            self.cached_extreme_inds.append(index)

    def isExtrema(self, label):
        if (label == 0 or label == self.extrema_range):
            return True
        return False



    def get_item_loc(self, index):
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

        # if index in self.cached_data:
        #     if self.sample_extremes:
        #         extremaInd = index % self.extremaLength()
        #         return self.cached_extremes[extremaInd]
        #     else:
        #         return self.cached_data[index]

        if index >= len(self.files):
            flip_index = index
            index = index - len(self.files)
            return_flip = True
        else:
            flip_index = index + len(self.files)
            return_flip = False



        if self.csv_loader:
            file_index = index
            if index >= len(self.files):
                file_index = index - len(self.files)

            data_struct_interpolated = pd.read_csv(self.files[file_index])
            data_struct_interpolated.fillna(data_struct_interpolated.mean(numeric_only=True), inplace=True)


            data_struct = {} 
            with open(self.files[file_index]) as f:        
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

            gait_feature_vec = [0.0] * self.num_gait_feats

            # Load in the gait features
            if self.use_gait_feats:
                # How we clean the walk name depends if we have 2D or 3D data
                # For 3D data we need to keep the state
                clean_walk_name = data_struct['walk_name'][0][0:-8]

                # Use the gait features if requested and available
                if self.gait_feats is not None:
                    row = self.gait_feats.loc[self.gait_feats['walk_name'] == clean_walk_name, self.gait_feats_names]
                    if not row.empty:
                        gait_feature_vec = row.values.tolist()[0]


            if self.layout == 'kinect_coco_simplified_head':
                num_kp = 13
                order_of_keypoints = ['Head', 
                    'LShoulder', 'RShoulder',
                    'LElbow', 'RElbow', 
                    'LWrist', 'RWrist', 
                    'LHip', 'RHip',
                    'LKnee', 'RKnee',
                    'LAnkle', 'RAnkle',
                ]

            else:
                raise ValueError(f"The layout {self.layout} does not exist")

            # print(data_struct)
            try:
                info_struct = {
                        "video_name": data_struct['walk_name'][0],
                        "resolution": [1920, 1080],
                        "num_frame": len(data_struct['time']),
                        "keypoint_channels": ["x", "y", "z"],
                        "num_keypoints": num_kp,
                        "version": "1.0"
                }

                if self.export_2d:
                    info_struct["keypoint_channels"] =  ["x", "y", "z_invalid"]
                else:
                    info_struct['resolution'] = [1, 1, 1] # The 3D data is already in meters so don't need to normalize
            except:
                print('data_struct', data_struct)            
                raise ValueError("something is wrong with the data struct", self.files[file_index])

            # If we have belmont data, reverse the order of the resolution parameter since the video is in portrait mode
            first_char = data_struct['walk_name'][0][0]
            if first_char.upper() == "B":
                info_struct['resolution'] = [1080, 1920]

            annotations = []
            annotations_flipped = []
            num_time_steps = len(data_struct['time'])
            for ts in range(num_time_steps):
                ts_keypoints, ts_keypoints_flipped = [], []
                for kp_num, kp in enumerate(order_of_keypoints):

                    x = data_struct[kp + '_x'][ts]          
                    y = data_struct[kp + '_y'][ts]
                    if self.export_2d: 
                        z = self.missing_joint_val
                    else:
                        z = data_struct[kp + '_z'][ts]      

                    # check if we are missing actual joint coordinates
                    try:
                        x = float(x)
                        y = float(y)
                        z = float(z)
                    except:
                        if self.interpolate_with_mean:
                            x = data_struct_interpolated[kp + '_x'][ts]          
                            y = data_struct_interpolated[kp + '_y'][ts] 
                            if self.export_2d: 
                                z = self.missing_joint_val
                            else:
                                z = data_struct_interpolated[kp + '_z'][ts]     

                        else:           
                            x = self.missing_joint_val
                            y = self.missing_joint_val
                            z = self.missing_joint_val
                        
                        if isinstance(x, str):
                            x = self.missing_joint_val

                        if isinstance(y, str):
                            y = self.missing_joint_val                        
                        
                        if isinstance(z, str):
                            z = self.missing_joint_val

                    if math.isnan(x) or math.isnan(y) or math.isnan(z):
                        x = self.missing_joint_val
                        y = self.missing_joint_val
                        z = self.missing_joint_val

                    # Flip the left and right sides (flipping x)
                    if kp_num == 0: # Nose isn't flipped
                        x_flipped = x
                    else:
                        cur_side = kp[0]
                        if cur_side.upper() == "L":
                            kp_other_side = "R" + kp[1:]
                        elif cur_side.upper() == "R":
                            kp_other_side = "L" + kp[1:]
                        else:
                            raise ValueError("cant flip: ", kp)
                        x_flipped = data_struct[kp_other_side + '_x'][ts]  

                        # missing actual joint coordinates
                        try:
                            x_flipped = float(x_flipped)
                        except:
                            if self.interpolate_with_mean:
                                x_flipped = data_struct_interpolated[kp_other_side + '_x'][ts]          
                            else:           
                                x_flipped = self.missing_joint_val

                        if math.isnan(x_flipped):
                            x_flipped = self.missing_joint_val

                    ts_keypoints.append([x, y, z])
                    ts_keypoints_flipped.append([x_flipped, y, z])

                cur_ts_struct = {'frame_index': ts,
                                'id': 0, 
                                'person_id': 0,
                                'keypoints': ts_keypoints}

                cur_ts_struct_flipped = {'frame_index': ts,
                                'id': 0, 
                                'person_id': 0,
                                'keypoints': ts_keypoints_flipped}

                annotations.append(cur_ts_struct)
                annotations_flipped.append(cur_ts_struct_flipped)

            outcome_cat = data_struct[self.outcome_label][0]
            try:
                outcome_cat = float(outcome_cat)
                outcome_cat = int(outcome_cat)   
            except:
                outcome_cat = -1

            if outcome_cat in self.class_dist:
                self.class_dist[outcome_cat] += 1
            else:
                self.class_dist[outcome_cat] = 1

            data = {'info': info_struct, 
                        'category_id': outcome_cat}
        
        else: # original loader 
            with open(self.files[index]) as f:
                data = json.load()

        info = data['info']
        num_frame = info['num_frame']
        num_keypoints = info[
            'num_keypoints'] if self.num_keypoints <= 0 else self.num_keypoints
        channel = info['keypoint_channels']
        num_channel = len(channel)

        data['data'] = np.zeros(
            (num_channel, num_keypoints, num_frame, self.num_track),
            dtype=np.float32)

        for a in annotations:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data['data'][:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()
        data['data_flipped'] = np.zeros(
                (num_channel, num_keypoints, num_frame, self.num_track),
                dtype=np.float32)

        for a in annotations_flipped:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data['data_flipped'][:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()


        data['num_ts'] = num_frame

        if data['category_id'] >= 0:
            data['have_true_label'] = 1
        else:
            data['have_true_label'] = 0

        data['gait_feats'] = gait_feature_vec


        flipped_data = copy.deepcopy(data)
        temp_flipped = flipped_data['data_flipped']
        flipped_data['data_flipped'] = flipped_data['data']
        flipped_data['data'] = temp_flipped
        flipped_data['name'] = self.files[file_index] + "_flipped"

        data['name'] = self.files[file_index]
        data['index'] = index
        flipped_data['index'] = flip_index
        # Add to extrema list if this score is on the extremes
        if self.isExtrema(data['category_id']):
            self.cached_extreme_inds.append(index)

        if self.cache:
            self.cached_data[index] = data

            if self.flip_skels:    
                self.cached_data[flip_index] = flipped_data
                

        if self.flip_skels and return_flip:
            return flipped_data

        return data


    def __getitem__(self, index):
        if index in self.cached_data:
            if self.sample_extremes and self.extremaLength() > 0:
                extremaInd = index % self.extremaLength()
                return copy.deepcopy(self.cached_data[self.cached_extreme_inds[extremaInd]])
            else:
                return self.cached_data[index]

        return self.get_item_loc(index)
