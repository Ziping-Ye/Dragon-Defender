"""
    This script preprocesses the extracted features:
        (1) one-hot encode categorical features
        (2) normalize numerical features
"""

import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from tqdm import tqdm
from utils import transform, construct_one_hot_encoder


base_dir = r"/home/zipingye/cellular-ids"
# base_dir = r"/Users/zipingye/Desktop/cellular-ids"

window_length = 32
in_dim = 157
feature_preprocessing_error_list = []
feature_preprocessing_error_file = os.path.join(base_dir, "preprocessing/feature_preprocessing_error.txt")

# define categorical and numerical features
# order of categorical features in this list must be the same as the order in csv file
categorical_features = ["message name", "Attach with IMSI", "Null encryption", 
                        "Enable IMEISV", "Cell ID", "TAC", "EMM state", 
                        "EMM substate", "EMM cause"]
numerical_features = ["paging_record_number"]

def pretrain_preprocess(filename, outfile=None):

    print("***** pretrain preprocessing *****")

    # construct the one-hot encoder
    OneHotEnc = construct_one_hot_encoder(base_dir, categorical_features) 

    # load data
    df = pd.read_csv(filename)

    window_list = []
    label_list = []

    for idx, row in tqdm(df.iterrows()):
        
        try:
            # one-hot encode categorical features
            cat_features = row.loc[categorical_features].tolist()
            # print("before transform"); print(len(cat_features)); print(len(cat_features[0].split()))
            cat_features = transform(cat_features)
            # print("after transform"); print(len(cat_features)); print(len(cat_features[0]))
            # print(cat_features)

            for i in range(len(cat_features)):
                assert len(cat_features[i]) == len(categorical_features), \
                       "number of categorical features does not match -- message " + str(i)
            
            OneHotEncodedResult = OneHotEnc.transform(cat_features) 
            OneHotEncodedResult = OneHotEncodedResult.tolist() # (window_length-1 * in_dim-num_numerical_features)
            # print(len(OneHotEncodedResult), len(OneHotEncodedResult[1]))
            # print("OneHotEncodedResult[0]", OneHotEncodedResult[0])

            # normalize numerical features within each sequence
            for numerical_feature in numerical_features:
                scaler = StandardScaler()
                scaled_sequence = scaler.fit_transform([[float(e)] for e in row.loc[numerical_feature].split()]).flatten()
                # print("numerical feature normalization:")
                # print(len(scaled_sequence)) # window_length-1
                # update the sequence
                row.loc[numerical_feature] = " ".join([str(round(e, 2)) for e in scaled_sequence])
            
            numerical_encodings = [[] for _ in range(len(OneHotEncodedResult))]
            for numerical_feature in numerical_features:
                num_feat_seq = [float(elem) for elem in row.loc[numerical_feature].split()]
                # print(len(num_feat_seq)) # window_length-1
                assert len(num_feat_seq) == len(OneHotEncodedResult), "the length of categorical feature sequence and numerical feature sequence not match"
                # add each numerical feature to numerical_encodings
                for i in range(len(num_feat_seq)):
                    numerical_encodings[i].append(num_feat_seq[i])

            # print("numerical_encodings:")
            # print(numerical_encodings)
            
            raw_message_embeddings = [[] for _ in range(window_length)] # a list of tensors, each tensor is the raw message embedding (concatenation of all features' vector)
            # the first is a tensor of all zeros - initial window embedding that does not carry any information
            raw_message_embeddings[0] = torch.zeros(in_dim)
            # raw_message_embeddings[0] = [0.0 for _ in range(in_dim)]
            
            counter = 1
            # concatenate categorical features and numerical features 
            for categorical_encoding, numerical_encoding in zip(OneHotEncodedResult, numerical_encodings):
                raw_message_embedding = categorical_encoding + numerical_encoding
                # print("raw_message_embedding dimension (in_dim) = ", len(raw_message_embedding)) # in_dim
                assert len(raw_message_embedding) == in_dim, "[concat] the length of raw message embedding and in_dim not match"
                # if idx == 0:
                # #     in_dim = len(raw_message_embedding)
                #     print("in_dim = ", len(raw_message_embedding))
                raw_message_embeddings[counter] = torch.tensor(raw_message_embedding)
                # raw_message_embeddings[counter] = raw_message_embedding
                counter += 1

            # use the below code when only have categorical features
            # counter = 1
            # for categorical_encoding in OneHotEncodedResult:
            #     raw_message_embedding = categorical_encoding
            #     assert len(raw_message_embedding) == in_dim, "the length of raw message embedding and in_dim not match"
            #     # print("raw_message_embedding dimension (in_dim) = ", len(raw_message_embedding)) # in_dim
            #     raw_message_embeddings[counter] = torch.tensor(raw_message_embedding)
            #     # raw_message_embeddings[counter] = raw_message_embedding
            #     counter += 1
            
            assert counter <= window_length, "window length should be less than or equal to predefined window_length " + str(counter)
            
            # pad all windows to window_length
            current_len = len(raw_message_embeddings)
            if current_len < window_length:
                pad = window_length - current_len
                for i in range(pad):
                    raw_message_embeddings[current_len + i] = torch.zeros(in_dim)
                    # raw_message_embeddings[current_len + i] = [0.0 for _ in range(in_dim)]
            
            window_list.append(torch.stack(raw_message_embeddings))
            # window_list.append(raw_message_embeddings)
            label = int(row.loc["label"])
            label_list.append(label)

        except Exception as e:
            print("exception raised at row", idx)
            error_message = f"{e} at row {idx}\n"
            print(error_message)
            feature_preprocessing_error_list.append(error_message)

    # check number of examples (here windows)
    assert len(window_list) == len(label_list), "the length of window_list and label_list do not match " + str(len(window_list)) + " " + str(len(label_list))

    # print(window_list[0].shape) # [window_length, in_dim]
    result_dict = {"window" : window_list, "label" : label_list}
    result = pd.DataFrame().from_dict(result_dict)
    # print("(after pretrain preprocessing - result shape: ", result.shape) # (num_examples, 2)
    # print("each window embedding shape: ", result.iloc[0,0].shape) # (window_length, in_dim)

    if outfile:
        result.to_csv(outfile, index=False)

    return result


def train_preprocess(filename, val, outfile=None):

    print("***** train preprocessing *****")

    # construct the one-hot encoder
    OneHotEnc = construct_one_hot_encoder(base_dir, categorical_features) 

    # load data
    df = pd.read_csv(filename)

    if not val:
        df.drop(columns=['attack_type'], inplace=True) # drop attack_type column for train.csv

    trace_list = []
    label_list = []

    if val:
        attack_type_list = []

    for idx, row in tqdm(df.iterrows()):
        
        try:
            trace_length = len(row.iloc[0].split())
            # print(trace_length)

            # normalize numerical features within each sequence
            for numerical_feature in numerical_features:
                scaler = StandardScaler()
                scaled_sequence = scaler.fit_transform([[float(e)] for e in row.loc[numerical_feature].split()]).flatten()
                # print(scaled_sequence)
                # update the sequence
                row.loc[numerical_feature] = " ".join([str(round(e, 4)) for e in scaled_sequence])

            # one-hot encode categorical features
            cat_features = row.loc[categorical_features].tolist()
            # print(cat_features)
            cat_features = transform(cat_features)
            # print(cat_features)
            OneHotEncodedResult = OneHotEnc.transform(cat_features)
            # OneHotEncodedResult = OneHotEncodedResult.todense().tolist() # (window_length-1*dimension of concatenated categorical encoding)
            OneHotEncodedResult = OneHotEncodedResult.tolist()
            # print(len(OneHotEncodedResult), len(OneHotEncodedResult[1]))

            numerical_encodings = [[] for _ in range(len(OneHotEncodedResult))]
            for numerical_feature in numerical_features:
                num_feat_seq = [float(elem) for elem in row.loc[numerical_feature].split()]
                # print(num_feat_seq)
                assert len(num_feat_seq) == len(OneHotEncodedResult), "the length of categorical feature sequence and numerical feature sequence not match"
                # add each numerical feature to numerical_encodings
                for i in range(len(num_feat_seq)):
                    numerical_encodings[i].append(num_feat_seq[i])   
            # print(numerical_encoding)
            
            raw_message_embeddings = [[] for _ in range(trace_length)] # a list of tensors, each tensor is the raw message embedding (concatenation of all features' vector)
            
            counter = 0
            # concatenate categorical features and numerical features 
            for categorical_encoding, numerical_encoding in zip(OneHotEncodedResult, numerical_encodings):
                raw_message_embedding = categorical_encoding + numerical_encoding
                assert len(raw_message_embedding) == in_dim, "[concat] the length of raw message embedding and in_dim not match"
                raw_message_embeddings[counter] = torch.tensor(raw_message_embedding)
                # raw_message_embeddings[counter] = raw_message_embedding
                counter += 1

            # use the below cod when only have categorical features
            # counter = 0
            # for categorical_encoding in OneHotEncodedResult:
            #     raw_message_embedding = categorical_encoding
            #     assert len(raw_message_embedding) == in_dim, "the length of raw message embedding and in_dim not match"
            #     # print("raw_message_embedding dimension (in_dim) = ", len(raw_message_embedding)) # in_dim
            #     raw_message_embeddings[counter] = torch.tensor(raw_message_embedding)
            #     # raw_message_embeddings[counter] = raw_message_embedding
            #     counter += 1
            
            assert counter == trace_length

            trace_list.append(torch.stack(raw_message_embeddings))
            # trace_list.append(raw_message_embeddings)
            label = [int(l) for l in row.loc["label"].split()]
            label_list.append(label)
            
            if val:
                attack_type = [int(at) for at in row.loc["attack_type"].split()]
                attack_type_list.append(attack_type)

        except Exception as e:
            error_message = f"{e} at row {idx}\n"
            print(error_message)
            feature_preprocessing_error_list.append(error_message)

    # check number of examples (here traces)
    assert len(trace_list) == len(label_list), "the length of trace_list and label_list not match " + str(len(window_list)) + " " + str(len(label_list))

    if val:
        assert len(label_list) == len(attack_type_list), "the length of label_list and attack_type_list not match"

    # print(trace_list[0].shape) # [trace_length, in_dim]
    result_dict = {"trace" : trace_list, "label" : label_list}

    if val:
        result_dict["attack_type"] = attack_type_list
        
    result = pd.DataFrame().from_dict(result_dict)

    if outfile:
        result.to_csv(outfile, index=False)

    return result


if __name__ == "__main__":

    # base_dir = r"/home/zipingye/cellular-ids"
    base_dir = r"/Users/zipingye/Desktop/cellular-ids"

    window_length = 32
    feature_preprocessing_error_list = []

    # define categorical and numerical features
    # order of categorical features in this list must be the same as the order in csv file
    categorical_features = ["message name", "Attach with IMSI", "Null encryption", 
                            "Enable IMEISV", "Cell ID", "TAC", "EMM state", 
                            "EMM substate", "EMM cause"]
    numerical_features = ["paging_record_number"]

    with open(os.path.join(base_dir, "preprocessing/feature_preprocessing_error.txt"), 'w') as f:
        f.writelines(feature_preprocessing_error_list)
    
    # pretrain_preprocess(os.path.join(base_dir, "traces/exclude_0_attacks_version_1/pretrain.csv"))
    # pretrain_preprocess("./traces/sample_raw_pretrain.csv", "./traces/sample_pretrain.csv")
    train_preprocess(os.path.join(base_dir, "traces/exclude_0_attacks_version_1/train.csv"), False)

