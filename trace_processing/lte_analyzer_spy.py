import xml.etree.ElementTree as ET
from mobile_insight.analyzer import *
import pandas as pd
import json
import os

__all__ = ["LteAnalyzerSpy"]

class LteAnalyzerSpy(Analyzer):

    def __init__(self, filepath):
        Analyzer.__init__(self)
        print("Init RRC Analyzer Spy")        
        self.include_analyzer("LteRrcAnalyzer", [self.__on_lte_rrc_msg])
        self.include_analyzer("TrackCellInfoAnalyzer", [self.__on_lte_rrc_msg])
        self.include_analyzer("LteNasAnalyzer", [])
        # load supported message json file
        self.filepath = filepath
        self.messages_file = open("./supported_messages.json")
        self.su_messages = json.load(self.messages_file) #dictionary
        self.messages_file.close()

        # facility carry-on
        self.gid = 0
        self.tac = 0
        self.SIB1_flag = 0
        self.out = SpyFeauresList()

    def __on_lte_rrc_msg(self, msg):
        features = SpyFeatures()
        supported_messages = [[msg[0], msg[1]] for msg in self.su_messages["SUPPORTED_MESSAGES"]] 
        if msg.type_id == "LTE_RRC_OTA_Packet": #Do we need other packet type here?
            found = 0
            # get RRC message name
            for field in msg.data.iter('field'):
                if found:
                    break
                for support in supported_messages:
                    if found:
                        break
                    if support[0] in field.get('name'):              
                        features.message_name = support[1]
                        found = 1
                        break
            # get NAS message name
            found = 0
            for proto in msg.data.iter('proto'):
                if found:
                    break
                # print(proto.get('name'))
                if proto.get('name') == "nas-eps":
                    for field in proto.iter('field'):
                        if found:
                            break
                        # print(field.get('name'))
                        # print(field.get('showname'))
                        for support in supported_messages:
                            if found:
                                break
                            if field.get('showname'):
                                if support[0] in field.get('showname'):
                                    if support[1] != features.message_name:
                                        features.message_name = features.message_name + "_" + support[1]
                                        found = 1
                                        # attach with IMSI
                                        if features.message_name == "dl_info_transfer_identity_req":
                                            for field in proto.iter('field'):
                                                if "IMSI" in field.get('showname'):
                                                    features.IMSI_attach = 1
                                        # null encryption
                                        if features.message_name == "dl_info_transfer_sm_command":
                                            for field in proto.iter('field'):
                                                if field.get("name") == "nas_eps.emm.toc":
                                                    if "null ciphering algorithm" in field.get('showname'):
                                                        features.null_encryption = 1
                                                if field.get("name") == "nas_eps.emm.imeisv_req":
                                                    if "IMEISV requested" in field.get('showname'):
                                                        features.enable_IMEISV = 1
                                        # EMM cause
                                        if "reject" in features.message_name:
                                            for field in proto.iter('field'):
                                                if field.get('show') == 'EMM cause':
                                                    for val in field.iter('field'):
                                                        if val.get('name') == 'nas_eps.emm.cause':
                                                            features.emm_cause = val.get('show')
                                        break
            # print(features.message_name)
        else:
            return
        # ignore empty message name
        if not features.message_name:
            return
        # if SIB1, get cellid and tac
        if features.message_name == "SIB1":
            # ET.dump(msg.data)
            for field in msg.data.iter('field'):
                if field.get('name') == 'lte-rrc.trackingAreaCode':
                    features.tac = int(field.get("value"), 16)
                    self.tac = features.tac
                if field.get('name') == 'lte-rrc.cellIdentity':
                    features.gid = int(field.get("value"), 16) >> 4
                    self.gid = features.gid
            self.SIB1_flag = 1
            
        else:
            if self.SIB1_flag == 1:
                if "rrc_conn" in features.message_name:
                    self.SIB1_flag = 0
                features.tac = self.tac
                features.gid = self.gid
            else:
                gid = self.get_analyzer("TrackCellInfoAnalyzer").get_cur_cell_id()
                if gid:
                    features.gid = gid

                tac = self.get_analyzer("TrackCellInfoAnalyzer").get_cur_cell_tac()
                if tac:
                    features.tac = tac

        if features.message_name == "paging":
            for field in msg.data.iter('field'):
                if field.get('name') == 'lte-rrc.pagingRecordList':
                    features.paging_record_number = field.get("show")
        # get EMM state
        features.emm_state = self.get_analyzer("LteNasAnalyzer")._LteNasAnalyzer__emm_status.state
        features.emm_substate = self.get_analyzer("LteNasAnalyzer")._LteNasAnalyzer__emm_status.substate
        # features.rsrp = self.rsrp
        self.out.append(features)

        # write results to file
        # with open("./RRClog.txt", 'a') as f:
        #     f.write(f'Message Name:{features.message_name}\n')
        #     f.write(f'Attach with IMSI = {features.IMSI_attach}\n')
        #     f.write(f'Null encryption = {features.null_encryption}\n')
        #     f.write(f'Cell ID = {features.gid}\n')
        #     f.write(f'TAC = {features.tac}\n')           
        #     f.write(f'cell reselection threshold = {features.threshserv}\n')
        #     f.write(f'EMM state = {features.emm_state}\n')
        #     f.write(f'EMM substate = {features.emm_substate}\n')
    
    def toCSV(self):
        if len(self.out.message_name) > 16: # min trace length
            dict = {"message name" : self.out.message_name,
                    "Attach with IMSI" : self.out.IMSI_attach,
                    "Null encryption" : self.out.null_encryption,
                    "Enable IMEISV" : self.out.enable_IMEISV,
                    "Cell ID" : self.out.gid,
                    "TAC" : self.out.tac,
                    "EMM state" : self.out.emm_state,
                    "EMM substate" : self.out.emm_substate,                   
                    "EMM cause" : self.out.emm_cause,
                    "paging_record_number" : self.out.paging_record_number,
                    "label" : self.out.label,
                    "attack_type" : self.out.attack_type
                    }
        
            df = pd.DataFrame(dict)
            df.loc[: "EMM substate"].replace("EMM sub-state = 7", "EMM_sub-state=7", inplace=True)
            df.to_csv(self.filepath, index=False)

class SpyFeatures():
    def __init__(self):
        self.message_name = ""
        self.IMSI_attach = 0
        self.null_encryption = 0
        self.enable_IMEISV = 0
        self.gid = 0
        self.tac = 0
        self.emm_state = ""
        self.emm_substate = ""
        self.emm_cause = "0"
        self.paging_record_number = "0"
        self.label = 0
        self.attack_type = 0

class SpyFeauresList():
    def __init__(self):
        self.message_name = []
        self.IMSI_attach = []
        self.null_encryption = []
        self.enable_IMEISV = []
        self.gid = []
        self.tac = []
        self.emm_state = []
        self.emm_substate = []
        self.emm_cause = []
        self.paging_record_number = []
        self.label = []
        self.attack_type = []    

    def append(self, features):
        self.message_name.append(features.message_name)
        self.IMSI_attach.append(features.IMSI_attach)
        self.null_encryption.append(features.null_encryption)
        self.enable_IMEISV.append(features.enable_IMEISV)
        self.gid.append(features.gid)
        self.tac.append(features.tac)
        self.emm_state.append(features.emm_state)
        self.emm_substate.append(features.emm_substate)
        self.emm_cause.append(features.emm_cause)
        self.paging_record_number.append(features.paging_record_number)
        self.label.append(features.label)
        self.attack_type.append(features.attack_type)