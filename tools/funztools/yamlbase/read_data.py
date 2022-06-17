import os,sys
import yaml

class ReadData:
    def __init__(self, gold):
        self.gold = gold
        self.path = os.getcwd() + os.sep + "para_data" + os.sep + self.gold
        
    def loading_data(self):
        with open(self.path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data    