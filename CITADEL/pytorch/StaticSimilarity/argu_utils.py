import os


class Arguments(object):  
   def __init__(self, nam, typ): 
        self.name = nam
        self.type = set()
        if typ!=None:
            self.type.add(typ)