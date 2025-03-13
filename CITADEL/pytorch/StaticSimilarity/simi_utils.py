import textdistance
import numpy as np
from munkres import Munkres, print_matrix



def arg_similar(def_a,def_b):
    # def_a and def_b are class of one single argument, have attributes: name, type
    
    def string_similarity(str1, str2):
        # evaluate the similarity between strings (e.g., name)
        return textdistance.levenshtein.normalized_similarity(str1, str2)

    def type_similarity(set1,set2):
        # evaluate the similarity between sets (e.g., arguments sets)
        if len(set1) == 0:
            type_sim = 0
        else:
            type_sim = len(set1.intersection(set2)) / len(set1)
        return type_sim
    
    name_sim = string_similarity(def_a.name,def_b.name)
    type_sim = type_similarity(def_a.type,def_b.type)
    # print(type_sim)
    return name_sim + type_sim 



def similarity(argdefs_a,argdefs_b):
    # argdefs_a and argdefs_b are list of single arguments.
    # each element in them are a class initialized with two attributes: name and type.
    sim = []
    for def_a in argdefs_a:
        temp = []
        for def_b in argdefs_b:
            t = arg_similar(def_a,def_b)
            temp.append(t)
        sim.append(temp)
    return sim

def match_arguments(args_A,args_B):
    # args_A and args_b need to process to be list of single arguments.
    # each element in them are a class initialized with two attributes: name and type.
    
    sim = similarity(args_A, args_B)
    sim_matrix = [[5 - y for y in x] for x in sim]
    m = Munkres()
    try:
        indices = m.compute(sim_matrix)
        indices.sort(key=lambda x: x[1])
    except:
        indices=[]
    return indices
    # need to read paper to know how to use this. 