import numpy as np
import struct
import pickle
import json
def readImageFeatures(path,subset=None):
    map = dict()
    with open(path, 'rb') as f:
        while True:
            itemId = f.read(8)
            if itemId == b'': break
            if subset is not None and int(itemId) not in subset:
                f.seek(4*4096,1)
            else:
                try:
                    feature = struct.unpack('f'*4096, f.read(4*4096))
                except:
                    break
                map[int(itemId)] = feature
                print(int(itemId))
    return map
            
if __name__ == "__main__":
    k= open("data2.json","r")
    subset = set([int(s) for s in json.load(k).keys()])
    k.close()
    map = readImageFeatures("./Behance_Image_Features.b",subset)
    print(map)
    with open("embeddings.pickle","wb+") as h:
        pickle.dump(map,h)
    with open("embeddings.pickle","rb") as h:
        map = pickle.load(h)
        assert len(map.keys())==len(set(map.keys()).intersection(subset))