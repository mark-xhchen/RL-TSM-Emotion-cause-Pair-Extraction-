import os


class ECPEDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data(self.filepath)
        
    def load_data(self, datapath):
        ret = []

        lines = open(datapath, 'r', encoding='utf-8').readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == '': 
                break

            instance = {}
            texts, sen_len = [], []
            
            firstline = lines[i].strip().split()
            instance['doc_id'] = firstline[0]
            d_len = int(firstline[1])
            instance['doc_len'] = d_len
            pairs = eval('[' + lines[i+1].strip() + ']')

            emo_tags = [0 for _ in range(d_len)]
            cau_tags = {}
            for pair in pairs:
                emo_tags[pair[0]-1] = 1
                if pair[0]-1 not in cau_tags:
                    cau_tags[pair[0]-1] = [0 for _ in range(d_len)]
                cau_tags[pair[0]-1][pair[1]-1] = 1

            for j in range(d_len):
                words = lines[i+j+2].strip().split(',')[-1]
                sen_len.append(len(words.split()))
                texts.append(words)

            instance['texts'] = texts
            instance['sen_len'] = sen_len
            instance['emo_tags'] = emo_tags
            instance['cau_tags'] = cau_tags

            ret.append(instance)

            i += d_len + 2

        return ret
