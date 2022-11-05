import torch
from tqdm import tqdm
import random
import sys



class recSysDataset(torch.utils.data.Dataset):
    """
    when used correctly, dataloader will return a list containing 2 tensors
    first being the sequences of shape (batch_size, max_len)
    second being the user's number of sequences of shape (batch_size)
    """
    def __init__(self, root="D:\\LSTM-Based-Art-Recommendation\\data\\kcore_5_collated.txt", max_len=35, users=-1):
        self.sequences = []
        self.counts = []
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device="gpu"
        with open(root, 'r') as f:
            self.idx_to_item = f.readline().strip().split(",")
            self.item_to_idx = {v: i for i, v in enumerate(self.idx_to_item)}

            for idx, l in tqdm(enumerate(f)):
                if users != -1 and idx >= users:
                    break

                split_seq = l.strip().split(",")
                # for i_ in split_seq:
                #     self.allitemsSadd(int(i_))
                if len(split_seq) < max_len-1:
                    self.sequences += [[1] + [int(i) for i in split_seq] + [0]*(max_len-len(split_seq)-1)]
                    self.counts += [1]
                else:
                    holder = []
                    while len(split_seq) > max_len-1:
                        delimiter = random.randint(2, max_len-1)  # place where random delimiter is selected
                        holder += [[1] + [int(i) for i in split_seq[:delimiter]] + [0]*(max_len-delimiter-1)]
                        split_seq = split_seq[delimiter:]
                    if split_seq:
                        holder += [[1] + [int(i) for i in split_seq] + [0]*(max_len-len(split_seq)-1)]
                    self.sequences += holder
                    self.counts += [len(holder) for _ in holder]
        self.sequences = torch.tensor(self.sequences, dtype=torch.int64).to(self.device)
        self.counts = torch.tensor(self.counts, dtype=torch.int64).to(self.device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.counts[idx]


class train_val_test_split(torch.utils.data.Dataset):
    def __init__(self, dataset, split=(.85, .1, .05), mode="train"):
        self.dataset = dataset
        self.split = split
        self.mode = 0 if mode == "train" else 1 if mode == "val" else 2 if mode == "test" else -1
        self.interval = (round(sum(split[0:self.mode])*len(dataset)), round(sum(split[0:self.mode+1])*len(dataset)))

    def __len__(self):
        return round(len(self.dataset) * self.split[self.mode])

    def __getitem__(self, idx):
        # if idx+self.interval[0] >= self.interval[1]:
        #     raise StopIteration
        return self.dataset[idx + self.interval[0]]


def parse(data="D:\\kcore_5.json"):
    root = data.split(".")[0] + "_collated.txt"
    user_to_items = {}
    item_to_idx = {"<PAD>": 0, "<START>": 1}
    with open(data, 'r') as f:
        for raw_line in tqdm(f, desc="collating"):
            line = eval(raw_line)
            if line["asin"] not in item_to_idx:
                item_to_idx[line["asin"]] = len(item_to_idx)
            if line["reviewerID"] in user_to_items:
                user_to_items[line["reviewerID"]] += [(item_to_idx[line["asin"]], get_time(line))]
            else:
                user_to_items[line["reviewerID"]] = [(item_to_idx[line["asin"]], get_time(line))]
    idx_to_item = list(item_to_idx.keys())

    with open(root, 'w') as f:
        f.write(",".join(idx_to_item) + "\n")

        for idx, items in tqdm(enumerate(user_to_items.values()), desc="writing"):
            if idx+1 < len(user_to_items.values()):
                f.write(",".join(map((lambda x: str(x[0])), sorted(items, key=(lambda z: z[1])))) + "\n")
            else:
                f.write(",".join(map((lambda x: str(x[0])), sorted(items, key=(lambda z: z[1])))))


def get_time(line):
    try:
        return int(line["unixReviewTime"])
    except: #happens very rarely, about 1 in 5 million when month is missing from reviewTime so unixReviewTime is not calculated
        return (int(line["reviewTime"][-4:]) - 1970) * 32140800 #approximate the unix timestamp based on year                

    
# ==========SAMPLE USAGE==========
if __name__ == "__main__":
    dataset = recSysDataset(max_len=20, root = "data\\kcore_5_collated.txt")
    train_data = train_val_test_split(dataset, split=(.5,.45,.05), mode = "train")
    val_data = train_val_test_split(dataset, split=(.5,.45,.05), mode = "val")
    test_data = train_val_test_split(dataset, split=(.5,.45,.05), mode = "test")

    dataloader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True)
    for (item,n) in tqdm(dataloader,file=sys.stdout,total=len(dataloader)):
        tqdm.write(f"{item.size()}, {len(n)}")
