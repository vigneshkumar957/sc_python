from torch.utils.data import Dataset

class SkinCancerDataset(Dataset):

    def __init__(me, dataimage, labels, transforms):
        me.dataimage = dataimage
        me.labels = labels
        me.transforms = transforms

    def __len__(me):
        return len(me.dataimage)

    def __getitem__(me, index):
        return me.transforms(me.dataimage[index]), me.labels[index]