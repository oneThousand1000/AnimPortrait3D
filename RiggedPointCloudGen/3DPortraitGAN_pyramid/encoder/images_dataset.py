from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, target_root, opts, target_transform=None):
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		

		return to_im