from __future__ import print_function

import argparse
import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.datasets.utils as utils
import torchvision.transforms as transforms

import glob

from PIL import Image
from typing import Any, Callable, Optional

class TextureDataset(dset.VisionDataset):

	url = "https://github.com/adnphllps/adnphllps.github.io/blob/master/datasets/texture_256_ds.tar.gz?raw=true"
	filename = "texture_256_ds.tar.gz"
	tgz_md5 = "b001a3d63e23633f75f50185c065eb6f"

	def __init__(
		self,
		root: str,
		train: bool = True,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = False,
	) -> None:

		super(TextureDataset, self).__init__(root, transform=transform,
									  target_transform=target_transform)
		
		self.download_location = './'
		self.root = root
		self.train = train

		if download:
			self.download()

		self.data: Any = []

		for file_path in glob.iglob(str(self.root) + '/**/*.jpg', recursive=True):
			self.data.append([file_path])

		print('Loaded {0} files from the dataset'.format(len(self.data)))


	def __getitem__(self, index: int): #-> Tuple[Any, Any]:
 
		img = self.data[index]

		img_path = self.data[index]

		img = Image.open(img_path[0])

		if self.transform is not None:
			img = self.transform(img)

		return img 

	
	def __len__(self) -> int:
		return len(self.data)
	
	def download(self) -> None:

		if os.path.isfile(self.download_location + '/' + self.filename):
			print('Found files at location specified, skipping download.')
			return
		
		utils.download_and_extract_archive(self.url, download_root=self.download_location, extract_root=self.root, filename=self.filename)

		if not utils.check_integrity(self.download_location + '/' + self.filename, self.tgz_md5):
			print('MD5 Checksums do not match. Verify download location and check for an updated version of this file.')
		else:
			print('Dataset downloaded and verified')


#Test
if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataroot', default='./data', required=False, help='path to dataset')

	opt = parser.parse_args()

	dataset = TextureDataset(root=opt.dataroot, download=True,
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))

	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
										 shuffle=True, num_workers=int(1))

	print("Success!")




