
import torch 
import os
from tqdm import tqdm
import pandas as pd
import torch.optim as optim
import numpy as np
from model import ResNet
from utils import LeaveDataset,load_data
from args import get_test_args
from torch.utils.data.sampler import SequentialSampler

def test(model, data_loader, batch_size, device):
	
	model.eval()
	with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as pbar:
		predictions = []
		for img in data_loader:
			img = img.to(device)
			y_hat = model(img)
			pred = torch.argmax(y_hat, dim=1)

			predictions.extend(pred.tolist())
			pbar.update(batch_size)
      
	return predictions


if __name__ == "__main__":

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	args = get_test_args()

	test_dataset = LeaveDataset(
	    csv_path=args.test_path, 
	    file_path=args.image_path,
	    mode="test"
	)

	train_dataset = LeaveDataset(
	    csv_path=args.train_path, 
	    file_path=args.image_path,
	    mode="train"
	)

	pre_trained_path = os.path.join(args.save_dir, "checkpoint_15.pt")
	n_class = len(train_dataset)
	model = ResNet(n_class).to(device)

	test_loader = load_data(
		test_dataset, args.batch_size, 0, SequentialSampler(test_dataset))

	checkpoint = torch.load(pre_trained_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	predictions = test(model, test_loader, args.batch_size, device)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	output_path = os.path.join(args.output_dir, "predictions.csv")
	predictions = pd.DataFrame(predictions)
	predictions.to_csv(output_path)


