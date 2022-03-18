from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import LeaveDataset, load_data, evaluate
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from args import get_train_valid_args
from model import ResNet
import torch
import torch.nn as nn
import os
import numpy as np

# def init_weights(m):
#     if type(m) in [nn.Linear, nn.Conv2d]:
#     	nn.init.xavier_uniform_(m.weight)

def train(model, 
          data_loader, 
          optimizer, 
          scheduler,
          loss_fn, 
          device, 
          batch_size,
          epoch,
          tbx):

    print(f'start training...')
    model.train()

    with torch.enable_grad(), tqdm(total=len(data_loader.dataset)) as pbar:
        losses = []
        for batch in data_loader:
            optimizer.zero_grad()
            img, label = batch

            img, label = img.to(device), label.to(device)
            y_hat = model(img)
            loss = loss_fn(y_hat, label)

            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.update(batch_size)
            pbar.set_postfix(epoch=epoch, loss=loss.item())
            losses.append(loss.item())
        pbar.set_postfix(epoch=epoch, loss=np.mean(losses))
        tbx.add_scalar('training loss', np.mean(losses), epoch)

def valid(model,
		  data_loader,
		  device,
		  loss_fn,
		  batch_size,
		  epoch):

	print(f'start validation...')
	model.eval()

	with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as pbar:
		accs = []
		for batch in data_loader:
			img, label = batch
			img, label = img.to(device), label.to(device)
			y_hat = model(img)
			pred = torch.argmax(y_hat, dim=1)

			acc = (pred == label).float().mean()
			pbar.update(batch_size)
			accs.append(acc)
		
		pbar.set_postfix(epoch=epoch, acc=sum(accs)/len(accs))
	return acc

def save(model, save_dir, epoch, optimizer, best_score):

	if os.path.isfile(save_dir):
		print(f"Provided path ({save_dir}) should be a directory, not a file")
		return

	file = os.path.join(save_dir, f"checkpoint_{epoch}.pt") 
	torch.save({
		'epoch': epoch,
        'best_score': best_score,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
		}, file)

if __name__ == "__main__":

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	args = get_train_valid_args()
	start_epoch = -1
	tbx = SummaryWriter(args.save_dir)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	train_dataset = LeaveDataset(
	    csv_path=args.train_path, 
	    file_path=args.image_path,
	    mode="train"
	)
	valid_dataset =  LeaveDataset(
	    csv_path=args.train_path, 
	    file_path=args.image_path,
	    mode="valid"
	)

	n_class = len(train_dataset)
	model = ResNet(n_class).to(device)
	# model.apply(init_weights)
	params = filter(lambda param: param.requires_grad, model.parameters())
	optimizer = optim.AdamW(
		params=params, lr=args.lr, betas=(args.beta1, args.beta2),\
		weight_decay=args.decay, eps=args.eps)
	scheduler = optim.lr_scheduler.StepLR(optimizer, 
		args.lr_period, args.lr_decay)
	loss_fn = nn.CrossEntropyLoss()

	train_loader = load_data(
		train_dataset, args.batch_size, 0, RandomSampler(train_dataset))
	valid_loader = load_data(
		valid_dataset, args.batch_size, 0, SequentialSampler(valid_dataset))

	best_score = -1

	pre_trained_path = os.path.join(args.save_dir, "checkpoint_10.pt")
	if os.path.exists(pre_trained_path):
		print("loading pretrained model...")
		checkpoint = torch.load(pre_trained_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		best_scores = checkpoint['best_score']
		start_epoch = checkpoint['epoch']

	for epoch in range(start_epoch + 1, args.num_epochs):
		train(model, train_loader, optimizer, scheduler, loss_fn,
			device, args.batch_size, epoch, tbx)
		# if epoch % 3 == 0 and epoch > 0:
		acc = valid(model, valid_loader, device, loss_fn, args.batch_size, epoch)
		tbx.add_scalar('Accuracy', acc, epoch)

		if acc > best_score:
			best_score = acc
			save(model, args.save_dir, epoch, optimizer, best_score)
