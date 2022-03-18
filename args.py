
import argparse


def get_train_valid_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--num-epochs', type=int, default=20)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--beta1', type=float, default=.9)
	parser.add_argument('--beta2', type=float, default=.999)
	parser.add_argument('--decay', type=float, default=.001)
	parser.add_argument('--eps', type=float, default=1e-7)
	parser.add_argument('--save-dir', type=str, default='save/')
	parser.add_argument('--image-path', type=str, default="data/")
	parser.add_argument('--train-path', type=str, default="data/train.csv")
	parser.add_argument('--lr-decay', type=float, default=.05)
	parser.add_argument('--lr-period', type=int, default=20)
	args = parser.parse_args()
	return args

def get_test_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--save-dir', type=str, default='save/')
	parser.add_argument('--image-path', type=str, default="data/")
	parser.add_argument('--test-path', type=str, default="data/test.csv")
	parser.add_argument('--output-dir', type=str, default="output/")
	args = parser.parse_args()
	return args