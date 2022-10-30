import argparse


class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
        parser.add_argument('--epochPRE', type=int, default=50, help="Number of epoch on pretraining")
        parser.add_argument('--epochRL', type=int, default=30, help="Number of epoch on training with RL")
        parser.add_argument('--batchsize', type=int, default=32, help="Batch size on training")
        parser.add_argument('--dropout', type=float, default=0.5, help="Dropout")
        
        parser.add_argument('--hiddendim', type=int, default=200, help="Dimension of embeddings")
        parser.add_argument('--statedim', type=int, default=200, help="Dimension of state")
        parser.add_argument('--sampleround', type=int, default=1, help="Sample round in RL")
        parser.add_argument('--clsmode', type=str, default='bilstm', help='transformer or bilstm')
        parser.add_argument('--max_doc_len', type=int, default=75, help='max_doc_len')

        parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
        parser.add_argument('--ckptpath', type=str, default='', help="path to pretrained model")
        parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
        parser.add_argument('--datadir', type=str, default='./data/', help="Data directory")
        parser.add_argument('--seed', type=int, default=1, help="PyTorch seed value")
        parser.add_argument('--testfile', type=str, default='', help='path for a specific test file')
        parser.add_argument('--pretrainedLM', type=str, default='bert', help='name of pretrained LM')
        return parser
