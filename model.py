import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, AlbertModel
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel


class EmoModel(nn.Module):
    def __init__(self, hiddendim, statedim, elnum):
        super(EmoModel, self).__init__()
        self.hid2state = nn.Linear(hiddendim + statedim, statedim)
        self.state2prob = nn.Linear(statedim, elnum)
        # # if we want to further refine the label to the emotion type, then use:
        # self.state2prob = nn.Linear(statedim, elnum+1)

    def forward(self, sen_vec, prev_state_vec, training, dropout):
        inp = torch.cat([sen_vec, prev_state_vec], dim=1)
        outp = F.dropout(torch.tanh(self.hid2state(inp)), p=dropout, training=training)
        prob = F.softmax(self.state2prob(outp), dim=1)
        return outp, prob


class CauModel(nn.Module):
    def __init__(self, hiddendim, statedim):
        super(CauModel, self).__init__()
        self.hid2state = nn.Linear(hiddendim * 2 + statedim + 50, statedim)
        self.state2prob = nn.Linear(statedim, 2)

    def forward(self, pair_vec, prev_state_vec, pos_vec, training, dropout):
        inp = torch.cat([pair_vec, prev_state_vec, pos_vec], dim=1)
        outp = F.dropout(torch.tanh(self.hid2state(inp)), p=dropout, training=training)
        prob = F.softmax(self.state2prob(outp), dim=1)
        return outp, prob


class Model(nn.Module):
    def __init__(self, elnum, args):
        super(Model, self).__init__()
        # self.hiddendim = hiddendim
        self.statedim = args.statedim
        # self.elnum = elnum

        if args.pretrainedLM == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.albert = AutoModel.from_pretrained("bert-base-chinese")
            wordemdsize = self.albert.pooler.dense.in_features
        else:
            self.tokenizer = BertTokenizer.from_pretrained('clue/albert_chinese_tiny')
            self.albert = AlbertModel.from_pretrained('clue/albert_chinese_tiny')
            wordemdsize = self.albert.pooler.in_features

        self.top2bot = nn.Linear(args.statedim, args.statedim)
        self.posvector = nn.Embedding(args.max_doc_len, 50)
        self.bot2top = nn.Linear(args.statedim, args.statedim)

        self.wordencmodule = nn.Transformer(d_model=wordemdsize, batch_first=True)
        
        self.clsmode = args.clsmode
        if args.clsmode == 'transformer':
            self.clsencmodule = nn.Transformer(d_model=wordemdsize)
            self.pairencmodule = nn.Transformer(d_model=wordemdsize)
            self.topModel = EmoModel(wordemdsize, args.statedim, elnum)
            self.cauModel = CauModel(wordemdsize, args.statedim)
        else:
            self.clsencmodule = nn.LSTM(input_size=wordemdsize, hidden_size=args.hiddendim, bidirectional=True)
            self.pairencmodule = nn.LSTM(input_size=4*args.hiddendim, hidden_size=2*args.hiddendim, bidirectional=True)
            self.topModel = EmoModel(2*args.hiddendim, args.statedim, elnum)
            self.cauModel = CauModel(2*args.hiddendim, args.statedim)

        self.dropout = args.dropout


    def clsenc(self, input_cls):
        return self.clsencmodule(input_cls, input_cls) if self.clsmode == 'transformer' else self.clsencmodule(input_cls)[0]

    def pairenc(self, input_pair):
        return self.pairencmodule(input_pair, input_pair) if self.clsmode == 'transformer' else self.pairencmodule(input_pair)[0]

    def sample(self, prob, training, pre_gttags, position, device):
        if not training:
            # testing
            return torch.argmax(prob, 1)
        elif pre_gttags is not None:
            # pre-training
            return torch.LongTensor(1, ).fill_(pre_gttags[position]).to(device)
        else:
            # RL training
            return torch.squeeze(torch.multinomial(prob, 1), dim=0).to(device)


    def forward(self, test, input_text, pre_emotags=None, pre_cautags=None, device=torch.device("cpu")):
        emo_tags, emo_actprobs, cau_tags, cau_actprobs = [[] for _ in range(4)]
        #-----------------------------------------------------------------
        # BERT encoding for high-level process
        encoded_text = self.tokenizer(input_text, padding='longest', return_tensors='pt').to(device)
        
        doclen = len(input_text)
        if not test:
            self.albert.train()
        else:
            self.albert.eval()
        # [doclen, longest_senlen, hidden_dim]
        wordemb = self.albert(**encoded_text).last_hidden_state

        # if we keep the first word of every sentence
        clsemb = torch.unsqueeze(wordemb[:,0,:], dim=1)  # [doclen, 1, hidden_dim]
        clsemb = self.clsenc(clsemb)
        
        #------------------------------------------------------------------
        # First Layer
        state = torch.FloatTensor(1, self.statedim, ).fill_(0).to(device)
        action = torch.LongTensor(1, ).fill_(0).to(device)
        for x in range(doclen):
            state, prob = self.topModel(clsemb[x], state, bool(1-test), self.dropout)
            action = self.sample(prob, bool(1-test), pre_emotags, x, device)
            # get sentiment probability for chosen sentiment
            actprob = prob[0][action]
            emo_tags.append(action.cpu().data[0])
            if test:
                emo_actprobs.append(actprob.cpu().data[0])
            else:
                emo_actprobs.append(actprob)

            # Second Layer, see if the current clause is an emotional clause
            if action.data[0] > 0:
                botstate = self.top2bot(state)

                tmpaction = torch.LongTensor(1, ).fill_(0).to(device)
                cau_actions, cau_actprob = [], []
                
                paircls = torch.stack([torch.cat([clsemb[x], clsemb[y]], dim=1) for y in range(doclen)]) # [doclen, 1, clsemd * 2]
                paircls = self.pairenc(paircls)

                for y in range(doclen):
                    pos_vec = self.posvector(torch.LongTensor(1, ).fill_(abs(y-x)).to(device))
                    botstate, cauprob = self.cauModel(paircls[y], botstate, pos_vec, bool(1-test), self.dropout)
                    tmpaction = self.sample(cauprob, bool(1-test), pre_cautags[x] if pre_cautags is not None else None, y, device)
                    cau_actions.append(tmpaction.cpu().data[0])
                    cauprobb = cauprob[0][tmpaction]
                    del tmpaction
                    if test:
                        cau_actprob.append(cauprobb.cpu().data[0])
                    else:
                        cau_actprob.append(cauprobb)

                cau_tags.append(cau_actions)
                cau_actprobs.append(cau_actprob)

                state = self.bot2top(botstate)
                
        return emo_tags, emo_actprobs, cau_tags, cau_actprobs