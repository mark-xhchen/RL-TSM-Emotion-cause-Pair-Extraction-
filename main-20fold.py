import random, sys, time, os
import torch
from torch import optim
from dataloader import ECPEDataset
from model import Model
from Parser import Parser
from utils import run_for_batch
from metrics import calcF1
import datetime
import json
import numpy as np
from tqdm import tqdm


def run_for_fold(lr, fold, train_data, dev_data, test_data, model, args, sampleround, epoch, device, experiment_id, pretrain, best_pretrain_dev_metric=None, best_pretrain_test_metric=None):
    best_dev_metric = best_pretrain_dev_metric
    best_dev_test_metric = best_pretrain_test_metric
    log_f = open("checkpoints/" + experiment_id + "/log.txt", 'a')
    log_f.write("Fold %d begins:\n" % fold)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for e in range(epoch):
        #random.shuffle(train_data)
        print("Fold %d Train epoch %d" % (fold, e))
        log_f.write("Fold %d Train epoch %d\n" % (fold, e))

        # training
        model.train()
        trainloss = 0.
        batchcnt = (len(train_data) - 1) // args.batchsize + 1
        tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]
        with tqdm(total=batchcnt) as t:
            for b in range(batchcnt):
                start = time.time()
                data = train_data[b * args.batchsize : (b+1) * args.batchsize]
                # optimizer.zero_grad()
                batch_tp, batch_fp, batch_fn, _ = run_for_batch(optimizer, model, data, sampleround, device, pretrain, False)
                tp += batch_tp
                fp += batch_fp 
                fn += batch_fn
                # optimizer.step()
                torch.cuda.empty_cache()
                t.update(1)

        model.eval()
        with torch.no_grad():
            # validation
            batchcnt = (len(dev_data) - 1) // args.batchsize + 1
            tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]
            with tqdm(total=batchcnt) as t:
                for b in range(batchcnt):
                    data = dev_data[b * args.batchsize : (b+1) * args.batchsize]
                    batch_tp, batch_fp, batch_fn, _ = run_for_batch(None, model, data, 1, device, False, True)
                    tp += batch_tp
                    fp += batch_fp 
                    fn += batch_fn
                    t.update(1)
            dev_metrics = [calcF1(tp[i], fp[i], fn[i]) for i in range(3)]

            # testing
            batchcnt = (len(test_data) - 1) // args.batchsize + 1
            tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]
            with tqdm(total=batchcnt) as t:
                for b in range(batchcnt):
                    data = test_data[b * args.batchsize : (b+1) * args.batchsize]
                    batch_tp, batch_fp, batch_fn, _ = run_for_batch(None, model, data, 1, device, False, True)
                    tp += batch_tp
                    fp += batch_fp 
                    fn += batch_fn
                    t.update(1)
            test_metrics = [calcF1(tp[i], fp[i], fn[i]) for i in range(3)]

            # save stats and model
            print("Epoch", e, "validation emo-F1", dev_metrics[0][-1], "cau-F1", dev_metrics[1][-1], "pair-F1", dev_metrics[2][-1])
            log_f.write("Epoch {}, validation emo-F1 {} cau-F1 {} pair-F1 {}\n" .format(e, dev_metrics[0][-1], dev_metrics[1][-1], dev_metrics[2][-1]))
            print("Epoch", e, "test emo-F1", test_metrics[0][-1], "cau-F1", test_metrics[1][-1], "pair-F1", test_metrics[2][-1])
            log_f.write("Epoch {}, test emo-F1 {} cau-F1 {} pair-F1 {}\n" .format(e, test_metrics[0][-1], test_metrics[1][-1], test_metrics[2][-1]))
            
            if best_dev_metric is None or dev_metrics[2][-1] > best_dev_metric[-1]:
                torch.save(model, "checkpoints/{}/fold_bestmodel".format(experiment_id))
                print("Fold", fold, "Best dev result updated in Epoch", e, "\n")
                log_f.write("Fold {} Best dev result updated in Epoch {}\n\n".format(fold, e))
                best_dev_metric = dev_metrics[2]
                best_dev_test_metric = test_metrics

    log_f.close()

    return best_dev_metric, best_dev_test_metric


if __name__ == "__main__":
    # get args
    argv = sys.argv[1:]
    parser = Parser().getParser()
    args, _ = parser.parse_known_args(argv)

    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_id = str(experiment_id)

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    avg_fn = lambda x, index: sum([z[index] for z in x]) / len(x)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.test:
        test_data = ECPEDataset(args.testfile).data
        print("#test_data:", len(test_data))

        assert args.ckptpath != '', 'checkpoints path should not be empty under testing mode'
        assert 'model' in '', 'under testing mode, the path should be specified to which model to load'
        print("loading model", args.ckptpath)
        pretrain_model = torch.load(args.ckptpath, map_location='cpu') 
        model_dict = model.state_dict()
        pretrained_dict = pretrain_model.state_dict() 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

        model.eval()
        with torch.no_grad():
        # testing
            batchcnt = (len(test_data) - 1) // args.batchsize + 1
            tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]
            with tqdm(total=batchcnt) as t:
                for b in range(batchcnt):
                    data = test_data[b * args.batchsize : (b+1) * args.batchsize]
                    batch_tp, batch_fp, batch_fn, _ = run_for_batch(optimizer, model, data, 1, device, False, True)
                    tp += batch_tp
                    fp += batch_fp 
                    fn += batch_fn
                    t.update(1)
            metrics = [calcF1(tp[i], fp[i], fn[i]) for i in range(3)]

        print("test emo-F1", metrics[0][-1], "cau-F1", metrics[1][-1], "pair-F1", metrics[2][-1])  
    
    else:
        fold_dev_metric = []
        fold_emo_test_metric = []
        fold_cau_test_metric = []
        fold_pair_test_metric = []
        fold_pretrain_dev_metric = []
        fold_pretrain_emo_metric = []
        fold_pretrain_cau_metric = []
        fold_pretrain_pair_metric = []
        all_data = ECPEDataset(os.path.join(args.datadir, "all_data_pair.txt")).data
        for fold in range(1, 21):
            print("Loading data for fold", fold)

            doc_mark = {}
            test_data = []
            while len(test_data) < (len(all_data)) // 10:
                dataid = np.random.randint(len(all_data))
                while dataid in doc_mark:
                    dataid = np.random.randint(len(all_data))
                doc_mark[dataid] = 1
                test_data.append(all_data[dataid])
            
            dev_data = []
            while len(dev_data) < (len(all_data)) // 10:
                dataid = np.random.randint(len(all_data))
                while dataid in doc_mark:
                    dataid = np.random.randint(len(all_data))
                doc_mark[dataid] = 1
                dev_data.append(all_data[dataid])

            train_data = []
            for i in range(len(all_data)):
                if i not in doc_mark:
                    train_data.append(all_data[i])

            print("#train_data:", len(train_data))
            print("#dev_data:", len(dev_data))
            print("#test_data:", len(test_data))

            if not os.path.exists('checkpoints'):
                 os.mkdir('checkpoints')
                
            model = Model(2, args)
            model.to(device)

            if args.pretrain:
                experiment_id_pretrain = experiment_id + "_pretrain"
                if not os.path.exists('checkpoints/{}'.format(experiment_id_pretrain)):
                    os.mkdir('checkpoints/{}'.format(experiment_id_pretrain))
                with open('checkpoints/{}/args.json'.format(experiment_id_pretrain), 'w') as f:
                    json.dump(vars(args), f)
                
                pretrain_dev, pretrain_metric = run_for_fold(args.lr, fold, train_data, dev_data, test_data, model, args, 1, args.epochPRE, device, experiment_id_pretrain, True)

                print("Fold {} pretrain finished.".format(fold))

                fold_pretrain_dev_metric.append(pretrain_dev)
                fold_pretrain_emo_metric.append(pretrain_metric[0])
                fold_pretrain_cau_metric.append(pretrain_metric[1])
                fold_pretrain_pair_metric.append(pretrain_metric[2])

                log_f = open("checkpoints/" + experiment_id_pretrain + "/log.txt", 'a')
                log_f.write("#######################################\n")
                log_f.write("Fold %d finished\n" % fold)

                ave_dev = avg_fn(fold_pretrain_dev_metric, 0), avg_fn(fold_pretrain_dev_metric, 1), avg_fn(fold_pretrain_dev_metric, 2)
                ave_emo = avg_fn(fold_pretrain_emo_metric, 0), avg_fn(fold_pretrain_emo_metric, 1), avg_fn(fold_pretrain_emo_metric, 2)
                ave_cau = avg_fn(fold_pretrain_cau_metric, 0), avg_fn(fold_pretrain_cau_metric, 1), avg_fn(fold_pretrain_cau_metric, 2)
                ave_pair = avg_fn(fold_pretrain_pair_metric, 0), avg_fn(fold_pretrain_pair_metric, 1), avg_fn(fold_pretrain_pair_metric, 2)
                
                ave_dev = (fold,) + ave_dev
                ave_emo = (fold,) + ave_emo
                ave_cau = (fold,) + ave_cau
                ave_pair = (fold,) + ave_pair

                log_f.write("%d fold average dev pair result: P %0.4f R %0.4f F %0.4f \n" % ave_dev)
                log_f.write("%d fold average test emo result: P %0.4f R %0.4f F %0.4f \n" % ave_emo)
                log_f.write("%d fold average test cau result: P %0.4f R %0.4f F %0.4f \n" % ave_cau)
                log_f.write("%d fold average test pair result: P %0.4f R %0.4f F %0.4f \n" % ave_pair)

                log_f.close()
                
                print("%d fold average test emo result: P %0.4f R %0.4f F %0.4f" % ave_emo)
                print("%d fold average test cau result: P %0.4f R %0.4f F %0.4f" % ave_cau)
                print("%d fold average test pair result: P %0.4f R %0.4f F %0.4f" % ave_pair)

                ckptpath = 'checkpoints/{}/fold_bestmodel'.format(experiment_id_pretrain)
                if 'fold' not in ckptpath:
                    ckptpath = os.path.join(ckptpath, 'fold_bestmodel')
                    assert os.path.exists(ckptpath), "the predefined load path for fold doesn't exist".format(fold)

                print("Loading pretrained model:", ckptpath)
                pretrain_model = torch.load(ckptpath, map_location='cpu') 
                model_dict = model.state_dict()
                pretrained_dict = pretrain_model.state_dict() 
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
                model_dict.update(pretrained_dict) 
                model.load_state_dict(model_dict)
            
            if not args.pretrain and args.ckptpath != '':
                ckptpath = args.ckptpath if 'model' in args.ckptpath else os.path.join(args.ckptpath, 'fold_bestmodel')
                print(ckptpath)
                pretrain_model = torch.load(ckptpath, map_location='cpu') 
                model_dict = model.state_dict()
                pretrained_dict = pretrain_model.state_dict() 
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
                model_dict.update(pretrained_dict) 
                model.load_state_dict(model_dict)

                model.eval()
                with torch.no_grad():
                    # validation
                    batchcnt = (len(dev_data) - 1) // args.batchsize + 1
                    tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]
                    with tqdm(total=batchcnt) as t:
                        for b in range(batchcnt):
                            data = dev_data[b * args.batchsize : (b+1) * args.batchsize]
                            batch_tp, batch_fp, batch_fn, _ = run_for_batch(None, model, data, 1, device, False, True)
                            tp += batch_tp
                            fp += batch_fp 
                            fn += batch_fn
                            t.update(1)
                    metrics = [calcF1(tp[i], fp[i], fn[i]) for i in range(3)]

                    fold_pretrain_dev_metric.append(metrics[2])

                    # testing
                    batchcnt = (len(test_data) - 1) // args.batchsize + 1
                    tp, fp, fn = [np.array([0, 0, 0]) for _ in range(3)]
                    with tqdm(total=batchcnt) as t:
                        for b in range(batchcnt):
                            data = test_data[b * args.batchsize : (b+1) * args.batchsize]
                            batch_tp, batch_fp, batch_fn, _ = run_for_batch(None, model, data, 1, device, False, True)
                            tp += batch_tp
                            fp += batch_fp 
                            fn += batch_fn
                            t.update(1)
                    metrics = [calcF1(tp[i], fp[i], fn[i]) for i in range(3)]

                    fold_pretrain_emo_metric.append(metrics[0])
                    fold_pretrain_cau_metric.append(metrics[1])
                    fold_pretrain_pair_metric.append(metrics[2])
                    # print(metrics)
                    print("fold {} pair-F1 after pretrained: {}".format(fold, metrics[2][-1]))


            experiment_id_train = experiment_id + "_train"
            if not os.path.exists('checkpoints/{}'.format(experiment_id_train)):
                os.mkdir('checkpoints/{}'.format(experiment_id_train))
            with open('checkpoints/{}/args.json'.format(experiment_id_train), 'w') as f:
                json.dump(vars(args), f)

            best_dev_metric, best_test_metric = run_for_fold(args.lr / 10, fold, train_data, dev_data, test_data, model, args, args.sampleround, args.epochRL, device, experiment_id_train, False, fold_pretrain_dev_metric[-1] if len(fold_pretrain_dev_metric) > 0 else None, [fold_pretrain_emo_metric[-1], fold_pretrain_cau_metric[-1], fold_pretrain_pair_metric[-1]] if len(fold_pretrain_pair_metric) > 0 else None)

            fold_dev_metric.append(best_dev_metric)
            fold_emo_test_metric.append(best_test_metric[0])
            fold_cau_test_metric.append(best_test_metric[1])
            fold_pair_test_metric.append(best_test_metric[2])

            ave_dev = avg_fn(fold_dev_metric, 0), avg_fn(fold_dev_metric, 1), avg_fn(fold_dev_metric, 2)
            ave_emo = avg_fn(fold_emo_test_metric, 0), avg_fn(fold_emo_test_metric, 1), avg_fn(fold_emo_test_metric, 2)
            ave_cau = avg_fn(fold_cau_test_metric, 0), avg_fn(fold_cau_test_metric, 1), avg_fn(fold_cau_test_metric, 2)
            ave_pair = avg_fn(fold_pair_test_metric, 0), avg_fn(fold_pair_test_metric, 1), avg_fn(fold_pair_test_metric, 2)

            ave_dev = (fold,) + ave_dev
            ave_emo = (fold,) + ave_emo
            ave_cau = (fold,) + ave_cau
            ave_pair = (fold,) + ave_pair

            print("Fold %d training finished" % fold)
            print("%d fold average dev result: P %0.4f R %0.4f F %0.4f" % ave_dev)
            print("%d fold average test emo result: P %0.4f R %0.4f F %0.4f" % ave_emo)
            print("%d fold average test cau result: P %0.4f R %0.4f F %0.4f" % ave_cau)
            print("%d fold average test pair result: P %0.4f R %0.4f F %0.4f" % ave_pair)

            log_f = open("checkpoints/" + experiment_id_train + "/log.txt", 'a')
            log_f.write("#######################################\n")
            log_f.write("Fold %d finished\n" % fold)
            log_f.write("%d fold average dev result: P %0.4f R %0.4f F %0.4f \n" % ave_dev)
            log_f.write("%d fold average test emo result: P %0.4f R %0.4f F %0.4f \n" % ave_emo)
            log_f.write("%d fold average test cau result: P %0.4f R %0.4f F %0.4f \n" % ave_cau)
            log_f.write("%d fold average test pair result: P %0.4f R %0.4f F %0.4f \n" % ave_pair)
            log_f.close()

        print("#################################")
        print("Final results")
        print("%d fold average dev result: P %0.4f R %0.4f F %0.4f" % ave_dev)
        print("%d fold average test emo result: P %0.4f R %0.4f F %0.4f" % ave_emo)
        print("%d fold average test cau result: P %0.4f R %0.4f F %0.4f" % ave_cau)
        print("%d fold average test pair result: P %0.4f R %0.4f F %0.4f" % ave_pair)
        for i in range(len(fold_pair_test_metric)):
            print("Fold {} Pair-F1: {}".format(i+1, fold_pair_test_metric[i][2]))
