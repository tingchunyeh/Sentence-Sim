from helper import *
from data import *
from models import *
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='NLI training')

parser.add_argument("--data_path", type=str, default='./data', help="path to data")


# model
parser.add_argument("--encoder_type", type=str, default='GRUEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=512, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=256, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--use_cuda", type=bool, default=True, help="True or False")


# train
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.3, help="classifier dropout")
parser.add_argument("--dpout_embed", type=float, default=0.2, help="embed dropout")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate for adam")
parser.add_argument("--last_model", type=str, default="", help="train on last saved model")
parser.add_argument("--saved_model_name", type=str, default="model_new", help="saved model name")

params, _ = parser.parse_known_args()


'''
SEED
'''
np.random.seed(10)
torch.manual_seed(10)


"""
DATA
"""
train, dev, test = get_nl(params.data_path)
wv, default_wv = build_vocab(np.append(train['s1'], train['s2']), "w2v-model.txt")


'''
MODEL
'''

config_nli_model = {
    'n_words'        :  len(wv),
    'word_emb_dim'   :  300,
    'enc_lstm_dim'   :  params.enc_lstm_dim,
    'n_enc_layers'   :  params.n_enc_layers,
    'dpout_model'    :  params.dpout_model,
    'dpout_fc'       :  params.dpout_fc,
    'fc_dim'         :  params.fc_dim,
    'bsize'          :  params.batch_size,
    'n_classes'      :  params.n_classes,
    'pool_type'      :  params.pool_type,
    'encoder_type'   :  params.encoder_type,
    'use_cuda'       :  params.use_cuda,
}

nli_net = NLINet(config_nli_model)
if params.last_model:
    print("load model {}".format(params.last_model))
    nli_net.load_state_dict(torch.load(os.path.join("saved_model", params.last_model)))
print(nli_net)

# loss 
weight = torch.FloatTensor(3).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
from torch import optim
optimizer = optim.Adam(nli_net.parameters(), lr=params.lr)

# cuda 
if params.use_cuda:
    torch.cuda.manual_seed(10)
    torch.cuda.set_device(0)
    nli_net.cuda()
    loss_fn.cuda()


'''
TRAIN
'''
def trainepoch(epoch):
    all_costs = []
    tot_costs = []
    logs = []
    correct = 0.0
    
    nli_net.train()
    permutation = np.random.permutation(len(train['s1']))
    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]
    
    for stidx in tqdm(range(0, len(s1), params.batch_size)):
        s1_batch, s1_len = get_batch(s1[stidx:stidx+params.batch_size], wv, default_wv, params.dpout_embed)
        s2_batch, s2_len = get_batch(s2[stidx:stidx+params.batch_size], wv, default_wv, params.dpout_embed)
        
        if params.use_cuda:
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx+params.batch_size])).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx+params.batch_size]))
        k = s1_batch.size(1)
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        tot_costs.append(loss.item())
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         if len(all_costs) == 100:
#             logs.append('{0} ; loss {1}  ; accuracy train : {2}'.format(stidx, 
#                             round(np.mean(all_costs), 2), round(100.*correct/(stidx+k), 2)))
#             print(logs[-1])
#             all_costs = []
            
    train_acc = round(100 * correct/len(s1), 2)
    train_loss = round(np.mean(tot_costs), 2)
    return train_loss, train_acc    

val_acc_best = -1e10
adam_stop = False
stop_training = False

def evaluate(epoch, eval_type='dev', final_eval=False):
    nli_net.eval()
    correct = 0.0
    global val_acc_best, lr, stop_training, adam_stop
    
    s1 = dev['s1'] if eval_type == 'dev' else test['s1']
    s2 = dev['s2'] if eval_type == 'dev' else test['s2']
    target = dev['label'] if eval_type == 'dev' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], wv, default_wv, params.dpout_embed)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], wv, default_wv, params.dpout_embed)
        
        if params.use_cuda:
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size]))
            
        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :{2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'dev' and eval_acc > val_acc_best:
        with open( os.path.join("saved_model", params.saved_model_name+"_cnofig.pickle" ), 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saving model at epoch {0}'.format(epoch))
        if not os.path.exists("saved_model"): os.makedirs("saved_model")
        torch.save(nli_net.state_dict(), os.path.join("saved_model", params.saved_model_name))
        val_acc_best = eval_acc

    return eval_acc


"""
Train model 
"""
### TRAINING 

train_loss_ls = []
train_acc_ls = []
eval_acc_ls = []

for i in range(params.n_epochs):
    print('\nTRAINING : Epoch ' + str(i))
    train_loss, train_acc = trainepoch(i)
    train_loss_ls.append(train_loss)
    train_acc_ls.append(train_acc)
    print('results : epoch {0} ; loss: {1} mean accuracy train : {2}'.format(i, train_loss, train_acc))
    if i%1==0:
        print("-"*100)
        print('\nEVALIDATING : Epoch ' + str(i))
        eval_acc = evaluate(i, eval_type='dev', final_eval=False)
        eval_acc_ls.append(eval_acc)
        print('results : epoch {0} ;  mean accuracy dev : {1}'.format(i, eval_acc))
        print("-"*100)



