from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from eval import MultiWozEvaluator
from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
from tqdm import tqdm


import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, TFRobertaModel
from tqdm import tqdm

from config import global_config as cfg 
# from config21 import global_config as cfg  # global, already initialized


import warnings
warnings.filterwarnings("ignore")

# def build_model(n_categories):
#         with strategy.scope():
#             input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
#             input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
#             input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

#             # Import RoBERTa model from HuggingFace
#             roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
#             x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

#             # Huggingface transformers have multiple outputs, embeddings are the first one,
#             # so let's slice out the first position
#             x = x[0]

#             x = tf.keras.layers.Dropout(0.1)(x)
#             x = tf.keras.layers.Flatten()(x)
#             x = tf.keras.layers.Dense(256, activation='relu')(x)
#             x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)
#             model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(lr=1e-5),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])

#             return model    


# strategy = tf.distribute.get_strategy()
# MODEL_NAME = 'roberta-base'
# MAX_LEN = 256

# d = {'PersonalAppeal' : 0,
#      'LogicalAppeal' : 1,
#      'CreadibilityAppeal' : 2,
#      'EmotionalAppeal' : 3,
#      'PersonaAppeal' : 4}
     
# n_categories_persuasion = 5
# n_categories_parent = 4

# model_parent = build_model(n_categories_parent)
# model_persuasion = build_model(n_categories_persuasion)

# # Load the previously saved weights
# checkpoint_dir_persuasion = "persuasive_roberta/cp.ckpt"
# model_persuasion.load_weights(checkpoint_dir_persuasion)
# print("HEEEEELLLLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

# checkpoint_dir_parent = "persuasive_roberta/main_4_parent/cp.ckpt"
# model_parent.load_weights(checkpoint_dir_parent)

# tokenizer_roberta = RobertaTokenizer.from_pretrained(MODEL_NAME)

class Modal(object):
    def __init__(self, device):
        self.device = device
        # initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)

        # create model: gpt2
        self.model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)  # single gpu

        #
        self.evaluator = MultiWozEvaluator(self.reader)
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log')
        else:
            self.tb_writer = None

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] *\
            cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:4]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor(
            [inputs['context']]).to(self.device)
        return inputs

    # def calculate_persuasive_and_sentiment_reward(self,inputs):
    #     REWARD = 0
    #     PENALTY = 0
    #     for I in range(len(inputs['contexts'])):
    #         s = self.tokenizer.decode(inputs['contexts_tensor'][I])
    #         a = s.split(" ")

    #         sentiments = []
    #         contexts = []
    #         actions = [] 
    #         i = 0
    #         while i < len(a) :
    #             if a[i] == "<sos_u>" :
    #                 contxt = ""
    #                 j = i+1

    #                 while j < len(a)-1 :
    #                     if a[j] == "<eos_u>" :
    #                         break
    #                     j += 1    
    #                 for k in range(i,j) :
    #                     contxt = contxt + a[k]

    #                 i = j-1 

    #             if a[i] == "<sos_a>" :
    #                 act = a[i+2]
    #                 actions.append(act[1:-1])

    #             if a[i] == "<sos_r>" :
    #                 j = i+1

    #                 while j < len(a)-1 :
    #                     if a[j] == "<eos_r>" :
    #                         break
    #                     j += 1    
    #                 for k in range(i,j) :
    #                     contxt = contxt + a[k]
                    
    #                 contexts.append(contxt)
    #                 i = j-1

    #             if a[i] == "<sos_s>" :
    #                 sentiments.append(a[i+1])    

    #             i += 1 
    #         s_counter = 0
    #         penalty = 0
    #         for i in sentiments:
    #             if i == 'negative' and s_counter < 0 :
    #                 s_counter += 1
    #             elif i == "negative" and s_counter >= 3 :
    #                 penalty += 1  
    #             else :
    #                 s_counter = 0      
    #         penalty /= len(sentiments)
    #         PENALTY += penalty

    #         r = 0
    #         for i in range(len(actions)) :
    #             context = [contexts[i]]
    #             clss = actions[i]

    #             if "req" in clss.lower() :
    #                 cls_n1 = 0
    #             elif "appeal" in clss.lower() :
    #                 cls_n1 = 3 
    #             elif clss.lower() == 'inform' :
    #                 cls_n1 = 2
    #             else : 
    #                 cls_n1 = 1

    #             if cls_n1 == 3 :
    #                 if clss == "CreadibilitylAppeal" :
    #                     clss = "CreadibilityAppeal"
    #                 elif clss == "PersonAppeal" :
    #                     clss = "PersonaAppeal"  

    #                 cls_n2 = d[clss] 

    #             pred = model_parent.predict(self.roberta_encode(context, tokenizer_roberta))
    #             pred = [i for i in pred[0]]
    #             #print(pred)

    #             clss = cls_n1

    #             reward = 0

    #             if int(clss) != 3:
    #                 if clss == 0 :
    #                     reward = 2*pred[clss] - 1
    #                 else :
    #                     reward = 2*pred[clss] + pred[0] - 1 
    #             else: 
    #                 pprob = pred[clss]
    #                 clss = cls_n2

    #                 prediction = model_persuasion.predict(self.roberta_encode(context, tokenizer_roberta))
    #                 prediction = [i for i in prediction[0]]
    #                 #print(prediction)

    #                 prob = pprob * prediction[clss] + pred[0]

    #                 reward = 2*prob - 1

    #             r += reward
    #         r /= len(actions) 
    #         REWARD += r

    #     REWARD /= len(inputs['contexts'])
    #     PENALTY /= len(inputs['contexts'])

    #     return REWARD,PENALTY      

    # def calculate_jaccard_and_meteor(self,inputs):
    #     final_jaccard = 0
    #     final_meteor = 0
    #     for I in range(len(inputs['contexts'])):
    #         s = self.tokenizer.decode(inputs['contexts_tensor'][I])
    #         a = s.split(" ")
    #         l = 0
    #         u = 0
    #         gt_responses = []
    #         contexts = []

    #         i = 0 
    #         while i < len(a)-1:
    #             if a[i] == "<sos_a>":
    #                 cntxt = ""
    #                 contexts.append([cntxt+a[k] for k in range(i)])
    #                 l = i
    #                 j = i+1 
    #                 #print(j)
    #                 while j < len(a) and a[j] != "<eos_r>" :
    #                   #print(j)
    #                   j += 1
    #                 u = j
    #                 for m in range(i,j):
    #                     if a[m] == "<sos_r>" :
    #                         l = m  

    #                 if j < len(a):
    #                     s = ""
    #                     s1 = [s+a[k] for k in range(l+1,u)]   
    #                     #print(s1)
    #                     gt_responses.append(s1)

    #                 i = u+1
    #             else:
    #               i = i+1

    #         JACCARD = 0
    #         METEOR = 0

    #         for i in range(len(gt_responses)):
    #             input1 = []
    #             a = self.tokenizer.encode(contexts[i])

    #             input1.append(a)
    #             input1.append(a)

    #             input1 = torch.IntTensor(input1)
                
    #             # print(inputs['contexts_tensor'].shape)
    #             # print(type(inputs['contexts_tensor']))
    #             # print(type(input1))
    #             #print("input_final: ",contexts[i])
              
    #             context_length = input1.shape[1]
    #             max_len = 60
    #             outputs = self.model.generate(input_ids=input1.to(self.device),
    #                                                 max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
    #                                                 pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
    #                                                 #,no_repeat_ngram_size=4)
                       
                      
    #             generated = outputs[0].cpu().numpy().tolist()
    #             generated = generated[context_length-1:]
    #             #print("generated action and response:",self.tokenizer.decode(generated))

    #             # calculate jaccard similarity between current and previous gt responses
    #             jaccard = 0

    #             for j in range(i+1):
    #                 jaccard += utils.jaccard_similarity(gt_responses[j],self.tokenizer.decode(generated).split(" "))
    #             if i >= 1 :    
    #                 jaccard = jaccard/(i)  

    #             JACCARD += jaccard                

    #             # calculate meteor score between generated and gt responses                 
    #             # meteor = meteor_score([gt_responses[i]],self.tokenizer.decode(generated).split(" "),gamma=0.5)
    #             # METEOR += meteor

    #         JACCARD /= len(gt_responses)
    #         final_jaccard += JACCARD

    #         METEOR /= len(gt_responses)
    #         final_meteor += METEOR
    #     #print("jaccard similarity: ",final_jaccard/len(inputs['contexts']))
    #     #print("meteor score: ",final_meteor/len(inputs['contexts']))

    #     return final_jaccard/len(inputs['contexts']),final_meteor/len(inputs['contexts'])
    def get_n_strategy_GAT_embedddings(self, n_strategy):
        # print(n_strategy)
        n_strategy_dict = {'assertive_count_seller': 0,
        'seller_neg_sentiment': 1,
        'politeness_seller_please': 2,
        'first_person_plural_count_seller': 3,
        'third_person_singular_seller': 4,
        'politeness_seller_greet': 5,
        'third_person_plural_seller': 6,
        'liwc_certainty': 7,
        'hedge_count_seller': 8,
        'first_person_singular_count_seller': 9,
        'who_propose': 10,
        'seller_propose': 11,
        'number_of_diff_dic_pos': 12,
        'politeness_seller_please_s': 13,
        'friend': 14,
        'seller_pos_sentiment': 15,
        'family': 16,
        'seller_trade_in': 17,
        'liwc_informal': 18,
        'personal_concern_seller': 19,
        'factive_count_seller': 20,
        'politeness_seller_gratitude': 21,
        'number_of_diff_dic_neg': 22}

        # n_strategy_GAT_embeddings = [[  77.5088],[ 269.6295],[ 249.7413],[  91.2195],[ 164.2853],[ 314.7302],
               #  [-158.7896],[ 186.0732],[ 125.5586],[ 143.6914],[ 239.3672],[  67.8141],[ 184.5765],[ 281.3583],
               #  [ 313.1075],[ 219.6080],[ 291.6160],[ 196.4630],[ 230.3395],[ 169.5111],[ 100.0341],[ 276.1250],[  19.8230]]

        n_strategy_GAT_embeddings = [[-3678.2053,  1618.5809, -5768.2031,  5368.6572,  7347.3550],
                                    [-3674.6389,  1610.1393, -5794.4219,  5390.0767,  7376.8257],
                                    [-3687.0627,  1631.0425, -5769.0791,  5368.8389,  7348.3311],
                                    [-3675.8547,  1609.2190, -5803.3364,  5398.7119,  7388.0952],
                                    [-3673.7910,  1610.5140, -5781.2266,  5380.2495,  7361.6938],
                                    [-3676.0610,  1611.1135, -5808.4692,  5402.5464,  7392.7002],
                                    [-3671.6311,  1605.2032, -5797.6538,  5393.4683,  7381.1807],
                                    [-3677.8882,  1616.1155, -5790.3965,  5387.6812,  7371.7378],
                                    [-3685.0459,  1624.9315, -5776.1484,  5377.4775,  7357.2119],
                                    [-3672.4187,  1607.6600, -5802.8262,  5395.9404,  7385.0557],
                                    [-3670.8674,  1605.6176, -5796.2329,  5389.9639,  7378.4043],
                                    [-3680.1731,  1617.6873, -5799.5718,  5395.7485,  7382.2944],
                                    [-3666.8896,  1604.1775, -5774.7920,  5370.6484,  7353.0776],
                                    [-3665.2256,  1598.4028, -5773.7588,  5374.2451,  7354.6411],
                                    [-3683.9568,  1622.3293, -5793.1226,  5392.7603,  7375.5898],
                                    [-3671.1118,  1608.0133, -5793.6060,  5387.9678,  7374.5542],
                                    [-3671.3313,  1609.9087, -5785.5190,  5380.2681,  7365.0229],
                                    [-3695.2578,  1641.3744, -5784.1821,  5386.5093,  7363.1035],
                                    [-3680.8018,  1619.3429, -5798.4678,  5396.2827,  7379.9531],
                                    [-3668.8899,  1606.0234, -5797.3999,  5388.9507,  7377.0439],
                                    [-3681.2490,  1622.4958, -5767.4448,  5368.0068,  7346.4795],
                                    [-3693.9114,  1631.8101, -5792.3530,  5398.5034,  7378.8418],
                                    [-3688.4163,  1633.0482, -5776.4917,  5376.8892,  7355.5005]]

        # embedding = n_strategy_GAT_embeddings[n_strategy_dict[n_strategy]][0]
        embedding = n_strategy_GAT_embeddings[n_strategy_dict[n_strategy]]
        embedding = ['['+str(int(i))+']' for i in embedding]
        embedding = ' '.join(embedding)

        # return int(embedding)
        return embedding

    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss


    def train(self):
        """
        UBARU
        """
        all_batches = self.reader.get_batches('train')
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_dials']*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size))

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        sw = time.time()
        f = open('loss_records.txt','w')
        f.close()

        for epoch in tqdm(range(cfg.epoch_num)):
            f = open('loss_records.txt','a')
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            lowest_loss = 99999.9999
            self.model.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(
                all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session(dial_batch)
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    # to tensor
                    print("----------------------------------------")
                    print("EPOCH:{}, epoch step:{}".format(epoch,epoch_step))
                    inputs = self.add_torch_input(inputs)
                    print("input shape: ",inputs['contexts_tensor'].shape)
                    print("context: ",self.tokenizer.decode(inputs['contexts_tensor'][0]))
                    # loss
                    outputs = self.model(inputs['contexts_tensor'])

                    # persuasive_reward,sentiment_penalty = self.calculate_persuasive_and_sentiment_reward(inputs)
                    # jaccard,meteor = self.calculate_jaccard_and_meteor(inputs)
                  
                    # reward = (1) * jaccard# + (-1) * meteor + (-1) * persuasive_reward + (2) * sentiment_penalty  # have to add more rewards
                    
                    # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                                        
                    # print("repeatitiveness reward: ",jaccard)
                    # print("consistency reward: ",meteor)
                    # print("persuasive dialogue generation reward: ",persuasive_reward)
                    # print("sentiment based penalty: ",sentiment_penalty)
                    # print("overall reward: ",reward)
                    loss = self.calculate_loss_and_accuracy(
                        outputs, labels=inputs['contexts_tensor'])
                    print("loss: ",loss)  

                    # loss = loss + reward + 1.0   
                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 5.0)
                    epoch_step += 1
                    # print("final_loss: ",loss)
                    # print("epoch step: ",epoch_step)
                    print("----------------------------------------")

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                        # end of an epoch
                        (epoch_step + \
                         1) == set_stats['num_training_steps_per_epoch']
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # global_step: actual step the optimizer took
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                            loss_scalar = (tr_loss - logging_loss) / \
                                cfg.report_interval
                            logging_loss = tr_loss
                            logs['loss'] = loss_scalar
                            logging.info(
                                'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                    global_step, epoch_step, loss_scalar
                                ))
                            # validate
                            # add to tensorboard...
                            if cfg.evaluate_during_training and loss_scalar < 10:
                                results = self.validate()
                                for k, v in results.items():
                                    eval_key = "eval_{}".format(k)
                                    logs[eval_key] = v

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(
                                        k, v, global_step)
                            # save model... 

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time()-btm)/60, tr_loss))
            # save model after every epoch
            # if epoch > 10 or tr_loss/epoch_step < 1:
            # if epoch % 5 == 0:
            flag = 0
            epoch_loss = tr_loss/epoch_step
            f.writelines("Epoch: "+str(epoch)+", loss: "+str(epoch_loss)+"\n")
            f.close()

            self.save_model(epoch, epoch_loss, flag)

            if epoch_loss <= lowest_loss:
                flag = 1
                self.save_model(epoch, epoch_loss, flag)


    def save_model(self, epoch, loss, flag=0):
        # save_path = os.path.join(
        #     cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2'.format(epoch+1, loss))

        save_path = os.path.join(
            cfg.exp_path, 'final_epoch_GAT_5_emb_n_strategy')
        if flag == 1:
            save_path = os.path.join(
            cfg.exp_path, 'minimum_loss_model_GAT_5_emb_n_strategy')

        print("loss :",loss)
        # save_path = os.path.join(
        #       cfg.exp_path,'best_model_gpt2')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg

    
    def validate(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()

        # all_batches = self.reader.get_batches('dev')
        # data_iterator = self.reader.get_data_iterator(all_batches)
        eval_data = self.reader.get_eval_data(data)

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        n_strategy_track = {'generated':[], 'actual':[]}
        action_track = {'generated':[], 'actual':[]}

        f = open('outputs.txt','w')
        f.close()

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            f = open('outputs.txt','a')
            for dial_idx, dialog in enumerate(eval_data):
                print("dialog:",dial_idx)
                f.writelines("\n############################# " + str(dial_idx) + " #############################")
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    print("turn:",self.tokenizer.decode(turn['user']))
                    # print("turn:",self.tokenizer.decode(turn['n_strategy']))
                    f.writelines("\n"+"turn: "+self.tokenizer.decode(turn['user'])+"\n")
                    first_turn = (turn_idx == 0)
                    # print("first turn status: ",first_turn)
                    inputs = self.reader.convert_turn_eval(
                        turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set

                    context_length = len(inputs['context'])
                    # print("context lenghth: ",context_length)
                    # print("context tensor length: ",len(inputs['context_tensor'][0]))

                    if cfg.use_true_curr_bspn: # generate act, response
                        max_len=60
                        # inputs['context_tensor_ns'] = torch.tensor([inputs['context'][:-1] + self.tokenizer.encode(['<sos_ns>'])]).to(self.device)
                        # context_length = len(inputs['context_tensor_ns'][0])
                        print("context:",self.tokenizer.decode(inputs['context_tensor'][0]))
                        # outputs_ns = self.model.generate(input_ids=inputs['context_tensor_ns'],
                        #                             max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                        #                             pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_ns>'])[0])
                        # generated_ns = outputs_ns[0].cpu().numpy().tolist()
                        # generated_ns = generated_ns[context_length-1:]
                        # print("generated negotiation strategy: ",self.tokenizer.decode(generated_ns))
                        # f.writelines("generated negotiation strategy:"+self.tokenizer.decode(generated_ns)+"\n")

                        # inputs['context_tensor_act'] = torch.tensor([inputs['context'][:-1] + turn['n_strategy'] + self.tokenizer.encode(['<sos_a>'])]).to(self.device)
                        # context_length = len(inputs['context_tensor_act'][0])

                        if not cfg.use_true_curr_aspn:
                            max_len = 80
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode('<eos_r>')[0])
                                                    #   no_repeat_ngram_size=4
                        # turn['generated'] = self.tokenizer.decode(outputs[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length-1:]
                        print("generated action and response:",self.tokenizer.decode(generated))   
                        f.writelines("generated ar:"+self.tokenizer.decode(generated)+"\n")
                        f.writelines("-----------------------------------------------------------------\n")
                        print("------------------------------------------------------------")
                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else: # predict bspn, access db, then generate act and resp
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode('<eos_b>')[0])
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
                        print("generated belief state:",self.tokenizer.decode(bspn_gen))
                        f.writelines("generated belief state:"+self.tokenizer.decode(bspn_gen)+"\n")
                        # check DB result
                        # if cfg.use_true_db_pointer:
                        #     # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                        #     db = turn['db']
                        # else:
                        #     db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
                        #     db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
                        inputs['context_tensor_ns'] = torch.tensor([inputs['context'][:-1] + bspn_gen + self.tokenizer.encode('<sos_ns>')]).to(self.device)
                        context_length = len(inputs['context_tensor_ns'][0])
                        # print("ns generation conetxt: ",self.tokenizer.decode(inputs['context_tensor_ns'][0]))
                        outputs_ns = self.model.generate(input_ids=inputs['context_tensor_ns'],
                                                    max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode('<eos_ns>')[0])
                        # generated_ns = outputs_ns[0].cpu().numpy().tolist()
                        # generated_ns = generated_ns[context_length-2:]
                        # s = self.tokenizer.decode(generated_ns)
                        # s1 = s.split('<')[0]
                        # s2 = "<sos_ns> " + s1.split('>')[1] + " <eos_ns>"
                        end = 'yes'
                        gen_ns = []
                        # print("generated negotiation strategies: ")
                        while len(end) != 0:
                            context = self.tokenizer.decode(inputs['context_tensor_ns'][0])
                            # print("context: ",context)
                            string = outputs_ns[0].cpu().numpy().tolist()[context_length:]
                            # print("yo: ",self.tokenizer.decode(outputs_ns[0].cpu().numpy().tolist()[context_length+1]))
                            
                            s = self.tokenizer.decode(string)
                            s1 = s.split('<')[0]
                            end = s1.split(' ')[1]
                            # print(s1)
                            # print(len(end))
                            # print(end) 
                            if len(end)>4: 
                                gen_ns.append(end)

                                try :
                                    # print(str(end[1:-1]))
                                    # GAT_emb = str([self.get_n_strategy_GAT_embedddings(str(end[1:-1]))])
                                    GAT_emb = str(self.get_n_strategy_GAT_embedddings(str(end[1:-1])))
                                    # print(GAT_emb)
                                except :
                                    GAT_emb = ''
                                    print('end: ',end)


                                gen_ns.append(GAT_emb)

                              
                                inputs['context_tensor_ns'] = torch.tensor([self.tokenizer.encode(context+" "+end+" "+GAT_emb)]).to(self.device)
                                context_length = len(inputs['context_tensor_ns'][0])
                                outputs_ns = self.model.generate(input_ids=inputs['context_tensor_ns'],
                                                        max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                        pad_token_id=self.tokenizer.eos_token_id) 

                            else:
                                break

                        n_strategy_track['generated'].append(" ".join(gen_ns))
                        n_strategy_track['actual'].append(self.tokenizer.decode(turn['n_strategy']))
                        
                        print("generated negotiation strategy: "," ".join(gen_ns))
                        f.writelines("generated negotiation strategy:"+" ".join(gen_ns)+"\n")
                        # print("context: ",context)
                        inputs['context_tensor_act'] = torch.tensor([self.tokenizer.encode(context+" <eos_ns>") + self.tokenizer.encode('<sos_a>')]).to(self.device)
                        context_length = len(inputs['context_tensor_act'][0])
                        print("final_context: ",self.tokenizer.decode(inputs['context_tensor_act'][0]))
                        # print("hello")
                        outputs_act = self.model.generate(input_ids=inputs['context_tensor_act'],
                                                    max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode('<eos_r>')[0])
                        generated_ar = outputs_act[0].cpu().numpy().tolist()
                        generated_ar = generated_ar[context_length-1:]
                        print("generated ar:",self.tokenizer.decode(generated_ar))
                        f.writelines("generated ar:"+self.tokenizer.decode(generated_ar)+"\n")
                        f.writelines("-----------------------------------------------------------------\n")
                        print("------------------------------------------------------------")

                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated_ar))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                        action_track['actual'].append(self.tokenizer.decode(turn['aspn']))
                        action_track['generated'].append(self.tokenizer.decode(decoded['aspn']))

                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    # turn['dspn_gen'] = turn['dspn']

                    # check DB results
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    pv_turn['labels'] = inputs['labels'] # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    generated_n_strategy = self.tokenizer.encode("<sos_ns> "+" ".join(gen_ns)+" <eos_ns>")
                    pv_turn['n_strategy'] = generated_n_strategy
                    # pv_turn['n_strategy'] = turn['n_strategy'] #############################################
                    # pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))
            f.close()
        logging.info("inference time: {:.2f} min".format((time.time()-btm)/60))

        pd.DataFrame(n_strategy_track).to_csv("GAT_n_strategy_track.csv")
        pd.DataFrame(action_track).to_csv("action_track.csv")
        
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
        print("results :",results)
        bleu, success, match, rouge1, rougeL, BLEU1, BLEU2, BLEU3, BLEU4, prc, rec, f1 = self.evaluator.validation_metric(results)
        logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f  rouge1: %2.2f rougeL: %2.2f BLEU1: %2.4f BLEU2: %2.2f BLEU3: %2.2f BLEU4: %2.2f precision: %2.4f recall: %2.4f f1_score: %2.4f score: %.2f' % (
            match, success, bleu, rouge1, rougeL, BLEU1, BLEU2, BLEU3, BLEU4, prc, rec, f1, score))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match 
        eval_results['score'] = score
        eval_results['rouge1'] = rouge1
        eval_results['rougeL'] = rougeL
        eval_results['BLEU1'] = BLEU1
        eval_results['BLEU2'] = BLEU2
        eval_results['BLEU3'] = BLEU3
        eval_results['BLEU4'] = BLEU4
        eval_results['precision'] = prc
        eval_results['recall'] = rec
        eval_results['f1_score'] = f1
        #eval_results['perp'] = perp
        eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f  rouge1: %2.2f rougeL: %2.2f BLEU1: %2.4f BLEU2: %2.2f BLEU3: %2.2f BLEU4: %2.2f precision: %2.4f recall: %2.4f f1_score: %2.4f score: %.2f' % (
            match, success, bleu, rouge1, rougeL, BLEU1, BLEU2, BLEU3, BLEU4, prc, rec, f1, score)
        
        model_setting, epoch_setting = cfg.eval_load_path.split('/')[1], cfg.eval_load_path.split('/')[2]
        eval_on = '-'.join(cfg.exp_domains)
        if data == 'test':
            eval_on += '_test'
        if not os.path.exists(cfg.log_path):
            os.mkdir(cfg.log_path)
        log_file_name = os.path.join(cfg.log_path, model_setting+'-'+eval_on+'.json')
        if os.path.exists(log_file_name):
            eval_to_json = json.load(open(log_file_name, 'r'))
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        else:
            eval_to_json = {}
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        logging.info('update eval results to {}'.format(log_file_name))
        return eval_results


    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated)-1
            logging.info('eos_r not in generated: ' + self.tokenizer.decode(generated))
        # eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated)-1
        
        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx+1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx+1]
            decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        # if cfg.use_true_curr_bspn:
            
        # else:  # predict bspn aspn resp
        #     eos_b_idx = generated.index(eos_b_id)
        #     eos_a_idx = generated.index(eos_a_id)
        #     decoded['bspn'] = generated[: eos_b_idx+1]
        #     decoded['aspn'] = generated[eos_b_idx+1: eos_a_idx+1]
        #     decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated)-1
        return generated[: eos_b_idx+1]

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21'):
        os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'adjust':
        parse_arg_cfg(args)
        # cfg.model_path = cfg.eval_load_path
        cfg.gpt_path = cfg.eval_load_path
    else:  # train
        parse_arg_cfg(args)
        if cfg.exp_path in ['', 'to be generated']:
            # log file path, control the factors: seed, learning_rate, batch_size, early_stop_count, weight decay...
            # cfg.exp_path = 'experiments/{}_{}_sd{}_lr{}_bs{}_sp{}_dc{}/'.format('-'.join(cfg.exp_domains),
            #                                                                     cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
            #                                                                     cfg.early_stop_count, cfg.weight_decay_count)
            
            experiments_path = './experiments' if 'all' in cfg.exp_domains else './experiments_Xdomain'
            cfg.exp_path = os.path.join(experiments_path,'{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps))
            logging.info('save path:', cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device) == 1:
            cfg.multi_gpu = False
            # torch.cuda.set_device(cfg.cuda_device[0])
            device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
        else:
            pass  # multi-gpu
    else:
        device = torch.device('cpu')
        logging.info('Device: {}'.format(torch.cuda.current_device()))
    
    # IF ONLY CPU IS AVAILABLE ----->
    # device = torch.device('cpu')

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Modal(device)

    if args.mode == 'train':    # train
        if cfg.save_log:  # save cfg details.
            pass
        if cfg.context_scheme == 'UBARU':
            m.train()
        elif cfg.context_scheme == 'URURU':
            m.train_URURU()
        else:
            logging.info('Invalid context Scheme. must be UBARU or URURU')
            exit()
    elif args.mode == 'adjust':
        pass
    else:  # test
        logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_prev_resp,
                            cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
                        ))

        if cfg.context_scheme == 'UBARU':
            m.validate()
            m.validate('test')
        elif cfg.context_scheme == 'URURU':
            m.validate_URURU()
            m.validate_URURU('test')

        # logging.info('Running eavl on test')
        # m.validate('test')


#  testing:  python train.py -mode test -cfg eval_load_path=experiments/all__sd11_lr0.001_bs2_ga8/epoch5_trloss0.80_gpt2/


if __name__ == "__main__":
    main()
