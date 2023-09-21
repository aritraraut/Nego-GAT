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
import re


import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import global_config as cfg 

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

    # def calculate_loss_and_accuracy(self, outputs, labels):
    #     # GPT2-chicahat/train.py
    #     lm_logits = outputs[0]

    #     shift_logits = lm_logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()

    #     pad_id = cfg.pad_id
    #     loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    #     loss = loss_fct(
    #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    #     # avg loss
    #     not_ignore = shift_labels.ne(pad_id)
    #     num_targets = not_ignore.long().sum().item()

    #     loss /= num_targets
    #     return loss

    
    def validate(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()

        # all_batches = self.reader.get_batches('dev')
        # data_iterator = self.reader.get_data_iterator(all_batches)
        # eval_data = self.reader.get_eval_data(data)
        # print("eval data:",eval_data)
        # set_stats = self.reader.set_stats[data]
        # logging.info("***** Running Evaluation *****")
        # logging.info("  Num Turns = %d", set_stats['num_turns'])
        # # logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        
        # valid_losses = []
        btm = time.time()
        result_collection = {}
        f = open("end-to-end_Response_generation_GAT.txt","w")
        f.close()
        with torch.no_grad():
            print("starting chatbot...")
            f = open("end-to-end_Response_generation_GAT.txt","a")
            f.writelines("---------------------------------------------"+"\n")
            f.writelines("dialogue:"+str(2)+"\n")
            pv_turn = {}
            for i in range(20):
                domain = ['[Common]']
            
                # turn = input("turn: ")
                turn = '<sos_u>' + input("turn: ") + '<eos_u>'             
                f.writelines("user: "+turn+"\n")
                turn = {"user":self.tokenizer.encode(turn)}
                #print("turn:",turn)
                first_turn = (i == 0)
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
                    if context_length >= 900 :
                        inputs['context_tensor'][0] = inputs['context_tensor'][0][-900:]
                        context_length = len(inputs['context_tensor'][0])

                    outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode('<eos_s>')[0])
                    generated_s = outputs[0].cpu().numpy().tolist()
                    generated_s = generated_s[context_length-1:]
                    print("generated sentiment:",self.tokenizer.decode(generated_s[1:-1]))
                    f.writelines("generated sentiment:"+self.tokenizer.decode(generated_s[1:-1])+"\n") 
                    inputs['context_tensor'] = torch.tensor([inputs['context'][:-1] + generated_s + self.tokenizer.encode('<sos_b>')]).to(self.device) 
                    context_length = len(inputs['context_tensor'][0])                         

                    if context_length > 900:
                        inputs['context_tensor'][0] = inputs['context_tensor'][0][-900:]
                        context_length = len(inputs['context_tensor'][0])

                    outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                max_length=context_length+200, temperature=0.7, # top_p=0.9, num_beams=4,
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
                    inputs['context_tensor_ns'] = torch.tensor([inputs['context'][:-1] + generated_s + bspn_gen + self.tokenizer.encode('<sos_ns>')]).to(self.device)
                    # print(self.tokenizer.decode(inputs['context_tensor_ns'][0]))
                    context_length = len(inputs['context_tensor_ns'][0])

                    if context_length > 900 :
                        inputs['context_tensor_ns'][0] = inputs['context_tensor_ns'][0][-900:]
                        context_length = len(inputs['context_tensor_ns'][0])

                    outputs_ns = self.model.generate(input_ids=inputs['context_tensor_ns'],
                                                max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode('<eos_ns>')[0])
                    
                    
                    
                    # print("generated negotiation strategy: ")
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
                        gen_ns.append(end)   
                      
                        inputs['context_tensor_ns'] = torch.tensor([self.tokenizer.encode(context+" "+end)]).to(self.device)
                        context_length = len(inputs['context_tensor_ns'][0])
                        
                        if context_length > 900 :
                            inputs['context_tensor_ns'][0] = inputs['context_tensor_ns'][0][-900:]
                            context_length = len(inputs['context_tensor_ns'][0])

                        outputs_ns = self.model.generate(input_ids=inputs['context_tensor_ns'],
                                                max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                pad_token_id=self.tokenizer.eos_token_id)                      

                    # generated_ns = outputs_ns[0].cpu().numpy().tolist()
                    # generated_ns = generated_ns[context_length-2:]
                    # s = self.tokenizer.decode(generated_ns)
                    # s1 = s.split('<')[0]
                    # s2 = "<sos_ns> " + s1.split('>')[1] + " <eos_ns>"
                    
                    print("generated negotiation strategy: ",gen_ns)
                    f.writelines("generated negotiation strategy:"+" ".join(gen_ns)+"\n")

                    inputs['context_tensor_act'] = torch.tensor([self.tokenizer.encode(context+" <eos_ns>") + self.tokenizer.encode('<sos_a>')]).to(self.device)
                    context_length = len(inputs['context_tensor_act'][0])

                    if context_length > 900 :
                        print(len(inputs['context_tensor_act']))
                        print(len(inputs['context_tensor_act'][0]))
                        inputs['context_tensor_act'][0] = inputs['context_tensor_act'][0][-900:]
                        context_length = len(inputs['context_tensor_act'][0])

                    print("final_context: ",self.tokenizer.decode(inputs['context_tensor_act'][0]))

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
                gen_nego_strategy = " ".join(gen_ns)
                pv_turn['n_strategy'] = self.tokenizer.encode(gen_nego_strategy) #############################################
                # pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                pv_turn['sentiment'] = generated_s
                pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

            f.close()
                # result_collection.update(
                #     self.reader.inverse_transpose_turn(dialog))
                # # print("-------------------------------------------------------------")
                #print("result collection:",result_collection)    

        logging.info("inference time: {:.2f} min".format((time.time()-btm)/60))
        # score
        # btm = time.time()
        # results, _ = self.reader.wrap_result_lm(result_collection)
        # bleu, success, match = self.evaluator.validation_metric(results)
        # logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        # score = 0.5 * (success + match) + bleu
        # valid_loss = 130 - score
        # logging.info('validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
        #     match, success, bleu, score))
        # eval_results = {}
        # eval_results['bleu'] = bleu
        # eval_results['success'] = success
        # eval_results['match'] = match
        # eval_results['score'] = score
        # eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (match, success, bleu, score)

        
        # model_setting, epoch_setting = cfg.eval_load_path.split('/')[1], cfg.eval_load_path.split('/')[2]
        # eval_on = '-'.join(cfg.exp_domains)
        # if data == 'test':
        #     eval_on += '_test'
        # if not os.path.exists(cfg.log_path):
        #     os.mkdir(cfg.log_path)
        # log_file_name = os.path.join(cfg.log_path, model_setting+'-'+eval_on+'.json')
        # if os.path.exists(log_file_name):
        #     eval_to_json = json.load(open(log_file_name, 'r'))
        #     eval_to_json[epoch_setting] = eval_results
        #     json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        # else:
        #     eval_to_json = {}
        #     eval_to_json[epoch_setting] = eval_results
        #     json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        # logging.info('update eval results to {}'.format(log_file_name))
        # return eval_results

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
    # if not os.path.exists('./experiments'):
    #     os.mkdir('./experiments')

    # if not os.path.exists('./experiments_21'):
    #     os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = 'test'
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
    # if cfg.cuda:
    #     print("helllllllloooooo")
    #     if len(cfg.cuda_device) == 1:
    #         print("yooooooooooooo")
    #         cfg.multi_gpu = False
    #         # torch.cuda.set_device(cfg.cuda_device[0])
    #         device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
    #     else:
    #         pass  # multi-gpu
    # else:
    device = torch.device('cpu')
    #logging.info('Device: {}'.format(torch.cuda.current_device()))

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
        logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
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

