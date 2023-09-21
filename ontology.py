all_domains = ['common']

slot_list = ['Price_ratio']

# original slot names in goals (including booking slots)
normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

requestable_slots_in_goals = {
    "common": slot_list
}


informable_slots_in_goals = requestable_slots_in_goals

requestable_slots = {
    "common": slot_list
}

all_reqslot = ['Price_ratio']

informable_slots = {
    "common": slot_list
}

all_infslot = all_reqslot

all_slots = all_reqslot

get_slot = {}
for s in all_slots:
    get_slot[s] = 1


da_abbr_to_slot_name = {
    'brand': "Brand",
    'price': "Price",
    'Cost': "Price"
}

dialog_act_dom = all_domains

dialog_acts = {
    'common' : ['init-pricecounter-price','unknownunknownvague-priceinsist','disagree','disagreecounter-price','inquiryinquiry','unknownunknownagree','unknowndisagreedisagree',
 'init-priceagree','unknowndisagreeunknown','agree','counter-priceinquiry','unknowninit-price','informinit-price','disagreeinformunknown','introinit-price',
 'inquiryunknown','vague-price','counter-pricedisagree','unknownagree','insist','informinquiry','counter-priceunknown','unknowninsist','counter-pricecounter-pricecounter-pricecounter-price',
 'introinquiry','informinform','counter-priceinsist','introcounter-price','informunknown','init-price','disagreedisagree','inform','agreeunknown','unknowndisagree',
 'inquiryagree','insistunknown','introdisagree','inquiryvague-price','intro','unknownunknowncounter-price','insistunknowncounter-price','counter-pricecounter-price',
 'init-priceunknown','counter-pricevague-price','disagreeagree','informagree','unknownunknown','unknownagreeunknown','introunknown','unknowninforminform',
 'inquiry','counter-pricecounter-priceinsist','disagreeinform','introagree','agreeinquiry','unknowninquiry','disagreeinsist','counter-priceagree','disagreeunknown',
 'informdisagree','unknown','inquirycounter-priceunknownunknown','informinforminform','init-priceinquiry','unknowncounter-price','informinforminsist',
 'counter-price','unknownvague-price','informinsist']
}

all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)
# print(all_acts)

dialog_act_params = {i:slot_list for i in dialog_acts}

dialog_act_all_slots = all_slots #+ ['choice', 'open']

# special slot tokens in belief span
# no need of this, just covert slot to [slot] e.g. pricerange -> [pricerange]
slot_name_to_slot_token = {}


# special slot tokens in responses
# not use at the momoent
# slot_name_to_value_token = {
#        'Price_ratio' : '[value_price_ratio]',
# }


# db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']

special_tokens = ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>',
                            '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>','<eos_d>',
                            '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>','<sos_s>','<eos_s>'] #+ db_tokens

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>',
    'n_strategy': '<eos_ns>', 'n_strategy_gen': '<eos_ns>', 'pv_n_strategy': '<eos_ns>',
    'sentiment':'<eos_s>'}

sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>',
    'n_strategy': '<sos_ns>', 'n_strategy_gen': '<sos_ns>', 'pv_n_strategy': '<sos_ns>',
    'sentiment':'<sos_s>'
    }