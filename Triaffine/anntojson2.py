#!/usr/bin/env python

# Convert text and standoff annotations into CoNLL format.

from __future__ import print_function

import os
import re
import sys
import json
from collections import namedtuple
from io import StringIO
from os import path
import nltk
from nltk.tokenize import word_tokenize

from sentencesplit import sentencebreaks_to_newlines

# assume script in brat tools/ directory, extend path to find sentencesplit.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../server/src'))
sys.path.append('.')

options = None

EMPTY_LINE_RE = re.compile(r'^\s*$')
CONLL_LINE_RE = re.compile(r'^\S+\t\d+\t\d+.')

global_sentence = []
global_pos = []
global_entities = []
global_relations = []
global_boundary = []
backup_token = []


class FormatError(Exception):
    pass


def argparser():
    import argparse

    ap = argparse.ArgumentParser(description='Convert text and standoff ' +
                                 'annotations into CoNLL format.')
    ap.add_argument('-a', '--annsuffix', default="ann",
                    help='Standoff annotation file suffix (default "ann")')
    ap.add_argument('-c', '--singleclass', default=None,
                    help='Use given single class for annotations')
    ap.add_argument('-n', '--nosplit', default=False, action='store_true',
                    help='No sentence splitting')
    ap.add_argument('-o', '--outsuffix', default="conll",
                    help='Suffix to add to output files (default "conll")')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Verbose output')
    ap.add_argument('text', metavar='TEXT', nargs='+',
                    help='Text files ("-" for STDIN)')
    return ap


def read_sentence(f):
    """Return lines for one sentence from the CoNLL-formatted file.

    Sentences are delimited by empty lines.
    """

    lines = []
    for l in f:
        lines.append(l)
        if EMPTY_LINE_RE.match(l):
            break
        if not CONLL_LINE_RE.search(l):
            raise FormatError(
                'Line not in CoNLL format: "%s"' %
                l.rstrip('\n'))
    return lines


def strip_labels(lines):
    """Given CoNLL-format lines, strip the label (first TAB-separated field)
    from each non-empty line.

    Return list of labels and list of lines without labels. Returned
    list of labels contains None for each empty line in the input.
    """

    labels, stripped = [], []

    labels = []
    for l in lines:
        if EMPTY_LINE_RE.match(l):
            labels.append(None)
            stripped.append(l)
        else:
            fields = l.split('\t')
            labels.append(fields[0])
            stripped.append('\t'.join(fields[1:]))

    return labels, stripped


def attach_labels(labels, lines):
    """Given a list of labels and CoNLL-format lines, affix TAB-separated label
    to each non-empty line.

    Returns list of lines with attached labels.
    """

    assert len(labels) == len(
        lines), "Number of labels (%d) does not match number of lines (%d)" % (len(labels), len(lines))

    attached = []
    for label, line in zip(labels, lines):
        empty = EMPTY_LINE_RE.match(line)
        assert (label is None and empty) or (label is not None and not empty)

        if empty:
            attached.append(line)
        else:
            attached.append('%s\t%s' % (label, line))

    return attached


# NERsuite tokenization: any alnum sequence is preserved as a single
# token, while any non-alnum character is separated into a
# single-character token. TODO: non-ASCII alnum.
TOKENIZATION_REGEX = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')
NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')


def text_to_conll(f):
    """Convert plain text into CoNLL format."""
    global options

    if options.nosplit:
        sentences = f.readlines()
    else:
        sentences = []
        add_sent = []
        for l in f:
            add_sent.append(l)

        l = sentencebreaks_to_newlines(' '.join(add_sent).replace("\n", ""))
        sentences.extend([s for s in NEWLINE_TERM_REGEX.split(l) if s])
        
    
    # print("test1", sentences)
    lines = []
    
    sent_list = []
    sent_pos = []

    offset = 0
    for s in sentences:
        nonspace_token_seen = False

        tokens = [t for t in TOKENIZATION_REGEX.split(s) if t]
        tokens2 = [t for t in TOKENIZATION_REGEX.split(s) if t and t!=" " and t!="\n"]
        
        pos = nltk.pos_tag(tokens2)
        
        # pos1 = [p[0] for p in pos]
        pos2 = [p[1] for p in pos]
        # print("test1", pos1)
        # print("test2", pos2)
        
        sent_list.append(tokens2)
        sent_pos.append(pos2)

        for t in tokens:
            if not t.isspace():
                lines.append(['O', offset, offset + len(t), t])
                nonspace_token_seen = True
            offset += len(t)

        # sentences delimited by empty lines
        if nonspace_token_seen:
            lines.append([])
    
    global_sentence.append(sent_list)
    global_pos.append(sent_pos)
    #print("test2.2",sent_list)
    
    # add labels (other than 'O') from standoff annotation if specified
    # print(f.name)
    if options.annsuffix:
        lines = relabel(lines, get_annotations(f.name))

    lines = [[l[0], str(l[1]), str(l[2]), l[3]] if l else l for l in lines]
    return StringIO('\n'.join(('\t'.join(l) for l in lines)))
    #return sent_list


def relabel(lines, annotations):
    global options

    # TODO: this could be done more neatly/efficiently
    offset_label = {}
    
    overlap = 0
    for tb in annotations:
        for i in range(tb.start, tb.end):
            if i in offset_label:
                overlap+=1
                # print("Warning: overlapping annotations", file=sys.stderr)
            offset_label[i] = tb
            
    #print("test3", offset_label)

    prev_label = None
    temp_sent = []
    
    idx = 0
    for i, l in enumerate(lines):
        if not l:
            prev_label = None
            continue
        tag, start, end, token = l
        
        temp = []

        # TODO: warn for multiple, detailed info for non-initial
        label = None
        for o in range(start, end):
            if o in offset_label:
                if o != start:
                    print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                        token, offset_label[o].text), file=sys.stderr)
                label = offset_label[o].type
                break

        if label is not None:
            if label == prev_label:
                tag = 'I-' + label
            else:
                tag = 'B-' + label
        prev_label = label

        lines[i] = [tag, start, end, token]
        
        temp = [idx, start, end, token]

        # print("test3", temp) # conll 
        
        idx += 1
        
        temp_sent.append(temp)
        
    # print("test", temp_sent)
    
    backup_token.append(temp_sent)
        
        

    # optional single-classing
    if options.singleclass:
        for l in lines:
            if l and l[0] != 'O':
                l[0] = l[0][:2] + options.singleclass

    return lines


def process(f):
    return text_to_conll(f)
    #return extract_sent(f)


def process_files(files):
    global options

    nersuite_proc = []

    try:
        for fn in files:
            try:
                if fn == '-':
                    lines = process(sys.stdin)                    
                else:
                    with open(fn, 'rU') as f:
                        lines = process(f)

                # TODO: better error handling
                if lines is None:
                    raise FormatError

                if fn == '-' or not options.outsuffix:
                    sys.stdout.write(''.join(lines))
                else:
                    ofn = path.splitext(fn)[0] + options.outsuffix
                    #print("test_ofn", ofn)
                    with open(ofn, 'wt') as of:
                        m = ''.join(lines)
                        of.write(m)
                        
                        # of.write(''.join(lines))
                        #print("test_lines",m.split('\n\n'))
                        
                        sent_bry_chain = []
                        for i in m.split('\n\n'):
                            #print("test10", i.split('\t')[1],i.split('\t')[-2])
                            # sent_bry.append(i.split('\t')[1]+","+i.split('\t')[-2])
                            sent_bry = []
                            sent_bry.append(int(i.split('\t')[1]))
                            sent_bry.append(int(i.split('\t')[-2]))
                            
                            sent_bry_chain.append(sent_bry)
                            
                        # print("sent_bry",sent_bry)
                        # sent_bry_chain.append(sent_bry.split(',')[0])
                        
                        global_boundary.append(sent_bry_chain)
                        

            except BaseException:
                # TODO: error processing
                raise
    except Exception as e:
        for p in nersuite_proc:
            p.kill()
        if not isinstance(e, FormatError):
            raise

# start standoff processing


TEXTBOUND_LINE_RE = re.compile(r'^T\d+\t')

Textbound = namedtuple('Textbound', 'start end type text')


def parse_textbounds(f):
    """Parse textbound annotations in input, returning a list of Textbound."""
    
    textbounds = []
    relations = []
    locations = []
    dict_rl = {}
    dict_en = {}
        
    for l in f:
        l = l.rstrip('\n')
        a = l.split('\t')
        
        if "T" in a[0]:
            dict_rl[a[0]]= a[1].split(' ')[1] +','+a[1].split(' ')[-1]
            dict_en[a[0]]= a[1].split(' ')[1] +','+a[1].split(' ')[-1] +','+a[1].split(' ')[0]

        if "E" in a[0]:
            for i in a[1].split(' ')[1:]:
                relation = a[1].split(' ')[0].split(":")[0]+'-'+i.split(":")[0]
                location = a[1].split(' ')[0].split(":")[1] +' '+i.split(":")[1]
                # print(location)
                relations.append(relation)
                locations.append(location)

        if not TEXTBOUND_LINE_RE.search(l):
            continue

        id_, type_offsets, text = l.split('\t')
        type_ = type_offsets.split()[0]
        offsets = ' '.join(type_offsets.split()[1:]).split(';')
        
        if len(offsets) > 1:
            #print("test6", offsets[0].split()[0],offsets[1].split()[1], len(offsets))
                #start = int(offset.split()[0])
                #end = int(offset.split()[1])
            start =  int(offsets[0].split()[0])
            end = int(offsets[1].split()[1])
            textbounds.append(Textbound(start, end, type_, text))
        else:
            for offset in offsets:
                start = int(offset.split()[0])
                end = int(offset.split()[1])
                textbounds.append(Textbound(start, end, type_, text))
                
    #print(relations)
    #print(locations)
    #print(dict_en)
    
    sent_en = []
    for key in dict_en:
        temp1 = []
        temp1.append(int(dict_en[key].split(',')[0]))
        temp1.append(int(dict_en[key].split(',')[1]))
        temp1.append(str(dict_en[key].split(',')[2]))
        
        sent_en.append(temp1)
    
    
    sent_rl = []
    for lc,rl in zip(locations,relations):
        temp = []
        trigger_id = dict_rl[lc.split(' ')[0]].split(',')
        argument_id =dict_rl[lc.split(' ')[1]].split(',')
        
        temp.append(int(trigger_id[0]))
        temp.append(int(trigger_id[1]))
        temp.append(int(argument_id[0]))
        temp.append(int(argument_id[1]))
        temp.append(str(rl))
#         print(temp)
        
#         relation_lc = dict_rl[lc.split(' ')[0]]+","+dict_rl[lc.split(' ')[1]]+","+str(rl)
#         print(relation_lc)
        
        sent_rl.append(temp)
    
    global_entities.append(sent_en)
    global_relations.append(sent_rl)
    
    return textbounds


def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check
    for t1 in textbounds:
        for t2 in textbounds:
            if t1 is t2:
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # eliminate shorter
            if t1.end - t1.start > t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[t2] = True
            else:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[t1] = True

    return [t for t in textbounds if t not in eliminate]


def get_annotations(fn):
    global options

    annfn = path.splitext(fn)[0] + options.annsuffix

    with open(annfn, 'rU') as f:
        textbounds = parse_textbounds(f)

    #print("test4", annfn,len(textbounds),textbounds)
    
    #global_entities.append(textbounds)
    
    #textbounds = eliminate_overlaps(textbounds)

    return textbounds

# end standoff processing


# def index_doc():
    

def main(argv=None):
    if argv is None:
        argv = sys.argv

    global options
    options = argparser().parse_args(argv[1:])

    # make sure we have a dot in the suffixes, if any
    if options.outsuffix and options.outsuffix[0] != '.':
        options.outsuffix = '.' + options.outsuffix
    if options.annsuffix and options.annsuffix[0] != '.':
        options.annsuffix = '.' + options.annsuffix
        
    dict_doc = {}

    process_files(options.text)
    # print(type(options.text))
    # print("global_sent", global_sentence)
    # print("global_pos", global_pos)
    # print("global_en",global_entities)
    # print("backup_token", backup_token)
    # print("global_rl",global_relations)
    # print("global_boundary", global_boundary)
    
    
    # trigger_label_list = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']
    # span_only = ['Method','Frequency', 'Amount']
    # argument_label_list = ['StatusTime','StatusEmploy','TypeLiving', 'Type', 'Method', 'History', 'Duration', 'Frequency', 'Amount']
    
    # argument_label_list = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus','StatusTime','StatusEmploy','TypeLiving', 'Type', 'Method', 'History', 'Duration', 'Frequency', 'Amount']
    argument_label_list = ['Drug', 'Alcohol','Tobacco','Employment','LivingStatus']
    # argument_label_list = ['StatusTime','StatusEmploy','TypeLiving', 'Type', 'Method', 'History', 'Duration', 'Frequency', 'Amount']
    
    global_entities_argument_out = []
    
    for argument in global_entities:
        global_entities_argument_in = []
        for argu in argument:
            if argu[2] in argument_label_list:
                global_entities_argument_in.append(argu)
        
        # print(argument,global_entities_argument_in)       
        
        global_entities_argument_out.append(global_entities_argument_in)
    
    # print(global_entities_argument_out) 
    # print(backup_token)

    with open("temp.json", "w") as json_string_save:
        
        count_temp = 0
        max_words = 0
       
        for i,j,k,t,m,bk,ps in zip(options.text,global_sentence,global_entities_argument_out,global_relations,global_boundary,backup_token,global_pos):
            ner_out_sent = []
            relation_out_sent = []
            
            # sent_len = []
            # for sent_j in j:
            #     sent_len.append(len(sent_j))
            # print(sent_len)
            
            for len_j in j:
                count_temp += len(len_j)
                if len(len_j) > max_words:
                    max_words = len(len_j)
                # print(len_j,count_temp)
                # break
                
                

            for m_i in m:
                ner_in_sent = []
                relation_in_sent = []
                sent_start = m_i[0]
                sent_end = m_i[1]
                
                # print(sent_start,sent_end)
                
                
                for k_i in k:
                    entity_temp = []
                    
                    entity_start = k_i[0]
                    entity_end = k_i[1]
 
                    # print("entities", k_i)
                    # first_loc_in_sentence = bk[0]
                    # print(len(j[1]))
                    # print("test",first_loc_in_sentence)

                    for bk_i in bk:

                        idx_loc = bk_i[0]
                        idx_start = bk_i[1]
                        idx_end = bk_i[2]
                        
                        # print(idx_loc,idx_start,idx_end)
                        
                        if sent_start == idx_start:
                            start_idx_sent = idx_loc
                            # print(start_idx_sent)
                        
                        if entity_start == idx_start:
                            entity_temp.append(idx_loc - start_idx_sent)
                            
                        if entity_end == idx_end:
                            entity_temp.append(idx_loc+1 - start_idx_sent) # change here for triaffine-nested-ner
                            
                    entity_temp.append(k_i[2])
                       
                    # print('indexed in the document', entity_temp)
                    
                    if entity_start>=sent_start and entity_end <= sent_end:
                        
                        dict_temp = {}
                        # print(entity_temp)
                        dict_temp['type'] = entity_temp[2]
                        dict_temp['start'] = entity_temp[0]
                        dict_temp['end'] = entity_temp[1]
                        
                        # print(dict_temp)

                        ner_in_sent.append(dict_temp)   #k_i  

                # print(i,ner_in_sent)
                ner_out_sent.append(ner_in_sent)
                
                # relation 如果跨行了怎么办， 以 trigger为主
                
                for t_i in t:
                    
                    
                    trigger_start = t_i[0]
                    trigger_end = t_i[1]
                    argument_start = t_i[2]
                    argument_end = t_i[3]
                    # print(t_i,trigger_start,trigger_end,argument_start,argument_end)
                    
                    relation_temp = []
                    for bk_i in bk:
                        
                        idx_loc = bk_i[0]
                        idx_start = bk_i[1]
                        idx_end = bk_i[2]
                        
                        if trigger_start == idx_start:
                            trigger_start_loc = idx_loc
                            
                        if trigger_end == idx_end:
                            trigger_end_loc = idx_loc
                            
                        if argument_start == idx_start:
                            argument_start_loc = idx_loc
                        
                        if argument_end == idx_end:
                            argument_end_loc = idx_loc
                            
                    # print(trigger_start_loc,trigger_end_loc,argument_start_loc,argument_end_loc)
                    relation_temp.append(trigger_start_loc)
                    relation_temp.append(trigger_end_loc)
                    relation_temp.append(argument_start_loc)
                    relation_temp.append(argument_end_loc)
                    relation_temp.append(t_i[4])
                    
                    # print(relation_temp)
                    
                    if trigger_start>=sent_start and trigger_end <= sent_end:
                        # print(relation_start,sent_start,relation_end,sent_end)
                        relation_in_sent.append(relation_temp) #t_i
                
                # print("relation_in_sent",relation_in_sent)
                relation_out_sent.append(relation_in_sent)

            # print(ner_out_sent)
            # print(count_temp)
            
                
            dict_doc["entities"] = ner_out_sent
            dict_doc["relations"] = relation_out_sent
            dict_doc["sentences"] = j
            dict_doc["org_id"] = i
            dict_doc["pos"] = ps
            
            # dict_doc["boundary"] = m
            
            
            

            # print(i,m_i[0],m_i[1])

            json_string = json.dumps(dict_doc)
            json_string_save.write(json_string)
            json_string_save.write('\n')
            # print(i)
    
    print(count_temp,max_words)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))
