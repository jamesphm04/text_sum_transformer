from dataset import NewsSumDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path
from tqdm import tqdm
import warnings
import os
import pandas as pd

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, text_type):
    for item in ds:
        yield item[text_type] #load just a function, call by 'next' if needed (generator)

def get_or_build_tokenizer(config, text_type, ds={}):
    tokenizer_path = Path(config['tokenizer_file'].format(text_type)) #'/path/to/tokenizer_{text_type}.txt' 
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, text_type), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_ds(config):
    # only has the train -> divide train, test, valid by ourself
    ds_raw = load_dataset(f"{config['datasource']}", '1.0.0', split='train[:5%]')
    
    #cleaning the dataset
    df = pd.DataFrame(ds_raw)
    df = df.drop('id', axis=1)
    
    df['count_article'] = df['article'].apply(lambda x: len(str(x).split()))
    df['count_highlights'] = df['highlights'].apply(lambda x: len(str(x).split()))
    
    df = df[(df['count_article'] <= 300)]
    df = df[(df['count_highlights'] <= 50)]
    
    df = df.drop(['count_highlights', 'count_article'], axis=1)
    
    df = df.reset_index(drop=True)
    
    print('Len of dataset: ', len(df['article']))
    
    ds_raw = Dataset.from_pandas(df)
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, config['text_src'], ds_raw)
    tokenizer_tgt = get_or_build_tokenizer(config, config['text_tgt'], ds_raw)
    
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = NewsSumDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['text_src'], config['text_tgt'], config['src_seq_len'], config['tgt_seq_len'])
    val_ds = NewsSumDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['text_src'], config['text_tgt'], config['src_seq_len'], config['tgt_seq_len'])
    
    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['text_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['text_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["src_seq_len"], config['tgt_seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    #Define the device 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f'Using device {device}')
    
    #Make sure the weigths folder exists
    Path(f'{config["datasource"]}_{config["model_folder"]}').mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # if the user specified a model a preload before training, load it
    initial_epoch = 0 
    global_step = 0 
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, stating from scratch')
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # label_smoothing -> less sure -> distribute to the others pred -> less overfit
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch: {epoch:02d}')
        for batch in batch_iterator:
            
            # model.train() #just for testing validation function
            
            encoder_input = batch['encoder_input'].to(device) #(b, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(b, seb_len)
            encoder_mask = batch['encoder_mask'].to(device) #(b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(b, 1, 1, seq_len)

            #Run the tensors through the encoder and decoder and the projection laer 
            encoder_output = model.encode(encoder_input, encoder_mask) #(b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(b, seq_len, d_model)
            proj_output = model.project(decoder_output)
            
            #compare the output with the label 
            label = batch['label'].to(device) #(b, seq_len)
            
            #compute the loss using the simple cross entrophy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})
            
            #backpropagate the loss 
            loss.backward()
            
            #update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)    
            
            global_step += 1
            
        #run validation at the end of each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)
        #save the model at the end of every epoch 
        if epoch % 20 == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every step 
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
    
        # build mask for the target 
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # calculate output 
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # get next token
        prob = model.project(out[:, -1])
        #select the token with the max probability (greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)] # appending the nextword to the decoder_input
        , dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    try:
        # get the console window width 
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read.split()
            console_width = int(console_width)
    except:
        # if we can not get the console width, use 80 as default
        console_width = 80
    
    with torch.no_grad(): # no training
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            #check that the batch size is 1 
            assert encoder_input.size(0) == 1, 'Batch size must be 1 for validation'
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)