import torch

from train import get_or_build_tokenizer, greedy_decode
from config import get_config

def predict(model, input_text):
    device = 'cpu'
    config = get_config()
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, text_type=config['text_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, text_type=config['text_tgt'])
    
    # input_data = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['text_src'], config['text_tgt'], config['seq_len'])
    
    # Transform the text into tokens
    enc_input_tokens = tokenizer_src.encode(input_text).ids
    
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
    
    #Add sos, eos and padding to each sentence
    enc_num_padding_tokens = config['seq_len'] - len(enc_input_tokens) - 2 # sos and eos 
    
    encoder_input = torch.cat([
    sos_token,
    torch.tensor(enc_input_tokens, dtype=torch.int64),
    eos_token,
    torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)
])
    
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    
    
    model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
    model_out_text = tokenizer_src.decode(model_out.detach().cpu().numpy())
    return {'src': input_text, 'predict': model_out_text}