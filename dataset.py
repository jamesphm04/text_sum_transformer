import torch 
import torch.nn as nn
from torch.utils.data import Dataset 

class NewsSumDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_text, tgt_text, src_seq_len, tgt_seq_len):
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_text = src_text
        self.tgt_text = tgt_text
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds) 
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_text]
        tgt_text = src_target_pair[self.tgt_text]
        
        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids 
        
        #Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.src_seq_len - len(enc_input_tokens) - 2 # sos and eos 
        dec_num_padding_tokens = self.tgt_seq_len - len(dec_input_tokens) - 1 # only sos
              
        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Add sos and eos token to the encoder
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])
        
        # Add sos token to the decoder 
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        
        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len
        
        return {
            'encoder_input': encoder_input,  # (seq_len)
            'decoder_input': decoder_input,  # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) tokens are NOT pad is OK. add 1 dim for sequence dimension and 1 dim for batch dimension => compatible with expected shape for the attention mask
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            'label': label,  # (seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) # extract all value in the TRIangle part (after) of (1, seq_len, seq_len)
    return mask == 0 # return True for all masked elements (Tri)