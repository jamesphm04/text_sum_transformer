from pathlib import Path 
from config import get_config
from tokenizers import Tokenizer
import torch 
from model import build_transformer
from config import get_config



def export_to_onnx():
    
    config = get_config()
    seq_len = config['seq_len']

    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_src_len = 18368
    vocab_tgt_len = 6858

    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    model_path = 'cnn_dailymail_weights/tmodel_97.pt'
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    device = 'cpu'
    config = get_config()


    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['text_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['text_tgt']))))


    source = tokenizer_src.encode('Hi, my name is James')
    source = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        torch.tensor(source.ids, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64) 
    ], dim=0).to(device)
    decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
    source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
    decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)

    onnx_program = torch.onnx.export(model, (source, decoder_input, source_mask, decoder_mask), 'summerizer_model.onnx')
    
    return onnx_program