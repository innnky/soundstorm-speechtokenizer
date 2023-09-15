import torch

from vits_decoder import utils
from vits_decoder.models import SynthesizerTrn


def get_model(model_path='vits_decoder/G.pth', config_path='vits_decoder/config.json'):
    hps = utils.get_hparams_from_file(config_path)

    decoder = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = decoder.eval()
    try:
        _ = utils.load_checkpoint(model_path, decoder, None)
    except:
        print('Failed to load model, please check the model path')
    return decoder

def decode_tokens(model, tokens):
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    with torch.no_grad():
        length = torch.LongTensor([tokens.shape[-1]]).to(device)
        wav = model.infer(tokens, length)[0]
    return wav.cpu().numpy().squeeze(0).squeeze(0)


# if __name__ == '__main__':
#     model = get_model()
#     tokens = torch.zeros(1, 8, 220).long()
#     wav = decode_tokens(model, tokens)
#     print(wav.shape)
