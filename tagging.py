import argparse
import io
import json
import logging
import os

import dill
import torch
import torchtext
import yaml
from torch.utils.data import DataLoader
from torchtext.data import Dataset

from kma.common.util import syllable_to_eojeol
from kma.dataset import POSExample
from kma.decoders.rnn_decoder import RNNDecoderPointer
from kma.encoders.rnn_encoder import RNNEncoder
from kma.models.model import KMAModel
from kma.taggers.crf_tagger import CRFTagger


def parse_option():
    option = argparse.ArgumentParser(description='Korean morphological analyzer')
    option.add_argument('--input_file', type=str, required=True)
    option.add_argument('--output_file', type=str, required=True)
    return option.parse_args()


with open(os.path.join('config', 'kma.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

args = parse_option()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
logger.info('Device: %s' % device)

with open(config['vocab_name'], 'rb') as fin:
    vocab = dill.load(fin)
WORD, LEX, POS_TAG = vocab['WORD'], vocab['LEX'], vocab['POS']

encoder = RNNEncoder(vocab_size=len(WORD.vocab), pad_id=WORD.vocab.stoi[WORD.pad_token], **config['encoder'])
if config['encoder']['bidirectional']:
    hidden_size = config['encoder']['hidden_size'] * 2
else:
    config['encoder']['hidden_size']
decoder = RNNDecoderPointer(vocab_size=len(LEX.vocab), hidden_size=hidden_size,
                            sos_id=LEX.vocab.stoi[LEX.init_token], eos_id=LEX.vocab.stoi[LEX.eos_token],
                            pad_id=LEX.vocab.stoi[LEX.pad_token], **config['decoder'])
tagger = CRFTagger(hidden_size=hidden_size, num_tags=len(POS_TAG.vocab))

model = KMAModel(encoder, decoder, tagger).to(device)
checkpoint = torch.load(config['model_name'], map_location=device)
model.load_state_dict(checkpoint['model'])
logger.info(model)
model.eval()

with io.open(args.input_file, 'r', encoding='utf-8') as f:
    sents = f.readlines()

Default = torchtext.data.RawField()
named_fields = [('word', WORD), ('raw', Default)]
examples = []
for sent in sents:
    syllables = [list(eojeol) for eojeol in sent.split()]
    examples.append(POSExample.fromsent(syllables, named_fields))

text_dataset = Dataset(examples, named_fields)
text_iter = torchtext.data.BucketIterator(text_dataset, batch_size=5, shuffle=False)

outputs = []
sentsit = iter(sents)

print_iters = len(examples) // 100

with torch.no_grad():
    for t in text_iter:
        decoder_outputs, tagger_outputs, others = model.infer(t.word[0].to(device))

        for n in range(len(others['length'])):
            length = others['length'][n]
            tgt_id_seq = [others['sequence'][di][n].item() for di in range(length)]
            tag_id_seq = [tagger_outputs[n][di].item() for di in range(length)]

            sent = next(sentsit)
            result = []

            for i, (tgt, pos) in enumerate(zip(tgt_id_seq, tag_id_seq)):
                pos = POS_TAG.vocab.itos[pos]
                lex = WORD.vocab.itos[tgt]

                _, idx = others['attention_score'][i][n].max(1)
                attn_idx = idx.item() - 1
                raw = t.raw[n][attn_idx] if 0 <= attn_idx < len(t.raw[n]) else None
                text = ""

                if pos == "<blank>" or lex == "<blank>":
                    # Skip whitespace.
                    s = sent
                    sent = sent.lstrip()
                    text = s[:len(s) - len(sent)]
                elif raw and sent.startswith(raw):
                    # Take raw string.
                    text = raw
                    sent = sent[len(raw):]
                elif raw and result and result[-1][0] == raw and result[-1][1] == "<tok>":
                    # Move text from (previous) raw token to normal (current) token.
                    result[-1] = "", *result[-1][1:]
                    text = raw
                elif raw == lex and result and result[-1][0] == raw and result[-1][0] != result[-1][1]:
                    # Sometimes the `raw` value gets assigned too eagerly, but here we found a better
                    # match so we're assigning it here.
                    result[-1] = "", *result[-1][1:]
                    text = raw

                if tgt == WORD.vocab.stoi[WORD.unk_token] and config['replace_unk']:
                    result.append((text, raw, pos))
                else:
                    result.append((text, lex, pos))

            if sent.rstrip() and not sent.rstrip(" .?\n"):
                result.append((sent.rstrip(), sent.rstrip(), "SF"))
            elif len(sent.rstrip()) != 0:
                print(f"cannot fully parse sentence: {sent!r} vs {result}")
                result = []

            outputs.append(result)

            if len(examples) > 1000 and len(outputs) % print_iters == 0:
                print("{:.1f}%".format(len(outputs) / len(examples) * 100))

def _merge(tokens: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Merges subsequent tokens that represent the same POS."""
    result = []
    curr_token, curr_raw, curr_pos = "", "", None

    for raw, lex, pos in tokens + [("", "<eos>", "<eos>")]:  # Make sure we end the loop correctly.

        # End of sentence.
        if "<eos>" in (lex, pos):
            if not curr_token:
                continue

            result.append((curr_raw, curr_token, curr_pos))
            curr_token, curr_raw, curr_pos = "", "", None
            continue

        # Blank characters.
        if lex in ("<blank>", "<tok>"):
            if curr_token:
                result.append((curr_raw, curr_token, curr_pos))
                curr_token, curr_raw, curr_pos = "", "", None

            if lex == "<tok>" and raw:
                curr_raw += raw
            elif lex == "<blank>":
                result.append((raw, "<blank>", "<blank>"))

            continue

        curr_raw += raw
        curr_token += lex
        curr_pos = pos if curr_pos is None else curr_pos

    if curr_raw in ".?":
        result.append((curr_raw, curr_raw, "SF"))

    return result

with io.open(args.output_file, 'w', encoding='utf-8') as f:
    prefix = "[\n"

    for output in outputs:
        f.write(prefix)
        prefix = ",\n"
        json.dump(_merge(output), f, ensure_ascii=False)

    f.write("\n]")
