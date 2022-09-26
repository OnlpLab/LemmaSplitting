# The code is partially inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/Seq2Seq_attention
from copy import deepcopy
from os import listdir, scandir
from os.path import basename, isfile, join, split, splitext

import matplotlib.ticker as ticker
import numpy as np
import torch
from editdistance import eval as eval_edit_distance
from matplotlib import pyplot as plt
from torchtext.legacy.data import Field

INFLECTION_STR, REINFLECTION_STR = 'inflection', 'reinflection'

srcField = Field(tokenize=lambda x: x.split(','), init_token="<sos>", eos_token="<eos>")
trgField = Field(tokenize=lambda x: x.split(','), init_token="<sos>", eos_token="<eos>")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_and_log(log_file, string):
    print(string)
    open(log_file, 'a+', encoding='utf8').write(string + '\n')

def translate_sentence(model, sentence, german, english, device, max_length=50, return_attn=False):
    assert type(sentence) == list
    tokens = deepcopy(sentence)

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]
    attention_matrix = torch.zeros(max_length, max_length)
    for i in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hiddens, cells, attn = model.decoder(previous_word, outputs_encoder, hiddens, cells, return_attn=return_attn)
            best_guess = output.argmax(1).item()
            if return_attn: attention_matrix[i] = torch.cat((attn.squeeze(), torch.zeros(max_length - attn.shape[0]).to(device)), dim=0)

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    if return_attn:
        return translated_sentence[1:], attention_matrix[:len(translated_sentence) + 2, :len(sentence) + 2]
    else:
        return translated_sentence[1:]


def evaluate_model(data, model, german, english, device):
    targets, outputs = [], []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    # Count also Accuracy. Ignore <eos>, obviously.
    targets = [t[0] for t in targets]
    acc = np.array([a == b for a, b in zip(targets, outputs)]).mean()
    res = np.mean([eval_edit_distance(t, o) for t, o in zip(targets, outputs)])

    return res, acc


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, verbose=True):
    if verbose: print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def reinflection2sample(line, mode=REINFLECTION_STR):
    assert mode in {REINFLECTION_STR, INFLECTION_STR}
    if mode == REINFLECTION_STR:
        # The line format is [src_feat, src_form, trg_feat, trg_form]
        src_feat, src_form, trg_feat, trg_form = line
        src_feat, trg_feat = src_feat.split(";"), trg_feat.split(";")
        src_form, trg_form = list(src_form), list(trg_form)
        src = ','.join(src_feat + ['+'] + src_form + ['+'] + trg_feat)
        trg = ','.join(trg_form)
    else:  # inflection mode
        lemma, form, feat = line
        feat = feat.split(";")
        lemma, form = list(lemma), list(form)
        src = ','.join(lemma + ['$'] + feat)  # Don't use '+' as it is part of some tags (see the file tags.yaml).
        trg = ','.join(form)
    return src, trg

def convert_file_to_tsv(file_name, new_file_name, mode):
    data = open(file_name, encoding='utf8').read().split('\n')
    data = [line.split('\t') for line in data]

    examples = []
    for e in data:
        if e[0] == '': continue
        src, trg = reinflection2sample(e, mode=mode)
        examples.append(f"{src}\t{trg}")

    open(new_file_name, mode='w', encoding='utf8').write('\n'.join(examples))


def reinflection2TSV(file_name, dir_name="data", mode=REINFLECTION_STR):
    """
    Convert a file in the Reinflection format (src_feat\tsrc_form\ttrg_feat\ttrg_form) to a TSV file of the format
    src\ttrg, each one consists of CSV strings. If mode=inflection, then count the data as SIGMORPHON format, and file_name must be a tuple of 3 paths.
    :param mode: can be either 'inflection' or 'reinflection'.
    :param dir_name:
    :param file_name: if mode=reinflection, then file_name: str. else, file_name:Tuple(str)
    :return: The two paths of the TSV files.
    """
    assert mode in {REINFLECTION_STR, INFLECTION_STR}

    if mode == REINFLECTION_STR:
        file_name = join(dir_name, file_name)
        new_fn = f"{splitext(file_name)[0]}.tsv"
        convert_file_to_tsv(file_name, new_fn, mode)
    else:
        train_fn, test_fn = file_name[0], file_name[2]  # file paths without parent-directories prefix
        new_train_fn = join(dir_name, f"{basename(train_fn)}.tsv")  # use the paths without parent-directories prefixes
        new_test_fn = join(dir_name, f"{basename(test_fn)}.tsv")
        if isfile(new_train_fn) and isfile(new_test_fn): return [new_train_fn, new_test_fn]
        convert_file_to_tsv(train_fn, new_train_fn, mode)
        convert_file_to_tsv(test_fn, new_test_fn, mode)
        new_fn = [new_train_fn, new_test_fn]
        # The result is a directory "data\\LEMMA_TSV_FORMAT" with 180 files of the format 'language.{trn|tst}.tsv'
    return new_fn


def get_languages_and_paths(data_dir=''):
    """
    Return a list of the languages, and a dictionary of tuples: {language: (train_path,dev_path,test_paht)}.
    :param data_dir:
    :return:
    """
    train_dirs = ['DEVELOPMENT-LANGUAGES', 'SURPRISE-LANGUAGES']
    test_dir = 'GOLD-TEST'
    train_dirs, test_dir = [join(data_dir, d) for d in train_dirs], join(data_dir, test_dir)
    print(f"Requirements: this script must have the same path as the folders of SIGMORPHON"
          f" (SURPRISE-LANG., GOLD-TEST, etc.). The data folder's name should be {data_dir}.\n")

    test_paths = listdir(test_dir)
    langs = [splitext(p)[0] for p in test_paths]
    test_paths = {l: p for l, p in zip(langs, [join(test_dir, s) for s in test_paths])}

    dev_families = [f.path for f in scandir(train_dirs[0]) if f.is_dir()]
    surprise_families = [f.path for f in scandir(train_dirs[1]) if f.is_dir()]

    dev_families.sort()
    surprise_families.sort()

    develop_paths, surprise_paths, test_no_gold_paths = {}, {}, {}
    language2family = {}
    for family in dev_families + surprise_families:
        for file in listdir(family):
            language, ext = splitext(file)
            file = join(family, file)
            if ext == '.trn':
                develop_paths[language] = file
            elif ext == '.dev':
                surprise_paths[language] = file
            elif ext == '.tst':
                test_no_gold_paths[language] = file
            family_name = split(family)[1]
            if language not in language2family: language2family[language] = family_name

    files_paths = {k: (develop_paths[k], surprise_paths[k], test_paths[k]) for k in langs}
    return langs, files_paths, language2family


def showAttention(input_sentence, output_words, attentions, fig_name="Attention Weights.png"):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence + ['<eos>'])  # , rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.show()
    plt.savefig(fig_name)

def save_run_results_figure(file_path, edit_distances, accuracies):
    plt.figure()
    plt.subplot(211)
    plt.title("Average ED on Test Set")
    plt.plot(edit_distances)
    plt.subplot(212)
    plt.title("Average Accuracy on Test Set")
    plt.plot(accuracies)
    plt.savefig(file_path)
