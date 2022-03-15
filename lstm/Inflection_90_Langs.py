import random
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import spacy
import os
import pandas as pd

import utils
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, get_langs_and_paths
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
# from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from Network import Encoder, Decoder, Seq2Seq
from utils import srcField, trgField, device, reinflection2TSV, plt, showAttention, REINFLECTION_STR, INFLECTION_STR
def concat_to_file(fn, s):
    with open(fn, "a+", encoding='utf8') as f: f.write(s)
total_timer = datetime.now()
# Definition of tokenizers, Fields and device were moved to utils

datafields = [("src", srcField), ("trg", trgField)]

# Generate new datasets for Inflection:
training_mode_90langs = 'FORM' # choose either 'FORM' or 'LEMMA'.
data_dir = os.path.join('data',f'{training_mode_90langs}-SPLIT')
tsv_dir = os.path.join('data',f'{training_mode_90langs}_TSV_FORMAT')

langs, files_paths, lang2family = get_langs_and_paths(data_dir=data_dir)
if not os.path.exists(f'SIG20.{training_mode_90langs}'): os.mkdir(f'SIG20.{training_mode_90langs}')
if not os.path.exists(tsv_dir): os.mkdir(tsv_dir)
results_df = pd.DataFrame(columns=["Family", "Language", "Accuracy", "ED"])

langs1 = ['tgk', 'dje', 'mao', 'lin', 'xno', 'lud', 'zul', 'sot', 'vro', 'ceb', 'mlg', 'gmh', 'kon', 'gaa', 'izh', 'mwf', 'zpv', 'kjh', 'hil', 'gml', 'tel', 'vot', 'czn', 'ood', 'mlt', 'gsw',
'orm', 'tgl', 'sna', 'frr', 'syc', 'xty', 'ctp', 'dak', 'liv', 'aka', 'ben', 'nya', 'cly', 'swa', 'lug', 'bod', 'kan', 'kir', 'cre', 'pus', 'lld', 'ast', 'crh', 'cpa', 'uig', 'fur', 'evn',
'aze', 'kaz', 'azg', 'urd', 'bak']
langs2 = ['pei', 'nno', 'vec', 'nob', 'dan', 'tuk', 'otm', 'ote', 'san', 'glg', 'frm', 'uzb', 'fas', 'est']
langs3 = ['ang', 'hin', 'nld', 'sme', 'olo', 'mdf', 'cat', 'isl', 'swe', 'kpv', 'mhr']
langs4 = ['myv', 'krl', 'eng', 'udm', 'vep', 'fin', 'deu']
all_langs = [langs1, langs2, langs3, langs4]
choice = 1
langs = all_langs[choice-1]
log_file = os.path.join('SIG20', training_mode_90langs, f'log_file{choice}.txt')

for j, lang in enumerate(langs):
    lang_t0 = datetime.now()
    starter_s = f"Starting to train a new model on Language={lang}, from Family={lang2family[lang]}, at {str(datetime.now())}\n"
    concat_to_file(log_file, starter_s)
    print(starter_s)
    outputs_dir = os.path.join('SIG20', training_mode_90langs, lang)
    if not os.path.exists(os.path.join('SIG20', training_mode_90langs)): os.mkdir(os.path.join('SIG20', training_mode_90langs))
    # Add here the datasets creation, using TabularIterator (add custom functions for that)
    train_file, test_file = reinflection2TSV(files_paths[lang], dir_name=tsv_dir, mode=INFLECTION_STR)
    train_data, test_data = TabularDataset.splits(path="", train=train_file, test=test_file, fields=datafields, format='tsv')

    print("- Building vocabularies")
    srcField.build_vocab(train_data) # no limitation of max_size or min_freq is needed.
    trgField.build_vocab(train_data) # no limitation of max_size or min_freq is needed.

    print("- Starting to train the model:")
    print("- Defining hyper-params")

    ### We're ready to define everything we need for training our Seq2Seq model ###
    load_model = False
    save_model = True

    # Training hyperparameters
    num_epochs = 50 # if choice in {1,2} else 10
    learning_rate = 3e-4
    batch_size = 32

    # Model hyperparameters
    input_size_encoder = len(srcField.vocab)
    input_size_decoder = len(trgField.vocab)
    output_size = len(trgField.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 256
    num_layers = 1
    enc_dropout = 0.0
    dec_dropout = 0.0
    measure_str = 'Edit Distance'
    comment = f"epochs={num_epochs} lr={learning_rate} batch={batch_size} embed={encoder_embedding_size} hidden_size={hidden_size}"

    print(f"Hyper-Params: {comment}")
    concat_to_file(log_file, f"Hyper-Params: {comment}\n")
    print("- Defining a SummaryWriter object")
    # Tensorboard to get nice loss plot
    writer = SummaryWriter(os.path.join(outputs_dir,"runs"), comment=comment)
    step = 0

    print("- Generating BucketIterator objects")
    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    print("- Constructing networks")
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)

    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout,).to(device)
    model = Seq2Seq(encoder_net, decoder_net).to(device)
    print("- Defining some more stuff...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = srcField.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load(os.path.join(outputs_dir,"my_checkpoint.pth.tar")), model, optimizer)

    random.seed(42)
    indices = random.sample(range(len(test_data)), k=10)
    accs, eds = [], []
    # examples_for_printing = random.sample(test_data.examples,k=10)
    # validation_sentences = test_data.examples[indices]

    print("Let's begin training!\n")
    concat_to_file(log_file,"Training...\n")
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]  (lang={lang})")

        if save_model:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint, os.path.join(outputs_dir, "my_checkpoint.pth.tar"))

        model.train()

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        model.eval()
        examples_for_printing = [test_data.examples[i] for i in indices] # For a more sufficient evaluation, we apply translate_sentence on 10 samples.
        translated_sentences = [translate_sentence(model, ex.src, srcField, trgField, device, max_length=50) for ex in examples_for_printing]
        # print(f"Translated example sentence: \n {translated_sentences}")
        for i,translated_sent in enumerate(translated_sentences):
            ex = examples_for_printing[i]
            if translated_sent[-1]=='<eos>': translated_sent = translated_sent[:-1]
            src_print = ''.join(ex.src)
            trg_print = ''.join(ex.trg)
            pred_print = ''.join(translated_sent)
            ed_print = utils.editDistance(trg_print, pred_print)
            print(f"{i+1}. input: {src_print} ; gold: {trg_print} ; pred: {pred_print} ; ED = {ed_print}")
        result, accuracy = bleu(test_data, model, srcField, trgField, device, measure_str=measure_str)
        writer.add_scalar("Test Accuracy", accuracy, global_step=epoch)
        print(f"avgED = {result}; avgAcc = {accuracy}\n")
        accs.append(accuracy)
        eds.append(result)

    # running on entire test data takes a while
    # score = bleu(test_data[1:100], model, srcField, trgField, device, measure_str='ed')
    result, accuracy = bleu(test_data, model, srcField, trgField, device, measure_str=measure_str)
    lang_runtime = datetime.now() - lang_t0
    output_s = f"Results for Language={lang} from Family={lang2family[lang]}: {measure_str} score on test set" \
               f" is {result:.2f}. Average Accuracy is {accuracy:.2f}. Elapsed time is {lang_runtime}.\n\n"
    concat_to_file(log_file, output_s) # write to log file
    print(output_s) # write to screen
    results_df.loc[j] = [lang2family[lang], lang, np.round(accuracy,2), np.round(result,2)] # write to Excel file

    plt.figure()
    plt.subplot(211)
    plt.title("Average ED on Test Set")
    plt.plot(eds)
    plt.subplot(212)
    plt.title("Average Accuracy on Test Set")
    plt.plot(accs)
    plt.savefig(os.path.join(outputs_dir, "Results.png"))

tot_runtime_s = f'\nTotal runtime: {str(datetime.now() - total_timer)}\n'

concat_to_file(log_file, tot_runtime_s)
print(tot_runtime_s)

accs, eds = results_df['Accuracy'], results_df['ED']
avgAcc, avgED, medAcc, medED = np.mean(accs), np.mean(eds), np.median(accs), np.median(eds)
final_stats_s = f"avgAcc={avgAcc:.2f}, avgED={avgED:.2f}, medAcc={medAcc:.2f}, medED={medED:.2f}\n"
concat_to_file(log_file, final_stats_s)
print(final_stats_s)
results_df.to_excel(f"ResultsFile{len(langs)}Langs{choice}.{training_mode_90langs}.xlsx")