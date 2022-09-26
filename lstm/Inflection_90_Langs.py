import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from os import mkdir
from os.path import join, exists
import pandas as pd

import utils
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, get_languages_and_paths,\
    save_run_results_figure, print_and_log
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.legacy.data import BucketIterator, TabularDataset
from Network import Encoder, Decoder, Seq2Seq
from utils import srcField, trgField, device, reinflection2TSV, INFLECTION_STR

total_timer = datetime.now()

# Generate new datasets for Inflection:
training_mode = 'FORM' # choose either 'FORM' or 'LEMMA'.
data_dir = os.path.join('data', f'{training_mode}-SPLIT')
tsv_dir = os.path.join('data', f'{training_mode}_TSV_FORMAT')

languages, files_paths, language2family = get_languages_and_paths(data_dir=data_dir)

if not exists(f'SIG20.{training_mode}'): mkdir(f'SIG20.{training_mode}')
if not exists(tsv_dir): mkdir(tsv_dir)

results_df = pd.DataFrame(columns=["Family", "Language", "Accuracy", "ED"])

languages1 = ['tgk', 'dje', 'mao', 'lin', 'xno', 'lud', 'zul', 'sot', 'vro', 'ceb', 'mlg', 'gmh', 'kon', 'gaa', 'izh', 'mwf', 'zpv', 'kjh', 'hil', 'gml', 'tel', 'vot', 'czn', 'ood', 'mlt', 'gsw',
'orm', 'tgl', 'sna', 'frr', 'syc', 'xty', 'ctp', 'dak', 'liv', 'aka', 'ben', 'nya', 'cly', 'swa', 'lug', 'bod', 'kan', 'kir', 'cre', 'pus', 'lld', 'ast', 'crh', 'cpa', 'uig', 'fur', 'evn',
'aze', 'kaz', 'azg', 'urd', 'bak']
languages2 = ['pei', 'nno', 'vec', 'nob', 'dan', 'tuk', 'otm', 'ote', 'san', 'glg', 'frm', 'uzb', 'fas', 'est']
languages3 = ['ang', 'hin', 'nld', 'sme', 'olo', 'mdf', 'cat', 'isl', 'swe', 'kpv', 'mhr']
languages4 = ['myv', 'krl', 'eng', 'udm', 'vep', 'fin', 'deu']
all_languages = [languages1, languages2, languages3, languages4]
choice = 1
languages = all_languages[choice-1]
log_file = join('SIG20', training_mode, f'log_file{choice}.txt')

for j, language in enumerate(languages):
    language_t0 = datetime.now()

    print_and_log(log_file, f"Starting to train a new model on Language={language},"
                            f" from Family={language2family[language]}, at {str(datetime.now())}\n")

    outputs_dir = join('SIG20', training_mode, language)
    if not exists(join('SIG20', training_mode)): mkdir(join('SIG20', training_mode))
    # Add here the datasets creation, using TabularIterator (add custom functions for that)
    train_file, test_file = reinflection2TSV(files_paths[language], dir_name=tsv_dir, mode=INFLECTION_STR)
    train_data, test_data = TabularDataset.splits(path="", train=train_file, test=test_file,
                                                  fields=[("src", srcField), ("trg", trgField)], format='tsv')

    print("- Building vocabularies")
    srcField.build_vocab(train_data) # no limitation of max_size or min_freq is needed.
    trgField.build_vocab(train_data) # no limitation of max_size or min_freq is needed.

    print("- Starting to train the model:")
    print("- Defining hyper-params")

    ### We're ready to define everything we need for training our Seq2Seq model ###
    load_model = False
    save_model = True

    # Training hyperparameters
    num_epochs = 50
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

    print_and_log(log_file, f"Hyper-Params: {comment}")
    print("- Defining a SummaryWriter object")
    # Tensorboard to get nice loss plot
    writer = SummaryWriter(join(outputs_dir,"runs"), comment=comment)
    step = 0

    print("- Generating BucketIterator objects")
    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)

    print("- Constructing networks")
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    model = Seq2Seq(encoder_net, decoder_net).to(device)

    print("- Defining some more stuff...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = srcField.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load(join(outputs_dir,"my_checkpoint.pth.tar")), model, optimizer)

    random.seed(42)
    indices = random.sample(range(len(test_data)), k=10)
    accs, eds = [], []
    # examples_for_printing = random.sample(test_data.examples,k=10)
    # validation_sentences = test_data.examples[indices]

    print_and_log(log_file, "Training...\n")
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]  (language={language})")

        if save_model:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint, join(outputs_dir, "my_checkpoint.pth.tar"))

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
            ed_print = utils.eval_edit_distance(trg_print, pred_print)
            print(f"{i+1}. input: {src_print} ; gold: {trg_print} ; pred: {pred_print} ; ED = {ed_print}")

        edit_distance, accuracy = bleu(test_data, model, srcField, trgField, device)
        writer.add_scalar("Test Accuracy", accuracy, global_step=epoch)
        print(f"avgED = {edit_distance}; avgAcc = {accuracy}\n")
        accs.append(accuracy)
        eds.append(edit_distance)

    # running on entire test data takes a while
    # score = bleu(test_data[1:100], model, srcField, trgField, device)
    edit_distance, accuracy = bleu(test_data, model, srcField, trgField, device)
    language_runtime = datetime.now() - language_t0

    print_and_log(log_file, f"Results for Language={language} from Family={language2family[language]}: "
                            f"Edit Distance score on test set is {edit_distance:.2f}. Average Accuracy is "
                            f"{accuracy:.2f}. Elapsed time is {language_runtime}.\n\n")
    results_df.loc[j] = [language2family[language], language, np.round(accuracy,2), np.round(edit_distance, 2)] # write to Excel file

    save_run_results_figure(join(outputs_dir, "Results.png"), eds, accs)

print_and_log(log_file, f'\nTotal runtime: {str(datetime.now() - total_timer)}\n')

accs, eds = results_df['Accuracy'], results_df['ED']
avgAcc, avgED, medAcc, medED = np.mean(accs), np.mean(eds), np.median(accs), np.median(eds)
print_and_log(log_file, f"avgAcc={avgAcc:.2f}, avgED={avgED:.2f}, medAcc={medAcc:.2f}, medED={medED:.2f}\n")
results_df.to_excel(f"ResultsFile{len(languages)}Langs{choice}.{training_mode}.xlsx")
