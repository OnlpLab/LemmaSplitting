import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from os import mkdir
from os.path import join, exists
import pandas as pd


from configs import training_mode, data_dir, tsv_dir, languages, log_file, load_model, save_model, num_epochs, \
    learning_rate, batch_size, encoder_embedding_size, decoder_embedding_size, hidden_size, num_layers, \
    encoder_dropout, decoder_dropout, comment, excel_results_file
from utils import translate_sentence, evaluate_model, save_checkpoint, load_checkpoint, get_languages_and_paths, \
    save_run_results_figure, srcField, trgField, device, eval_edit_distance, reinflection2TSV, INFLECTION_STR, \
    print_and_log
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.legacy.data import BucketIterator, TabularDataset
from Network import Seq2Seq

total_timer = datetime.now()

_, files_paths, language2family = get_languages_and_paths(data_dir=data_dir)

if not exists(f'SIG20.{training_mode}'): mkdir(f'SIG20.{training_mode}')
if not exists(tsv_dir): mkdir(tsv_dir)

results_df = pd.DataFrame(columns=["Family", "Language", "Accuracy", "ED"])

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
    print("- Using hyper-params from config.py")

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
    model = Seq2Seq.from_hyper_parameters(encoder_embedding_size, decoder_embedding_size, hidden_size,
                                          num_layers, encoder_dropout, decoder_dropout).to(device)

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

        # For convenience, print the evaluation results for 10 random samples
        for i, sample_index in enumerate(indices):
            example = test_data.examples[sample_index]
            prediction = translate_sentence(model, example.src, srcField, trgField, device, max_length=50)

            if prediction[-1]=='<eos>': prediction = prediction[:-1]
            src_print, trg_print, pred_print = ''.join(example.src), ''.join(example.trg), ''.join(prediction)
            ed_print = eval_edit_distance(trg_print, pred_print)
            print(f"{i+1}. input: {src_print} ; gold: {trg_print} ; pred: {pred_print} ; ED = {ed_print}")

        # Evaluate the model on the entire test set
        edit_distance, accuracy = evaluate_model(test_data, model, srcField, trgField, device)
        writer.add_scalar("Test Accuracy", accuracy, global_step=epoch)
        print(f"avgED = {edit_distance}; avgAcc = {accuracy}\n")
        accs.append(accuracy)
        eds.append(edit_distance)

    # running on entire test data takes a while
    # score = evaluate_model(test_data[1:100], model, srcField, trgField, device)
    edit_distance, accuracy = evaluate_model(test_data, model, srcField, trgField, device)
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
results_df.to_excel(excel_results_file)
