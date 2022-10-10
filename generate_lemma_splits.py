import codecs
import random
from collections import defaultdict
from os import listdir, scandir, mkdir
from os.path import join, splitext, split, isdir

import numpy as np


def read_all_lemmas(path):
    """
    Return a sorted list of all the lemmas in the file. No need to save the forms and features!
    """
    data = open(path, encoding='utf8').read().split('\n')
    lemmas = [line.split('\t')[0] for line in data if not line in {'', ' '}]
    lemmas.sort()
    return lemmas


def check_lemma_split(train_path, dev_path, test_path):
    """
    Apply read_all_lemmas 3 times to get the lists of all the lemmas. Then, observe the lists'
     intersections with each other. If they aren't empty, return False!!!
    """
    train_lemmas, dev_lemmas, test_lemmas = [read_all_lemmas(path) for path in [train_path, dev_path, test_path]]
    # lang = splitext(test_path)[0][-3:]
    inter1 = set(train_lemmas) & set(dev_lemmas)
    inter2 = set(train_lemmas) & set(test_lemmas)
    inter3 = set(dev_lemmas) & set(test_lemmas)
    return bool(inter1), bool(inter2), bool(inter3)


def read(fname):
    """ read file name """
    D = {}
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lemma, word, tag = line.split("\t")
            if lemma not in D:
                D[lemma] = {}
            D[lemma][tag] = word
    return D


def dict2lists(d):
    """
    Convert the given defaultdict object to a list of (lemma, form, feat) tuples.
    d is a list of tuples, where wach tuple has the form (lemma, list) where list has
    (at most?) 3 dicts with items of the form (feat, form).
    """
    samples_list = []
    for t in d:  # t is a tuple
        lemma, forms = t  # type(lemma)==str, forms is the list
        assert len(forms) in {1, 2, 3}
        # train_dict, dev_dict, test_dict = forms
        for dict_set in forms:  # train_dict, dev_dict, test_dict = forms
            for k, v in dict_set.items():
                samples_list.append((lemma, v, k))
    return samples_list


def generate_new_datasets(train, dev, test):
    """
    Takes 3 paths for the files, and generates new train, dev & test datasets, in lemma split.
    """
    ds = [read(train), read(dev), read(test)]
    total_d = defaultdict(list)
    # Unite all the forms of the same lemma under a defaultdict value entry
    for d in ds:
        for k, v in d.items():
            total_d[k].append(v)
    lemmas = list(total_d.items())
    n = len(lemmas)
    random.seed(1)
    random.shuffle(lemmas)
    # Now that we shuffled the data, we're ready to split it to 3 new sets, this time with absolute separation between the lemmas!
    train_prop, dev_prop, test_prop = 0.7, 0.2, 0.1
    assert np.isclose(sum([train_prop, dev_prop, test_prop]), 1, atol=1e-08)
    train = lemmas[:int(train_prop * n)]
    test = lemmas[int(train_prop * n): int((train_prop + dev_prop) * n)]
    dev = lemmas[int((train_prop + dev_prop) * n):]

    train, dev, test = dict2lists(train), dict2lists(dev), dict2lists(test)
    return train, dev, test


def write_dataset(p, dataset):
    """
    Takes the new dataset, and writes it to the given (new) path.
    """
    with open(p, mode='w', encoding='utf8') as f:
        for sample in dataset:
            lemma, form, feat = sample
            f.write(f"{lemma}\t{form}\t{feat}\n")


if __name__ == '__main__':
    form_split_data_folder, lemma_split_folder = join('DataExperiments', 'FormSplit'), join('DataExperiments', 'GeneratedLemmaSplits')
    train_dirs, test_dir = ['DEVELOPMENT-LANGUAGES', 'SURPRISE-LANGUAGES'], 'GOLD-TEST'
    train_dirs, test_dir = [join(form_split_data_folder, p) for p in train_dirs], join(form_split_data_folder, test_dir)

    langs = [splitext(p)[0] for p in listdir(test_dir)]
    test_paths_map = dict((splitext(test_file)[0], join(test_dir, test_file)) for test_file in listdir(test_dir))

    dev_families = [f.path for f in scandir(train_dirs[0]) if f.is_dir()]
    surprise_families = [f.path for f in scandir(train_dirs[1]) if f.is_dir()]

    dev_families.sort()
    surprise_families.sort()

    train_paths_map, dev_paths_map, lang2family = {}, {}, {}
    for family in dev_families + surprise_families:
        for file_name in listdir(family):
            lang, ext = splitext(file_name)
            file_name = join(family, file_name)
            if ext == '.trn':
                train_paths_map[lang] = file_name
            elif ext == '.dev':
                dev_paths_map[lang] = file_name
            # ignoring .tst files because they're covered. Using the GOLD-TEST files instead
            family_name = split(family)[1]
            if lang not in lang2family: lang2family[lang] = family_name.lower()

    print("Intersections between train, dev & test sets for each of the 90 languages:")
    for i, lang in enumerate(langs):
        result = check_lemma_split(train_paths_map[lang], dev_paths_map[lang], test_paths_map[lang])  # returns a triplet
        inter1, inter2, inter3 = ["Non-Empty" if e else "Empty" for e in result]
        print(f"{i + 1}. {lang} => {(inter1, inter2, inter3)}")

    print("\nGenerating new lemma-split datasets:")
    if not isdir(lemma_split_folder): mkdir(lemma_split_folder)
    for i, lang in enumerate(langs):
        family_subfolder = join(lemma_split_folder, lang2family[lang])
        if not isdir(family_subfolder): mkdir(family_subfolder)
        new_paths = [join(family_subfolder, f'{lang}.{extension}') for extension in ['trn', 'dev', 'tst']]

        datasets = generate_new_datasets(train_paths_map[lang], dev_paths_map[lang],
                                         test_paths_map[lang])  # returns train, dev & test datasets
        for path, dataset in zip(new_paths, datasets):
            write_dataset(path, dataset)
        print(f"{i + 1}. Completed for {lang} in {family_subfolder}")
