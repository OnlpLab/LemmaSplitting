import codecs
import random
from collections import defaultdict
from os import getcwd, listdir, makedirs, scandir, walk
from os.path import isdir, join, relpath, splitext

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
    Takes 3 paths for the files, generates
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
    # test is returned twice, once for the gold data file and once for the incomplete file.
    return train, dev, test, test


def write_dataset(p, dataset, test_mode=False):
    """
    Takes the new dataset, and writes it to the given (new) path.
    """
    with open(p, mode='w', encoding='utf8') as f:
        for sample in dataset:
            lemma, form, feat = sample
            f.write(f"{lemma}\t{'' if test_mode else form}\t{feat}\n")


if __name__ == '__main__':
    print("Requirements: this script must be with the folders of SIGMORPHON (SURPRISE-LANG., GOLD-TEST, etc.).\n"
          "This script generates new datasets in the same structure of the old folders.")
    train_dirs, test_dir = ['DEVELOPMENT-LANGUAGES', 'SURPRISE-LANGUAGES'], 'GOLD-TEST'

    test_paths = listdir(test_dir)
    langs = [splitext(p)[0] for p in test_paths]
    test_paths = {l: p for l, p in zip(langs, [join(test_dir, s) for s in test_paths])}

    dev_families = [f.path for f in scandir(train_dirs[0]) if f.is_dir()]
    surprise_families = [f.path for f in scandir(train_dirs[1]) if f.is_dir()]

    dev_families.sort()
    surprise_families.sort()

    develop_paths, surprise_paths, test_no_gold_paths = {}, {}, {}
    for family in dev_families + surprise_families:
        for file in listdir(family):
            lang, ext = splitext(file)
            file = join(family, file)
            if ext == '.trn':
                develop_paths[lang] = file
            elif ext == '.dev':
                surprise_paths[lang] = file
            elif ext == '.tst':
                test_no_gold_paths[lang] = file
    # print(len(develop_paths))
    # print(len(surprise_paths))
    # print(len(test_no_gold_paths))

    print("Intersections between train, dev & test sets for each of the 90 languages:")
    for i, lang in enumerate(langs):
        result = check_lemma_split(develop_paths[lang], surprise_paths[lang], test_paths[lang])  # returns a triplet
        inter1, inter2, inter3 = ["Non-Empty" if e else "Empty" for e in result]
        print(f"{i + 1}. {lang} => {(inter1, inter2, inter3)}")
    print()

    print("Duplicating the original structure of the directories & sub-directories...\n")
    inputpath = getcwd()
    new_folder = "LEMMA-SPLIT"
    outputpath = join(inputpath, new_folder)

    # Create a new folder "LEMMA-SPLIT", and all the sub-folders like the original structure:
    for dirpath, dirnames, filenames in walk(inputpath):
        structure = join(outputpath, relpath(dirpath, inputpath))
        if not isdir(structure) and 'idea' not in structure and 'git' not in structure and join(
                new_folder, new_folder) not in structure:
            print(structure)
            makedirs(structure)
            print(f"Folder {structure}\tcreated.")
        else:
            print(f"Folder {structure}\texists.")
    print()

    print("Generating new datasets:")
    # For every language, add the "LEMMA-SPLIT" prefix, and write the new datasets (returned from generate_new_datasets) to the files in the corresponding folders
    # langs = ['aka'] # for debugging purposes
    for i, lang in enumerate(langs):
        paths = [develop_paths[lang], surprise_paths[lang], test_paths[lang], test_no_gold_paths[lang]]
        datasets = generate_new_datasets(*paths[:-1])  # returns 4 datasets - train, dev, test X 2
        new_paths = [join(new_folder, p) for p in paths]

        for j, (path, dataset) in enumerate(zip(new_paths, datasets)):
            write_dataset(path, dataset, test_mode=(j == 3))
        print(f"{i + 1}. Generated lemma-split datasets for {lang}")
